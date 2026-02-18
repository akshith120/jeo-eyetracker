import cv2
import numpy as np
import time
import tkinter as tk
import os
from tkinter import filedialog

# Configuration class for all tunable parameters
class TrackerConfig:
    CAMERA_WIDTH = 320
    CAMERA_HEIGHT = 240
    FORCE_GRAYSCALE = True
    ROI_EXPAND_FACTOR = 1.3
    ROI_MIN_SIZE = 60
    FACE_DETECTION_INTERVAL = 10
    FACE_MIN_SIZE = (60, 60)
    EYE_MIN_SIZE = (20, 20)
    IGNORE_BOUNDS = 10
    IMAGE_SKIP_SIZE = 8
    SEARCH_AREA = 15
    INTERNAL_SKIP_SIZE = 4
    THRESHOLD_STRICT = 5
    THRESHOLD_MEDIUM = 15
    THRESHOLD_RELAXED = 25
    MASK_SIZE = 150
    MIN_CONTOUR_AREA = 50
    MAX_CONTOUR_AREA = 5000
    MAX_ASPECT_RATIO = 3.0
    MIN_CIRCULARITY = 0.4
    MIN_POINTS_FOR_ELLIPSE = 5
    KERNEL_SIZE = 3
    DILATION_ITERATIONS = 1
    DRAW_ONLY_ESSENTIALS = True
    TIMING_LOG_INTERVAL = 60
    ENABLE_TIMING = True

class TimingStats:
    def __init__(self):
        self.reset()
    def reset(self):
        self.capture_time = []
        self.roi_time = []
        self.preprocess_time = []
        self.contour_time = []
        self.viz_time = []
        self.total_time = []
    def add(self, stage, duration):
        getattr(self, f"{stage}_time").append(duration)
    def get_averages(self):
        stages = ['capture', 'roi', 'preprocess', 'contour', 'viz', 'total']
        avgs = {}
        for stage in stages:
            times = getattr(self, f"{stage}_time")
            avgs[stage] = sum(times) / len(times) if times else 0
        return avgs

timing_stats = TimingStats()

# Initialize face and eye cascade classifiers
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    FACE_DETECTION_AVAILABLE = True
except:
    FACE_DETECTION_AVAILABLE = False

def crop_to_aspect_ratio(image, width=None, height=None):
    if width is None:
        width = TrackerConfig.CAMERA_WIDTH
    if height is None:
        height = TrackerConfig.CAMERA_HEIGHT
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height
    if current_ratio > desired_ratio:
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset+new_width]
    else:
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset+new_height, :]
    return cv2.resize(cropped_img, (width, height))

#apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    # Calculate the threshold as the sum of the two input values
    threshold = darkestPixelValue + addedThreshold
    # Apply the binary threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    return thresholded_image

def detect_face_and_eyes(frame, draw_rectangles=False):
    if not FACE_DETECTION_AVAILABLE:
        return None
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=TrackerConfig.FACE_MIN_SIZE)
    if len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    largest_idx = np.argmax(areas)
    fx, fy, fw, fh = faces[largest_idx]
    
    if draw_rectangles:
        if len(frame.shape) == 2:
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame
        cv2.rectangle(frame_color, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
    face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=TrackerConfig.EYE_MIN_SIZE)
    
    if len(eyes) == 0:
        return None
    
    eye_regions = []
    expand = TrackerConfig.ROI_EXPAND_FACTOR
    for (ex, ey, ew, eh) in eyes:
        abs_x = fx + ex
        abs_y = fy + ey
        center_x = abs_x + ew // 2
        center_y = abs_y + eh // 2
        new_w = int(ew * expand)
        new_h = int(eh * expand)
        new_x = max(0, center_x - new_w // 2)
        new_y = max(0, center_y - new_h // 2)
        new_x2 = min(frame.shape[1], new_x + new_w)
        new_y2 = min(frame.shape[0], new_y + new_h)
        if draw_rectangles and 'frame_color' in locals():
            cv2.rectangle(frame_color, (new_x, new_y), (new_x2, new_y2), (0, 255, 0), 2)
        eye_regions.append((new_x, new_y, new_x2 - new_x, new_y2 - new_y))
    return eye_regions if len(eye_regions) > 0 else None

def get_darkest_area(gray_image):
    cfg = TrackerConfig
    h, w = gray_image.shape
    y_samples = np.arange(cfg.IGNORE_BOUNDS, h - cfg.IGNORE_BOUNDS, cfg.IMAGE_SKIP_SIZE)
    x_samples = np.arange(cfg.IGNORE_BOUNDS, w - cfg.IGNORE_BOUNDS, cfg.IMAGE_SKIP_SIZE)
    min_sum = float('inf')
    darkest_point = (w // 2, h // 2)
    for y in y_samples:
        for x in x_samples:
            y_end = min(y + cfg.SEARCH_AREA, h)
            x_end = min(x + cfg.SEARCH_AREA, w)
            region = gray_image[y:y_end:cfg.INTERNAL_SKIP_SIZE, x:x_end:cfg.INTERNAL_SKIP_SIZE]
            current_sum = np.sum(region, dtype=np.int64)
            if current_sum < min_sum:
                min_sum = current_sum
                darkest_point = (x + cfg.SEARCH_AREA // 2, y + cfg.SEARCH_AREA // 2)
    return darkest_point

#mask all pixels outside a square defined by center and size
def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    # Create a mask initialized to black
    mask = np.zeros_like(image)

    # Calculate the top-left corner of the square
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)

    # Calculate the bottom-right corner of the square
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)

    # Set the square area in the mask to white
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
   
def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

def filter_contours_by_area_and_return_largest(contours):
    cfg = TrackerConfig
    if len(contours) == 0:
        return []
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < cfg.MIN_CONTOUR_AREA or area > cfg.MAX_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio > cfg.MAX_ASPECT_RATIO:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < cfg.MIN_CIRCULARITY:
            continue
        valid_contours.append((area, contour))
    if not valid_contours:
        return []
    largest = max(valid_contours, key=lambda x: x[0])
    return [largest[1]]

def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= TrackerConfig.MIN_POINTS_FOR_ELLIPSE:
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image, ellipse, color, 2)
        return image
    else:
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0, 0, 0]
    if len(contour) < TrackerConfig.MIN_POINTS_FOR_ELLIPSE:
        return ellipse_goodness
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    if ellipse_area == 0:
        return ellipse_goodness
    ellipse_goodness[0] = covered_pixels / ellipse_area
    axes_lengths = ellipse[1]
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    if major_axis_length > 0 and minor_axis_length > 0:
        ellipse_goodness[2] = min(major_axis_length/minor_axis_length, minor_axis_length/major_axis_length)
    
    return ellipse_goodness

def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    t_start = time.time() if TrackerConfig.ENABLE_TIMING else 0
    cfg = TrackerConfig
    final_rotated_rect = ((0,0),(0,0),0)
    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict]
    final_contours = []
    goodness = 0
    kernel = np.ones((cfg.KERNEL_SIZE, cfg.KERNEL_SIZE), np.uint8)
    for i in range(1, 4):
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=cfg.DILATION_ITERATIONS)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced_contours = filter_contours_by_area_and_return_largest(contours)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > cfg.MIN_POINTS_FOR_ELLIPSE:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)
            final_goodness = current_goodness[0] * total_pixels[0] * total_pixels[0] * total_pixels[1]
            if debug_mode_on:
                gray_copy = gray_frame.copy()
                cv2.ellipse(gray_copy, ellipse, (255, 0, 0), 2)
                cv2.putText(gray_copy, f"final: {final_goodness:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.imshow(f"threshold_{i}", gray_copy)
            if final_goodness > goodness:
                goodness = final_goodness
                final_contours = reduced_contours
    if TrackerConfig.ENABLE_TIMING:
        timing_stats.add('contour', time.time() - t_start)
    
    if render_cv_window or debug_mode_on:
        t_viz = time.time() if TrackerConfig.ENABLE_TIMING else 0
        test_frame = frame.copy()
        final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
        if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) > cfg.MIN_POINTS_FOR_ELLIPSE:
            ellipse = cv2.fitEllipse(final_contours[0])
            final_rotated_rect = ellipse
            if cfg.DRAW_ONLY_ESSENTIALS:
                cv2.ellipse(test_frame, ellipse, (55, 255, 0), 2)
                center_x, center_y = map(int, ellipse[0])
                cv2.circle(test_frame, (center_x, center_y), 3, (255, 255, 0), -1)
            cv2.putText(test_frame, "SPACE=pause Q=quit D=debug", (10, test_frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,90,30), 1)
        if render_cv_window:
            cv2.imshow('Pupil Tracker', test_frame)
        if TrackerConfig.ENABLE_TIMING:
            timing_stats.add('viz', time.time() - t_viz)
    return final_rotated_rect


# Finds the pupil in an individual frame and returns the center point
def process_frame(frame, use_face_detection=True, draw_debug=False):
    
    original_frame = frame.copy()
    eye_offset = (0, 0)  # Track offset for coordinate conversion
    
    # Try to detect face and eyes first
    if use_face_detection and FACE_DETECTION_AVAILABLE:
        eye_regions = detect_face_and_eyes(frame, draw_rectangles=draw_debug)
        
        if eye_regions is not None and len(eye_regions) > 0:
            # Use the first (or largest) eye region
            ex, ey, ew, eh = eye_regions[0]
            
            # Crop to eye region
            frame = frame[ey:ey+eh, ex:ex+ew]
            eye_offset = (ex, ey)
            
            if draw_debug:
                print(f"Processing eye region at ({ex}, {ey}) size ({ew}x{eh})")
        else:
            if draw_debug:
                print("No face/eyes detected, processing full frame")
            # Fall back to processing full frame
            pass
    
    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    
    # Convert coordinates back to original frame if we cropped to eye region
    if eye_offset != (0, 0) and final_rotated_rect is not None:
        # Adjust ellipse center coordinates
        # Note: This is a simplified adjustment, you may need to scale as well
        pass
    
    return final_rotated_rect

def process_video(video_path, input_method):
    cfg = TrackerConfig
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('C:/Storage/Source Videos/output_video.mp4', fourcc, 30.0, (cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT))
    if input_method == 1:
        cap = cv2.VideoCapture(video_path)
    elif input_method == 2:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        if cfg.FORCE_GRAYSCALE:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    else:
        return
    if not cap.isOpened():
        return
    debug_mode_on = False
    use_face_detection = True
    frame_count = 0
    cached_roi = None
    timing_stats.reset()
    while True:
        t_frame_start = time.time() if cfg.ENABLE_TIMING else 0
        t_cap = time.time() if cfg.ENABLE_TIMING else 0
        ret, frame = cap.read()
        if not ret:
            break
        if cfg.ENABLE_TIMING:
            timing_stats.add('capture', time.time() - t_cap)
        t_roi = time.time() if cfg.ENABLE_TIMING else 0
        if cfg.FORCE_GRAYSCALE and len(frame.shape) == 3:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 2:
            gray_full = frame
        else:
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_offset = (0, 0)
        current_roi = None
        if use_face_detection and FACE_DETECTION_AVAILABLE:
            if frame_count % cfg.FACE_DETECTION_INTERVAL == 0:
                eye_regions = detect_face_and_eyes(gray_full, draw_rectangles=debug_mode_on)
                if eye_regions is not None and len(eye_regions) > 0:
                    cached_roi = eye_regions[0]
                    current_roi = cached_roi
                else:
                    cached_roi = None
            else:
                current_roi = cached_roi
            if current_roi is not None:
                ex, ey, ew, eh = current_roi
                ex = max(0, ex)
                ey = max(0, ey)
                ew = min(ew, gray_full.shape[1] - ex)
                eh = min(eh, gray_full.shape[0] - ey)
                if ew >= cfg.ROI_MIN_SIZE and eh >= cfg.ROI_MIN_SIZE:
                    gray_frame = gray_full[ey:ey+eh, ex:ex+ew]
                    eye_offset = (ex, ey)
                else:
                    gray_frame = gray_full
            else:
                gray_frame = gray_full
        else:
            gray_frame = gray_full
        if cfg.ENABLE_TIMING:
            timing_stats.add('roi', time.time() - t_roi)
        t_prep = time.time() if cfg.ENABLE_TIMING else 0
        gray_frame = crop_to_aspect_ratio(gray_frame, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT)
        darkest_point = get_darkest_area(gray_frame)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, cfg.THRESHOLD_STRICT)
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, cfg.MASK_SIZE)
        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, cfg.THRESHOLD_MEDIUM)
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, cfg.MASK_SIZE)
        thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, cfg.THRESHOLD_RELAXED)
        thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, cfg.MASK_SIZE)
        if cfg.ENABLE_TIMING:
            timing_stats.add('preprocess', time.time() - t_prep)
        if len(gray_frame.shape) == 2:
            frame_vis = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_vis = gray_frame
        pupil_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame_vis, gray_frame, darkest_point, debug_mode_on, True)
        if cfg.ENABLE_TIMING:
            timing_stats.add('total', time.time() - t_frame_start)
        if cfg.ENABLE_TIMING and frame_count > 0 and frame_count % cfg.TIMING_LOG_INTERVAL == 0:
            avgs = timing_stats.get_averages()
            fps = 1.0 / avgs['total'] if avgs['total'] > 0 else 0
            print(f"[Frame {frame_count}] FPS: {fps:.1f} | Capture: {avgs['capture']*1000:.1f}ms | ROI: {avgs['roi']*1000:.1f}ms | Preprocess: {avgs['preprocess']*1000:.1f}ms | Contour: {avgs['contour']*1000:.1f}ms | Viz: {avgs['viz']*1000:.1f}ms")
            timing_stats.reset()
        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            debug_mode_on = not debug_mode_on
            if not debug_mode_on:
                cv2.destroyAllWindows()
        elif key == ord('f'):
            use_face_detection = not use_face_detection
        elif key == ord('q'):
            out.release()
            break
        elif key == ord(' '):
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('q'):
                    break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

#Prompts the user to select input source (camera or video file)
def select_video():
    root = tk.Tk()
    root.title("Select Input Source")
    root.geometry("400x250")
    
    tk.Label(root, text="Eye Tracker Input Selection", font=("Arial", 14, "bold")).pack(pady=20)
    
    def start_camera():
        root.destroy()
        process_video(None, 2)  # input_method=2 for camera
    
    def start_video():
        video_path = 'C:/Google Drive/Eye Tracking/fulleyetest.mp4'
        if not os.path.exists(video_path):
            video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi")])
        if video_path:
            root.destroy()
            process_video(video_path, 1)  # input_method=1 for video file
        else:
            print("No file selected.")
    
    # Create buttons for camera and video file
    tk.Button(root, text="Start Camera (Webcam)", command=start_camera, 
              width=25, height=2, bg="#4CAF50", fg="white", font=("Arial", 11)).pack(pady=10)
    
    tk.Button(root, text="Browse Video File", command=start_video, 
              width=25, height=2, bg="#2196F3", fg="white", font=("Arial", 11)).pack(pady=10)
    
    tk.Button(root, text="Exit", command=root.destroy, 
              width=25, height=2, bg="#f44336", fg="white", font=("Arial", 11)).pack(pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    select_video()


