"""Test complete pipeline with all features enabled"""
import cv2
import time
from OrloskyPupilDetector import TrackerConfig, timing_stats

print("Testing COMPLETE Pipeline (with face detection & contour processing)")
print("=" * 70)

# Simulate the actual process_video loop
cfg = TrackerConfig
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

print(f"Camera: {cfg.CAMERA_WIDTH}x{cfg.CAMERA_HEIGHT}")
print(f"Face detection interval: every {cfg.FACE_DETECTION_INTERVAL} frames")
print(f"Processing 60 frames...\n")

from OrloskyPupilDetector import (detect_face_and_eyes, crop_to_aspect_ratio, get_darkest_area,
                                   apply_binary_threshold, mask_outside_square, process_frames)

frame_count = 0
cached_roi = None
use_face_detection = True
FACE_DETECTION_AVAILABLE = True

frame_times = []
detection_frames = []
non_detection_frames = []

for i in range(60):
    t_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_full = frame
    
    current_roi = None
    
    # Face detection logic
    if use_face_detection and FACE_DETECTION_AVAILABLE:
        if frame_count % cfg.FACE_DETECTION_INTERVAL == 0:
            eye_regions = detect_face_and_eyes(gray_full, draw_rectangles=False)
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
            else:
                gray_frame = gray_full
        else:
            gray_frame = gray_full
    else:
        gray_frame = gray_full
    
    # Preprocessing
    gray_frame = crop_to_aspect_ratio(gray_frame, cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT)
    darkest_point = get_darkest_area(gray_frame)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # Thresholding
    thresholded_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, cfg.THRESHOLD_STRICT)
    thresholded_strict = mask_outside_square(thresholded_strict, darkest_point, cfg.MASK_SIZE)
    
    thresholded_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, cfg.THRESHOLD_MEDIUM)
    thresholded_medium = mask_outside_square(thresholded_medium, darkest_point, cfg.MASK_SIZE)
    
    thresholded_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, cfg.THRESHOLD_RELAXED)
    thresholded_relaxed = mask_outside_square(thresholded_relaxed, darkest_point, cfg.MASK_SIZE)
    
    # Convert for visualization
    if len(gray_frame.shape) == 2:
        frame_vis = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_vis = gray_frame
    
    # Process frames (contour finding, ellipse fitting)
    pupil_rotated_rect = process_frames(thresholded_strict, thresholded_medium, thresholded_relaxed,
                                        frame_vis, gray_frame, darkest_point, False, False)
    
    elapsed = time.time() - t_start
    frame_times.append(elapsed)
    
    is_detection_frame = (frame_count % cfg.FACE_DETECTION_INTERVAL == 0)
    if is_detection_frame:
        detection_frames.append(elapsed)
    else:
        non_detection_frames.append(elapsed)
    
    if i < 5 or i == 10 or i == 20 or i == 30 or i == 40 or i == 50 or i == 59:
        det_marker = "[DET]" if is_detection_frame else "     "
        pupil_found = "✓" if pupil_rotated_rect and pupil_rotated_rect[0] != (0, 0) else "✗"
        print(f"Frame {frame_count:2d}: {det_marker} {pupil_found} {elapsed*1000:6.1f}ms")
    
    frame_count += 1

cap.release()

print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)

if frame_times:
    avg_time = sum(frame_times) / len(frame_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    print(f"Total frames: {len(frame_times)}")
    print(f"Average time: {avg_time*1000:.1f}ms")
    print(f"Average FPS: {fps:.1f}")
    print(f"Min: {min(frame_times)*1000:.1f}ms | Max: {max(frame_times)*1000:.1f}ms")
    
    if detection_frames:
        avg_det = sum(detection_frames) / len(detection_frames)
        fps_det = 1.0 / avg_det
        print(f"\nDetection frames ({len(detection_frames)}): {avg_det*1000:.1f}ms ({fps_det:.1f} FPS)")
    
    if non_detection_frames:
        avg_non = sum(non_detection_frames) / len(non_detection_frames)
        fps_non = 1.0 / avg_non
        print(f"Non-detection frames ({len(non_detection_frames)}): {avg_non*1000:.1f}ms ({fps_non:.1f} FPS)")
    
    print("\n" + "=" * 70)
    if fps >= 25:
        print("✅ EXCELLENT: Real-time performance achieved (25+ FPS)")
    elif fps >= 20:
        print("✅ GOOD: Smooth performance (20-25 FPS)")
    elif fps >= 15:
        print("⚠️  ACCEPTABLE: Usable but could be smoother (15-20 FPS)")
    else:
        print("❌ POOR: Below acceptable threshold (< 15 FPS)")
        print("   Likely limited by camera hardware")
