"""
iPhone/Android as Webcam Pipeline for Eye Tracking
===================================================
Requirements:
1. Install Camo (iOS) or DroidCam (Android) on your phone
2. Connect phone to laptop via USB or WiFi
3. Phone camera will appear as a webcam device

Camo: https://reincubate.com/camo/
DroidCam: https://www.dev47apps.com/

Alternative: IP Webcam (Android) or EpocCam (iOS)
"""

import cv2
import numpy as np
import time
import sys
import os

# Import functions from OrloskyPupilDetector
sys.path.append(os.path.dirname(__file__))
from OrloskyPupilDetector import (
    crop_to_aspect_ratio,
    get_darkest_area,
    apply_binary_threshold,
    mask_outside_square,
    filter_contours_by_area_and_return_largest,
    optimize_contours_by_angle,
    process_frames
)

# Initialize face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_camera_index():
    """Auto-detect which camera index is the phone (Camo/DroidCam)"""
    print("\nDetecting available cameras (this may take a moment)...\n")
    available_cameras = []
    
    # Test multiple backends and indices
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for backend_id, backend_name in backends:
        print(f"Testing {backend_name} backend...")
        for i in range(10):  # Check indices 0-9
            try:
                cap = cv2.VideoCapture(i, backend_id)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        # Get camera name if possible
                        backend_info = f"{backend_name} (backend={backend_id})"
                        camera_info = f"Index {i}: {width}x{height} - {backend_info}"
                        
                        # Try to identify Camo (usually has "Camo" in the name or specific resolution)
                        if width >= 1280:  # Camo typically supports HD or higher
                            camera_info += " ** LIKELY EXTERNAL/PHONE CAMERA **"
                        
                        if (i, backend_id) not in [(c[0], c[1]) for c in available_cameras]:
                            available_cameras.append((i, backend_id, camera_info))
                            print(f"  ✓ {camera_info}")
                    cap.release()
            except:
                pass
    
    if not available_cameras:
        print("No cameras detected!")
        return 0, cv2.CAP_DSHOW
    
    print(f"\n{len(available_cameras)} camera(s) detected.")
    print("\nTip: Camo usually appears with higher resolution (1280x720 or 1920x1080)")
    print("     Integrated laptop cameras are often 640x480 or 1280x720")
    
    # Show preview option
    preview = input("\nPreview cameras before selecting? (y/n, default y): ").strip().lower()
    if preview != 'n':
        for idx, (cam_idx, backend, info) in enumerate(available_cameras):
            print(f"\nPreviewing: {info}")
            cap = cv2.VideoCapture(cam_idx, backend)
            if cap.isOpened():
                print("Press any key to continue to next camera, or 'S' to select this one...")
                for _ in range(30):  # Show 30 frames
                    ret, frame = cap.read()
                    if ret:
                        # Add text overlay
                        cv2.putText(frame, f"Camera {cam_idx} - Press 'S' to select, any other key to skip", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow(f'Camera Preview - Index {cam_idx}', frame)
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord('s') or key == ord('S'):
                            cv2.destroyAllWindows()
                            cap.release()
                            print(f"\n✓ Selected camera {cam_idx}")
                            return cam_idx, backend
                        elif key != 255:  # Any other key pressed
                            break
                cap.release()
                cv2.destroyAllWindows()
    
    # Manual selection
    print("\nAvailable cameras:")
    for idx, (cam_idx, backend, info) in enumerate(available_cameras):
        print(f"  {idx}: {info}")
    
    selection = input(f"\nEnter selection (0-{len(available_cameras)-1}, default 0): ").strip()
    selected_idx = int(selection) if selection.isdigit() else 0
    
    if 0 <= selected_idx < len(available_cameras):
        cam_idx, backend, info = available_cameras[selected_idx]
        print(f"\n✓ Selected: {info}")
        return cam_idx, backend
    else:
        print(f"\n✓ Using default: {available_cameras[0][2]}")
        return available_cameras[0][0], available_cameras[0][1]

def process_eye_roi(eye_frame, debug=False):
    """Process eye region through pupil detection pipeline"""
    try:
        # Ensure frame is color (3 channels)
        if len(eye_frame.shape) == 2:
            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_GRAY2BGR)
        
        # Crop to aspect ratio
        processed_frame = crop_to_aspect_ratio(eye_frame, 640, 480)
        
        # Find darkest point (pupil candidate)
        darkest_point = get_darkest_area(processed_frame)
        if darkest_point is None:
            return None, None
        
        # Convert to grayscale for thresholding
        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
        
        # Apply multiple threshold levels
        thresholded_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)
        thresholded_strict = mask_outside_square(thresholded_strict, darkest_point, 250)
        
        thresholded_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
        thresholded_medium = mask_outside_square(thresholded_medium, darkest_point, 250)
        
        thresholded_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)
        thresholded_relaxed = mask_outside_square(thresholded_relaxed, darkest_point, 250)
        
        # Process frames and get pupil ellipse
        pupil_ellipse = process_frames(
            thresholded_strict, thresholded_medium, thresholded_relaxed,
            processed_frame, gray_frame, darkest_point,
            debug_mode_on=debug, render_cv_window=False
        )
        
        return pupil_ellipse, processed_frame
    
    except Exception as e:
        if debug:
            print(f"Error processing eye ROI: {e}")
        return None, None

def run_phone_camera_pipeline(camera_index=0, backend=cv2.CAP_DSHOW, use_ip_camera=False, ip_url=None):
    """
    Main pipeline for phone camera eye tracking
    
    Args:
        camera_index: Camera device index (0 for default, 1+ for external)
        backend: OpenCV backend to use (CAP_DSHOW, CAP_MSMF, etc.)
        use_ip_camera: Set True if using IP Webcam app
        ip_url: URL for IP camera (e.g., 'http://192.168.1.100:8080/video')
    """
    
    # Open camera
    if use_ip_camera and ip_url:
        print(f"Connecting to IP camera: {ip_url}")
        cap = cv2.VideoCapture(ip_url)
    else:
        print(f"Opening camera index {camera_index} with backend {backend}...")
        cap = cv2.VideoCapture(camera_index, backend)
        
        # Optimize camera settings for eye tracking
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try higher res first
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify actual settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Camera opened at {actual_width}x{actual_height} @ {actual_fps} FPS")
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera opened successfully!")
    print("\nControls:")
    print("  Q - Quit")
    print("  D - Toggle debug mode")
    print("  F - Toggle face detection")
    print("  SPACE - Pause/Resume")
    print("  S - Save current frame")
    print("\nStarting eye tracking...\n")
    
    debug_mode = False
    use_face_detection = True
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        original_frame = frame.copy()
        pupil_detected = False
        
        # FPS calculation
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if use_face_detection:
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            if len(faces) > 0:
                # Use largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                fx, fy, fw, fh = largest_face
                
                # Draw face rectangle
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Detect eyes within face ROI
                face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
                face_roi_color = frame[fy:fy+fh, fx:fx+fw]
                
                eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.05, minNeighbors=8, minSize=(30, 30))
                
                for (ex, ey, ew, eh) in eyes:
                    # Convert to absolute coordinates
                    abs_ex = fx + ex
                    abs_ey = fy + ey
                    
                    # Expand eye region for better pupil detection
                    expand = 1.4
                    center_x = abs_ex + ew // 2
                    center_y = abs_ey + eh // 2
                    new_w = int(ew * expand)
                    new_h = int(eh * expand)
                    new_x = max(0, center_x - new_w // 2)
                    new_y = max(0, center_y - new_h // 2)
                    new_x2 = min(frame.shape[1], new_x + new_w)
                    new_y2 = min(frame.shape[0], new_y + new_h)
                    
                    # Draw eye rectangle
                    cv2.rectangle(frame, (new_x, new_y), (new_x2, new_y2), (0, 255, 0), 2)
                    
                    # Extract eye ROI
                    eye_roi = original_frame[new_y:new_y2, new_x:new_x2]
                    
                    if eye_roi.size > 0:
                        # Process eye through pupil detection
                        pupil_ellipse, processed_eye = process_eye_roi(eye_roi, debug=debug_mode)
                        
                        if pupil_ellipse is not None:
                            pupil_detected = True
                            # Extract ellipse parameters
                            (cx, cy), (major, minor), angle = pupil_ellipse
                            
                            # Convert coordinates back to original frame
                            scale_x = (new_x2 - new_x) / 640.0
                            scale_y = (new_y2 - new_y) / 480.0
                            abs_cx = int(new_x + cx * scale_x)
                            abs_cy = int(new_y + cy * scale_y)
                            abs_major = int(major * scale_x)
                            abs_minor = int(minor * scale_y)
                            
                            # Draw pupil on main frame
                            cv2.ellipse(frame, (abs_cx, abs_cy), (abs_major, abs_minor), 
                                       angle, 0, 360, (0, 255, 255), 2)
                            cv2.circle(frame, (abs_cx, abs_cy), 5, (255, 0, 255), -1)
                            cv2.putText(frame, f"Pupil", (abs_cx + 10, abs_cy - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
                            # Show processed eye in debug mode
                            if debug_mode and processed_eye is not None:
                                cv2.imshow('Processed Eye', processed_eye)
        
        # Status display
        status_color = (0, 255, 0) if pupil_detected else (0, 0, 255)
        status_text = "TRACKING" if pupil_detected else "SEARCHING"
        cv2.putText(frame, f"Status: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Face Detection: {'ON' if use_face_detection else 'OFF'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('iPhone/Android Eye Tracker', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('f'):
            use_face_detection = not use_face_detection
            print(f"Face detection: {'ON' if use_face_detection else 'OFF'}")
        elif key == ord(' '):
            print("Paused. Press SPACE to resume...")
            cv2.waitKey(0)
        elif key == ord('s'):
            filename = f'eye_tracking_capture_{int(time.time())}.png'
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nEye tracking stopped.")

def main():
    print("=" * 60)
    print("iPhone/Android Camera Eye Tracking Pipeline")
    print("=" * 60)
    print("\nSetup Instructions:")
    print("1. Install camera app on phone (Camo, DroidCam, IP Webcam, etc.)")
    print("2. Connect phone to laptop (USB or WiFi)")
    print("3. Start the camera app on your phone")
    print("4. Select camera input method below\n")
    
    print("Select input method:")
    print("1. USB/WiFi Camera (Camo/DroidCam as webcam)")
    print("2. IP Camera (IP Webcam app)")
    print("3. Auto-detect camera")
    
    choice = input("\nEnter choice (1-3, default 1): ").strip() or "1"
    
    if choice == "1":
        camera_idx = input("Enter camera index (default 0): ").strip()
        camera_idx = int(camera_idx) if camera_idx else 0
        
        # Ask for backend
        print("\nSelect backend:")
        print("1. DirectShow (DSHOW) - Default for Windows")
        print("2. Media Foundation (MSMF) - Alternative for Windows")
        print("3. Auto")
        backend_choice = input("Enter choice (1-3, default 1): ").strip() or "1"
        
        backend_map = {
            "1": cv2.CAP_DSHOW,
            "2": cv2.CAP_MSMF,
            "3": cv2.CAP_ANY
        }
        backend = backend_map.get(backend_choice, cv2.CAP_DSHOW)
        
        run_phone_camera_pipeline(camera_index=camera_idx, backend=backend)
    
    elif choice == "2":
        ip_url = input("Enter IP camera URL (e.g., http://192.168.1.100:8080/video): ").strip()
        if ip_url:
            run_phone_camera_pipeline(use_ip_camera=True, ip_url=ip_url)
        else:
            print("Invalid URL")
    
    elif choice == "3":
        camera_idx, backend = detect_camera_index()
        run_phone_camera_pipeline(camera_index=camera_idx, backend=backend)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
