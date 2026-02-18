"""Quick test of optimized eye tracker"""
import cv2
import time
import sys

# Import the optimized tracker
from OrloskyPupilDetector import TrackerConfig, process_frame, get_darkest_area, apply_binary_threshold, mask_outside_square

print("Testing Optimized Eye Tracker")
print("=" * 60)

# Open camera with optimized settings
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TrackerConfig.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TrackerConfig.CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

print(f"Camera opened at {TrackerConfig.CAMERA_WIDTH}x{TrackerConfig.CAMERA_HEIGHT}")
print("Processing 30 frames...")
print()

frame_times = []
success_count = 0

for i in range(30):
    t_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i+1}: Failed to capture")
        continue
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Resize to expected dimensions
    if gray.shape != (TrackerConfig.CAMERA_HEIGHT, TrackerConfig.CAMERA_WIDTH):
        gray = cv2.resize(gray, (TrackerConfig.CAMERA_WIDTH, TrackerConfig.CAMERA_HEIGHT))
    
    # Test darkest area search
    try:
        darkest_point = get_darkest_area(gray)
        darkest_value = gray[darkest_point[1], darkest_point[0]]
        
        # Test thresholding
        thresh = apply_binary_threshold(gray, darkest_value, 15)
        thresh = mask_outside_square(thresh, darkest_point, TrackerConfig.MASK_SIZE)
        
        white_pixels = cv2.countNonZero(thresh)
        
        elapsed = time.time() - t_start
        frame_times.append(elapsed)
        
        if white_pixels > 100:
            success_count += 1
            status = "✓"
        else:
            status = "✗"
        
        if i < 10 or i % 10 == 0:
            print(f"Frame {i+1:2d}: {status} {elapsed*1000:6.1f}ms | Darkest: {darkest_point} (val={darkest_value}) | Pixels: {white_pixels}")
    
    except Exception as e:
        print(f"Frame {i+1}: ERROR - {e}")
        import traceback
        traceback.print_exc()
        break

cap.release()

print()
print("=" * 60)
print("RESULTS:")
print("=" * 60)

if frame_times:
    avg_time = sum(frame_times) / len(frame_times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    min_time = min(frame_times)
    max_time = max(frame_times)
    
    print(f"Frames processed: {len(frame_times)}/30")
    print(f"Success rate: {success_count}/{len(frame_times)} ({success_count/len(frame_times)*100:.0f}%)")
    print(f"Average time: {avg_time*1000:.1f}ms")
    print(f"Average FPS: {fps:.1f}")
    print(f"Min time: {min_time*1000:.1f}ms")
    print(f"Max time: {max_time*1000:.1f}ms")
    
    if fps < 20:
        print(f"\n⚠️ WARNING: FPS is still low ({fps:.1f})")
        print("Possible issues:")
        print("  - Camera capture is slow")
        print("  - Darkest area search needs more optimization")
        print("  - System is under heavy load")
    else:
        print(f"\n✅ GOOD: FPS is adequate ({fps:.1f})")
else:
    print("No frames processed successfully")
