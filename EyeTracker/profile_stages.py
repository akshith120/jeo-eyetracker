"""Detailed profiling of each stage"""
import cv2
import time
import numpy as np
from OrloskyPupilDetector import (TrackerConfig, get_darkest_area, apply_binary_threshold, 
                                   mask_outside_square, filter_contours_by_area_and_return_largest,
                                   check_ellipse_goodness, check_contour_pixels, optimize_contours_by_angle)

print("Detailed Stage Profiling")
print("=" * 70)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TrackerConfig.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TrackerConfig.CAMERA_HEIGHT)

# Timing accumulators
times = {
    'capture': [],
    'gray_convert': [],
    'darkest': [],
    'threshold': [],
    'mask': [],
    'dilate': [],
    'contours': [],
    'filter': [],
    'goodness': [],
    'total': []
}

for i in range(20):
    t_total = time.time()
    
    # Capture
    t = time.time()
    ret, frame = cap.read()
    times['capture'].append(time.time() - t)
    
    if not ret:
        continue
    
    # Gray conversion
    t = time.time()
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.resize(gray, (TrackerConfig.CAMERA_WIDTH, TrackerConfig.CAMERA_HEIGHT))
    times['gray_convert'].append(time.time() - t)
    
    # Darkest point
    t = time.time()
    darkest_point = get_darkest_area(gray)
    times['darkest'].append(time.time() - t)
    
    darkest_value = gray[darkest_point[1], darkest_point[0]]
    
    # Thresholding
    t = time.time()
    thresh_medium = apply_binary_threshold(gray, darkest_value, TrackerConfig.THRESHOLD_MEDIUM)
    times['threshold'].append(time.time() - t)
    
    # Masking
    t = time.time()
    thresh_medium = mask_outside_square(thresh_medium, darkest_point, TrackerConfig.MASK_SIZE)
    times['mask'].append(time.time() - t)
    
    # Dilation
    t = time.time()
    kernel = np.ones((TrackerConfig.KERNEL_SIZE, TrackerConfig.KERNEL_SIZE), np.uint8)
    dilated = cv2.dilate(thresh_medium, kernel, iterations=TrackerConfig.DILATION_ITERATIONS)
    times['dilate'].append(time.time() - t)
    
    # Find contours
    t = time.time()
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    times['contours'].append(time.time() - t)
    
    # Filter contours
    t = time.time()
    reduced = filter_contours_by_area_and_return_largest(contours)
    times['filter'].append(time.time() - t)
    
    # Check goodness
    if len(reduced) > 0 and len(reduced[0]) > TrackerConfig.MIN_POINTS_FOR_ELLIPSE:
        t = time.time()
        goodness = check_ellipse_goodness(dilated, reduced[0], False)
        times['goodness'].append(time.time() - t)
    else:
        times['goodness'].append(0)
    
    times['total'].append(time.time() - t_total)

cap.release()

print("\nStage Timings (averaged over 20 frames):")
print("-" * 70)

total_accounted = 0
for stage in ['capture', 'gray_convert', 'darkest', 'threshold', 'mask', 'dilate', 'contours', 'filter', 'goodness']:
    avg = sum(times[stage]) / len(times[stage]) * 1000
    pct = (sum(times[stage]) / sum(times['total'])) * 100 if sum(times['total']) > 0 else 0
    total_accounted += avg
    print(f"{stage:15s}: {avg:6.2f}ms  ({pct:5.1f}%)")

total_avg = sum(times['total']) / len(times['total']) * 1000
print("-" * 70)
print(f"{'Total':15s}: {total_avg:6.2f}ms")
print(f"{'Accounted for':15s}: {total_accounted:6.2f}ms")
print(f"{'Unaccounted':15s}: {total_avg - total_accounted:6.2f}ms (overhead)")

fps = 1000 / total_avg if total_avg > 0 else 0
print(f"\nExpected FPS: {fps:.1f}")

print("\n" + "=" * 70)
print("BOTTLENECK ANALYSIS:")
print("=" * 70)

# Find top 3 slowest stages
stage_avgs = [(stage, sum(times[stage]) / len(times[stage]) * 1000) 
              for stage in times.keys() if stage != 'total']
stage_avgs.sort(key=lambda x: x[1], reverse=True)

for i, (stage, avg_time) in enumerate(stage_avgs[:3], 1):
    pct = (sum(times[stage]) / sum(times['total'])) * 100
    print(f"{i}. {stage:15s}: {avg_time:6.2f}ms ({pct:5.1f}% of total)")

print("\nRECOMMENDATIONS:")
if stage_avgs[0][0] == 'capture':
    print("• Camera capture is the bottleneck - this is hardware limited")
    print("• Other optimizations have minimal impact if capture is slow")
elif stage_avgs[0][0] == 'darkest':
    print("• Darkest area search needs more optimization")
    print("• Consider: larger skip size, or use optical flow for tracking")
elif stage_avgs[0][0] == 'contours':
    print("• Contour finding is slow - reduce image size or use simpler methods")
