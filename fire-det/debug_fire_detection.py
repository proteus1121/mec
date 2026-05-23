"""Debug script to check fire detection on a single image"""
import cv2
import numpy as np
from pathlib import Path

# Load test image
img_path = Path('xgboost/fire/images/test_10.jpg')
frame = cv2.imread(str(img_path))

print(f"Image shape: {frame.shape}")
print(f"Image dtype: {frame.dtype}")
print(f"Image min/max values: {frame.min()}, {frame.max()}")

# Resize if needed
if frame.shape != (480, 640, 3):
    frame = cv2.resize(frame, (640, 480))
    print(f"Resized to: {frame.shape}")

# Convert BGR to RGB
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Sample some pixels
print("\nSample pixels (RGB):")
for y in range(0, 480, 120):
    for x in range(0, 640, 160):
        r, g, b = frame_rgb[y, x]
        max_c = max(r, max(g, b))
        min_c = min(r, min(g, b))
        delta = max_c - min_c
        print(f"  [{y},{x}] R={r}, G={g}, B={b}, max={max_c}, delta={delta}, max>=180: {max_c >= 180}")

# Count pixels with brightness >= 180
bright_pixels = np.sum(np.max(frame_rgb, axis=2) >= 180)
print(f"\nBright pixels (>= 180): {bright_pixels} / {640*480}")
print(f"Bright pixel ratio: {bright_pixels / (640*480):.6f}")

# Analyze a specific bright region
bright_mask = np.max(frame_rgb, axis=2) >= 180
if np.any(bright_mask):
    bright_indices = np.where(bright_mask)
    sample_idx = 0
    y, x = bright_indices[0][sample_idx], bright_indices[1][sample_idx]
    r, g, b = frame_rgb[y, x]
    max_c = max(r, max(g, b))
    min_c = min(r, min(g, b))
    delta = max_c - min_c
    print(f"\nSample bright pixel [{y},{x}]: R={r}, G={g}, B={b}")
    print(f"  max={max_c}, min={min_c}, delta={delta}")
    if delta > 0:
        sat = (delta * 255) // max_c
        print(f"  saturation={sat}")
