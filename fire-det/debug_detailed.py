"""Detailed debug of fire detection algorithm"""
import cv2
import numpy as np
from pathlib import Path

class FireDetector:
    FIRE_PIXEL_RATIO = 0.004
    MIN_VALUE = 180
    MIN_SATURATION = 80
    MIN_RED_DOMINANCE = 30
    FLICKER_THRESH = 0.000005
    CONFIRM_FRAMES = 3
    HIST_LEN = 8
    
    @staticmethod
    def calc_hue(r, g, b, max_c, delta):
        if delta == 0:
            return 0
        if max_c == r:
            hue = 30 * (g - b) / delta
            if hue < 0:
                hue += 180
        elif max_c == g:
            hue = 30 * (b - r) / delta + 60
        else:
            hue = 30 * (r - g) / delta + 120
        return int(hue)
    
    @staticmethod
    def hue_is_flame(hue):
        return (hue <= 18) or (hue >= 162) or (19 <= hue <= 50)

# Load image
img_path = Path('xgboost/fire/images/test_10.jpg')
frame = cv2.imread(str(img_path))
frame = cv2.resize(frame, (640, 480))
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

print("Analyzing image: test_10.jpg")
print("="*70)

# Find pixels with max brightness >= 180
fire_candidates = []
for y in range(0, 480, 4):
    for x in range(0, 640, 4):
        r, g, b = int(frame_rgb[y, x, 0]), int(frame_rgb[y, x, 1]), int(frame_rgb[y, x, 2])
        max_c = max(r, max(g, b))
        min_c = min(r, min(g, b))
        delta = max_c - min_c
        
        if max_c >= FireDetector.MIN_VALUE:
            # Check saturation
            sat = int((delta / max_c) * 255) if max_c > 0 else 0
            
            # Check red dominance
            red_ok = (r >= b + FireDetector.MIN_RED_DOMINANCE)
            green_ok = (g <= r + 20)
            
            # Check hue
            hue = FireDetector.calc_hue(r, g, b, max_c, delta)
            hue_ok = FireDetector.hue_is_flame(hue)
            
            fire_candidates.append({
                'y': y, 'x': x, 'r': r, 'g': g, 'b': b,
                'max': max_c, 'delta': delta, 'sat': sat,
                'red_ok': red_ok, 'green_ok': green_ok, 'hue': hue, 'hue_ok': hue_ok,
                'is_fire': (delta > 0 and sat >= FireDetector.MIN_SATURATION 
                           and red_ok and green_ok and hue_ok)
            })

print(f"Found {len(fire_candidates)} bright pixels (max >= {FireDetector.MIN_VALUE})")

# Show sample bright pixels
print("\nSample bright pixels:")
for i, cand in enumerate(fire_candidates[:10]):
    status = "FIRE" if cand['is_fire'] else f"rejected"
    print(f"  [{cand['y']:3d},{cand['x']:3d}] R={cand['r']:3d} G={cand['g']:3d} B={cand['b']:3d} | "
          f"sat={cand['sat']:3d} red_ok={cand['red_ok']} green_ok={cand['green_ok']} "
          f"hue={cand['hue']:3d} hue_ok={cand['hue_ok']} | {status}")

# Count different failure reasons
fire_count = sum(1 for c in fire_candidates if c['is_fire'])
low_sat_count = sum(1 for c in fire_candidates if c['sat'] < FireDetector.MIN_SATURATION and not c['is_fire'])
bad_red_count = sum(1 for c in fire_candidates if not c['red_ok'] and not c['is_fire'])
bad_green_count = sum(1 for c in fire_candidates if not c['green_ok'] and not c['is_fire'])
bad_hue_count = sum(1 for c in fire_candidates if not c['hue_ok'] and not c['is_fire'])

print(f"\nPixel classification:")
print(f"  Fire pixels detected: {fire_count}")
print(f"  Low saturation rejections: {low_sat_count}")
print(f"  Bad red dominance: {bad_red_count}")
print(f"  Green too high: {bad_green_count}")
print(f"  Bad hue rejections: {bad_hue_count}")

print(f"\nFire ratio: {fire_count / len(fire_candidates) if fire_candidates else 0:.6f}")
print(f"Threshold: {FireDetector.FIRE_PIXEL_RATIO}")
