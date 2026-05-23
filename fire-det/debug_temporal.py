"""Debug the full fire detection including flicker"""
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
    
    def __init__(self):
        self.history = np.zeros(self.HIST_LEN)
        self.hist_idx = 0
        self.consec_count = 0
        self.last_ratio = 0.0
    
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
    
    def analyse_frame(self, frame_rgb):
        total = 0
        fire = 0
        
        for y in range(0, 480, 4):
            for x in range(0, 640, 4):
                r, g, b = int(frame_rgb[y, x, 0]), int(frame_rgb[y, x, 1]), int(frame_rgb[y, x, 2])
                max_c = max(r, max(g, b))
                min_c = min(r, min(g, b))
                delta = max_c - min_c
                
                total += 1
                
                if max_c < self.MIN_VALUE:
                    continue
                if delta == 0:
                    continue
                if r < b + self.MIN_RED_DOMINANCE or g > r + 20:
                    continue
                
                sat = int((delta / max_c) * 255) if max_c > 0 else 0
                if sat < self.MIN_SATURATION:
                    continue
                
                hue = self.calc_hue(r, g, b, max_c, delta)
                if not self.hue_is_flame(hue):
                    continue
                
                fire += 1
        
        fire_ratio = fire / total if total > 0 else 0.0
        return fire_ratio
    
    def variance(self):
        mean = np.mean(self.history)
        return np.mean(self.history ** 2) - mean ** 2
    
    def update(self, frame_rgb):
        fire_ratio = self.analyse_frame(frame_rgb)
        
        self.history[self.hist_idx] = fire_ratio
        self.hist_idx = (self.hist_idx + 1) % self.HIST_LEN
        
        color_match = fire_ratio >= self.FIRE_PIXEL_RATIO
        var = self.variance()
        flicker_ok = var >= self.FLICKER_THRESH
        
        print(f"  Frame: ratio={fire_ratio:.6f} (threshold={self.FIRE_PIXEL_RATIO}) | "
              f"color_match={color_match}")
        print(f"    Variance={var:.8f} (threshold={self.FLICKER_THRESH}) | "
              f"flicker_ok={flicker_ok}")
        
        if color_match and flicker_ok:
            self.consec_count = min(self.consec_count + 1, self.CONFIRM_FRAMES)
        else:
            self.consec_count = max(self.consec_count - 1, 0)
        
        print(f"    Consecutive count: {self.consec_count}/{self.CONFIRM_FRAMES}")
        print(f"    History: {self.history}")
        
        self.last_ratio = fire_ratio
        return self.consec_count >= self.CONFIRM_FRAMES

# Test with multiple frames (simulating sequential images)
detector = FireDetector()
images_dir = Path('xgboost/fire/images')

test_images = sorted(images_dir.glob('test_*.jpg'))[:5]

print("Testing fire detector on 5 sequential images:")
print("="*70)

for img_path in test_images:
    frame = cv2.imread(str(img_path))
    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print(f"\n{img_path.name}:")
    detected = detector.update(frame_rgb)
    print(f"  -> FIRE DETECTED: {detected}")
