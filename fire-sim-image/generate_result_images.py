"""
Generate annotated images with bounding boxes for manual review
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class FireDetector:
    """Simulates the ESP32-CAM FireDetector class"""
    
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
        self.last_bbox = None
    
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
        h, w = frame_rgb.shape[:2]
        
        total = 0
        fire = 0
        bbox = {'x1': 9999, 'y1': 9999, 'x2': 0, 'y2': 0, 'valid': False}
        
        step = 4
        for y in range(0, h, step):
            for x in range(0, w, step):
                r, g, b = int(frame_rgb[y, x, 0]), int(frame_rgb[y, x, 1]), int(frame_rgb[y, x, 2])
                total += 1
                
                max_c = max(r, max(g, b))
                min_c = min(r, min(g, b))
                delta = max_c - min_c
                
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
                
                if x < bbox['x1']:
                    bbox['x1'] = x
                if y < bbox['y1']:
                    bbox['y1'] = y
                if x > bbox['x2']:
                    bbox['x2'] = x
                if y > bbox['y2']:
                    bbox['y2'] = y
                bbox['valid'] = True
        
        self.last_bbox = bbox
        fire_ratio = fire / total if total > 0 else 0.0
        return fire_ratio
    
    def update(self, frame_rgb):
        fire_ratio = self.analyse_frame(frame_rgb)
        
        self.history[self.hist_idx] = fire_ratio
        self.hist_idx = (self.hist_idx + 1) % self.HIST_LEN
        
        color_match = fire_ratio >= self.FIRE_PIXEL_RATIO
        var = self.variance()
        flicker_ok = var >= self.FLICKER_THRESH
        
        if color_match and flicker_ok:
            self.consec_count = min(self.consec_count + 1, self.CONFIRM_FRAMES)
        else:
            self.consec_count = max(self.consec_count - 1, 0)
        
        self.last_ratio = fire_ratio
        return self.consec_count >= self.CONFIRM_FRAMES
    
    def variance(self):
        mean = np.mean(self.history)
        return np.mean(self.history ** 2) - mean ** 2


def draw_bbox_on_image(frame_bgr, bbox, confidence, fire_ratio, save_path):
    """Draw bounding box and info on image"""
    if not bbox or not bbox.get('valid', False):
        return False
    
    # Add padding to bbox
    PAD = 6
    x1 = max(0, bbox['x1'] - PAD)
    y1 = max(0, bbox['y1'] - PAD)
    x2 = min(frame_bgr.shape[1] - 1, bbox['x2'] + PAD)
    y2 = min(frame_bgr.shape[0] - 1, bbox['y2'] + PAD)
    
    # Draw rectangle (orange color for fire)
    thickness = 3
    color = (0, 165, 255)  # Orange in BGR
    
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)
    
    # Add text info
    text = f"Confidence: {confidence}/3 | Fire%: {fire_ratio*100:.1f}%"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = max(0, x1)
    text_y = max(20, y1 - 10)
    
    # Draw text background
    cv2.rectangle(frame_bgr, 
                  (text_x - 5, text_y - text_size[1] - 5),
                  (text_x + text_size[0] + 5, text_y + 5),
                  color, -1)
    
    # Draw text
    cv2.putText(frame_bgr, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Save image
    cv2.imwrite(str(save_path), frame_bgr)
    return True


def generate_result_images():
    """Generate annotated images for all detected fires"""
    
    # Read results
    results_df = pd.read_csv('fire_detection_results_analysis.csv')
    
    images_dir = Path('xgboost/fire/images')
    result_dir = Path('result_images')
    
    detector = FireDetector()
    
    # Get all predicted fire frames
    predicted_fires = results_df[results_df['predicted'] == True]
    
    print(f"\nGenerating annotated images for {len(predicted_fires)} detected fires...")
    print("="*70)
    
    for idx, row in tqdm(predicted_fires.iterrows(), total=len(predicted_fires)):
        img_name = row['image']
        img_path = images_dir / img_name
        
        if not img_path.exists():
            continue
        
        # Load and process image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        if frame.shape != (480, 640, 3):
            frame = cv2.resize(frame, (640, 480))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detector to get bbox
        detector.update(frame_rgb)
        bbox = detector.last_bbox
        confidence = row['confidence']
        fire_ratio = row['fire_ratio']
        
        # Draw bbox on original BGR image
        save_path = result_dir / f"FIRE_{img_name}"
        draw_bbox_on_image(frame, bbox, int(confidence), fire_ratio, save_path)
    
    print(f"\n✓ Generated {len(predicted_fires)} annotated images in result_images/")
    print(f"  Images saved with format: FIRE_test_N.jpg")
    
    # Generate summary file
    summary_file = result_dir / 'SUMMARY.txt'
    with open(summary_file, 'w') as f:
        f.write("MANUAL REVIEW - FIRE DETECTION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total detected fire frames: {len(predicted_fires)}\n")
        f.write(f"Total test images: {len(results_df)}\n")
        f.write(f"Dataset: xgboost/fire/images/\n\n")
        f.write("IMAGE NAMING: FIRE_test_N.jpg\n")
        f.write("ANNOTATION: Orange bounding box around detected fire region\n")
        f.write("INFO: Confidence (0-3) and Fire% displayed on each image\n\n")
        f.write("LEGEND:\n")
        f.write("  Confidence 0-2: Algorithm still building confidence\n")
        f.write("  Confidence 3:   Full confirmation - alarm would trigger\n")
        f.write("  Fire%:          Percentage of sampled pixels matching fire color\n\n")
        f.write("REVIEW INSTRUCTIONS:\n")
        f.write("  1. Open each FIRE_*.jpg image\n")
        f.write("  2. Verify the bounding box correctly identifies fire\n")
        f.write("  3. Check for false alarms (no actual fire visible)\n")
        f.write("  4. Note confidence levels and fire percentages\n")
        f.write("  5. Compare with original images to assess detection quality\n\n")
        f.write("DETAILED RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Image':<30} {'Conf':<6} {'Fire%':<10} {'G.Truth':<10}\n")
        f.write("-"*70 + "\n")
        for _, row in predicted_fires.iterrows():
            gt = "FIRE" if row['ground_truth'] else "NO-FIRE"
            f.write(f"{row['image']:<30} {int(row['confidence']):<6} "
                   f"{row['fire_ratio']*100:<10.2f} {gt:<10}\n")
    
    print(f"✓ Summary file created: result_images/SUMMARY.txt")
    
    # Print statistics
    print("\n" + "="*70)
    print("DETECTION BREAKDOWN:")
    print("="*70)
    
    tp = len(predicted_fires[predicted_fires['ground_truth'] == True])
    fp = len(predicted_fires[predicted_fires['ground_truth'] == False])
    
    print(f"True Positives (correct fires):       {tp}")
    print(f"False Positives (false alarms):       {fp}")
    print(f"Total detected:                       {len(predicted_fires)}")
    print(f"\nTrue Positive Rate:  {tp/len(predicted_fires)*100:.1f}%")
    print(f"False Positive Rate: {fp/len(predicted_fires)*100:.1f}%")


if __name__ == '__main__':
    generate_result_images()
