"""
Fire Detection Algorithm Simulator - Balanced Tuning
Adjusted parameters to reduce false positives while maintaining detection
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class FireDetector:
    """Simulates the ESP32-CAM FireDetector class - BALANCED TUNING"""
    
    # BALANCED THRESHOLDS - Moderate adjustments for better false alarm reduction
    FIRE_PIXEL_RATIO = 0.004
    MIN_VALUE = 200  # INCREASED from 180 (moderately stricter brightness)
    MIN_SATURATION = 90  # INCREASED from 80 (require more saturated colors)
    MIN_RED_DOMINANCE = 35  # INCREASED from 30 (stricter red requirement)
    FLICKER_THRESH = 0.000007  # INCREASED from 0.000005 (moderately stricter flicker)
    CONFIRM_FRAMES = 3  # Keep at 3 (original value)
    HIST_LEN = 8
    FRAME_W = 640
    FRAME_H = 480
    
    def __init__(self):
        self.history = np.zeros(self.HIST_LEN)
        self.hist_idx = 0
        self.consec_count = 0
        self.last_ratio = 0.0
        self.last_bbox = None
        self.rejection_stats = {
            'dark': 0,
            'grey': 0,
            'saturation': 0,
            'hue': 0,
            'fire': 0,
            'total': 0
        }
    
    @staticmethod
    def calc_hue(r, g, b, max_c, delta):
        """Calculate HSV hue value"""
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
        """Check if hue matches flame color"""
        # Slightly more restrictive - exclude pale yellows (15-25 range)
        return (hue <= 12) or (hue >= 165) or (27 <= hue <= 48)
    
    def analyse_frame(self, frame_rgb):
        """Analyze a single frame and return fire ratio and rejection stats"""
        h, w = frame_rgb.shape[:2]
        self.rejection_stats = {
            'dark': 0,
            'grey': 0,
            'saturation': 0,
            'hue': 0,
            'fire': 0,
            'total': 0
        }
        
        total = 0
        fire = 0
        bbox = {'x1': 9999, 'y1': 9999, 'x2': 0, 'y2': 0, 'valid': False}
        
        # Sample every 4th pixel like in Arduino code
        step = 4
        for y in range(0, h, step):
            for x in range(0, w, step):
                r, g, b = int(frame_rgb[y, x, 0]), int(frame_rgb[y, x, 1]), int(frame_rgb[y, x, 2])
                total += 1
                self.rejection_stats['total'] += 1
                
                max_c = max(r, max(g, b))
                min_c = min(r, min(g, b))
                delta = max_c - min_c
                
                # Dark pixels
                if max_c < self.MIN_VALUE:
                    self.rejection_stats['dark'] += 1
                    continue
                
                # Grayscale pixels
                if delta == 0:
                    self.rejection_stats['grey'] += 1
                    continue
                
                # Wrong hue
                if r < b + self.MIN_RED_DOMINANCE or g > r + 20:
                    self.rejection_stats['hue'] += 1
                    continue
                
                # Low saturation
                sat = int((delta / max_c) * 255) if max_c > 0 else 0
                if sat < self.MIN_SATURATION:
                    self.rejection_stats['saturation'] += 1
                    continue
                
                # Wrong hue for flame
                hue = self.calc_hue(r, g, b, max_c, delta)
                if not self.hue_is_flame(hue):
                    self.rejection_stats['hue'] += 1
                    continue
                
                # Fire pixel detected
                fire += 1
                self.rejection_stats['fire'] += 1
                
                # Update bbox
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
        """Process frame and determine if fire is detected"""
        fire_ratio = self.analyse_frame(frame_rgb)
        
        # Store in history
        self.history[self.hist_idx] = fire_ratio
        self.hist_idx = (self.hist_idx + 1) % self.HIST_LEN
        
        color_match = fire_ratio >= self.FIRE_PIXEL_RATIO
        var = self.variance()
        flicker_ok = var >= self.FLICKER_THRESH
        
        # Update consecutive counter
        if color_match and flicker_ok:
            self.consec_count = min(self.consec_count + 1, self.CONFIRM_FRAMES)
        else:
            self.consec_count = max(self.consec_count - 1, 0)
        
        self.last_ratio = fire_ratio
        return self.consec_count >= self.CONFIRM_FRAMES
    
    def variance(self):
        """Calculate variance of history"""
        mean = np.mean(self.history)
        return np.mean(self.history ** 2) - mean ** 2


def parse_yolo_label(label_path):
    """Parse YOLO format label file - returns True if contains fire (class 0)"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if lines:
                parts = lines[0].strip().split()
                return int(parts[0]) == 0
    except:
        pass
    return False


def run_simulation():
    """Run fire detection simulation processing images sequentially"""
    
    images_dir = Path('xgboost/fire/images')
    labels_dir = Path('xgboost/fire/labels')
    
    results = []
    detector = FireDetector()
    
    image_files = sorted(images_dir.glob('*.jpg'))
    print(f"Processing {len(image_files)} images sequentially (BALANCED tuning)...")
    
    # Process all images sequentially without resetting detector
    for img_path in tqdm(image_files):
        label_path = labels_dir / (img_path.stem + '.txt')
        
        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Resize to VGA if needed
        if frame.shape != (480, 640, 3):
            frame = cv2.resize(frame, (640, 480))
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detector (maintains state between frames)
        detected = detector.update(frame_rgb)
        
        # Parse label
        ground_truth = parse_yolo_label(label_path)
        
        # Store results
        results.append({
            'image': img_path.name,
            'predicted': detected,
            'ground_truth': ground_truth,
            'fire_ratio': detector.last_ratio,
            'confidence': detector.consec_count,
            'variance': detector.variance(),
            'bbox_valid': detector.last_bbox['valid'] if detector.last_bbox else False,
            'rejection_dark': detector.rejection_stats['dark'],
            'rejection_grey': detector.rejection_stats['grey'],
            'rejection_saturation': detector.rejection_stats['saturation'],
            'rejection_hue': detector.rejection_stats['hue'],
            'fire_pixels': detector.rejection_stats['fire'],
            'total_pixels': detector.rejection_stats['total']
        })
    
    return pd.DataFrame(results)


def analyze_results(df):
    """Analyze detection results"""
    
    # Classification metrics
    tp = ((df['predicted']) & (df['ground_truth'])).sum()
    tn = ((~df['predicted']) & (~df['ground_truth'])).sum()
    fp = ((df['predicted']) & (~df['ground_truth'])).sum()
    fn = ((~df['predicted']) & (df['ground_truth'])).sum()
    
    total = len(df)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*70)
    print("FIRE DETECTION PERFORMANCE METRICS (BALANCED TUNING)")
    print("="*70)
    print(f"Total frames: {total}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # False positives analysis
    fp_mask = ((df['predicted']) & (~df['ground_truth']))
    fp_df = df[fp_mask]
    
    print("\n" + "="*70)
    print("FALSE ALARM ANALYSIS")
    print("="*70)
    print(f"False Alarms: {len(fp_df)} ({len(fp_df)/total*100:.2f}%)")
    
    if len(fp_df) > 0:
        print(f"\nFalse positive details:")
        print(f"  Average fire ratio: {fp_df['fire_ratio'].mean():.6f}")
        print(f"  Average variance: {fp_df['variance'].mean():.8f}")
        print(f"  Average confidence: {fp_df['confidence'].mean():.2f}")
        print(f"\nFalse positive rejection breakdown:")
        total_rejected = (fp_df['rejection_dark'] + 
                         fp_df['rejection_grey'] + 
                         fp_df['rejection_saturation'] + 
                         fp_df['rejection_hue']).sum()
        if total_rejected > 0:
            print(f"  Dark pixels: {fp_df['rejection_dark'].sum()} ({fp_df['rejection_dark'].sum()/total_rejected*100:.1f}%)")
            print(f"  Grayscale: {fp_df['rejection_grey'].sum()} ({fp_df['rejection_grey'].sum()/total_rejected*100:.1f}%)")
            print(f"  Low saturation: {fp_df['rejection_saturation'].sum()} ({fp_df['rejection_saturation'].sum()/total_rejected*100:.1f}%)")
            print(f"  Wrong hue: {fp_df['rejection_hue'].sum()} ({fp_df['rejection_hue'].sum()/total_rejected*100:.1f}%)")
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1,
        'total': total, 'fp_df': fp_df
    }


def main():
    print("\n" + "="*70)
    print("Fire Detection Algorithm Simulator - BALANCED TUNING")
    print("="*70)
    print("\nADJUSTMENTS TO REDUCE FALSE POSITIVES (Balanced):")
    print("  MIN_VALUE:          180 → 200 (stricter brightness)")
    print("  MIN_SATURATION:      80 → 90  (require more saturated colors)")
    print("  MIN_RED_DOMINANCE:   30 → 35  (stricter red channel requirement)")
    print("  FLICKER_THRESH:      5e-6 → 7e-6 (moderately stricter variance)")
    print("  CONFIRM_FRAMES:      3 (unchanged)")
    print("  HUE RANGE:           Refined (exclude pale yellows)")
    print("="*70)
    
    # Run simulation
    results_df = run_simulation()
    
    # Save results
    results_df.to_csv('fire_detection_results_balanced.csv', index=False)
    print(f"\n✓ Results saved to: fire_detection_results_balanced.csv")
    
    # Analyze results
    metrics = analyze_results(results_df)
    
    print("\n" + "="*70)
    print("ALGORITHM PARAMETERS (BALANCED)")
    print("="*70)
    print(f"Fire Pixel Ratio Threshold: {FireDetector.FIRE_PIXEL_RATIO}")
    print(f"Min Value (Brightness): {FireDetector.MIN_VALUE}")
    print(f"Min Saturation: {FireDetector.MIN_SATURATION}")
    print(f"Min Red Dominance: {FireDetector.MIN_RED_DOMINANCE}")
    print(f"Flicker Threshold: {FireDetector.FLICKER_THRESH:.2e}")
    print(f"Confirmation Frames: {FireDetector.CONFIRM_FRAMES}")
    
    print("\n" + "="*70)
    print("COMPARISON: ORIGINAL vs BALANCED TUNING")
    print("="*70)
    
    # Load original results for comparison
    try:
        orig_df = pd.read_csv('fire_detection_results_analysis.csv')
        orig_fp = ((orig_df['predicted']) & (~orig_df['ground_truth'])).sum()
        orig_tp = ((orig_df['predicted']) & (orig_df['ground_truth'])).sum()
        
        improvement = orig_fp - metrics['fp']
        recall_change = metrics['tp'] - orig_tp
        
        print(f"\n{'Metric':<30} {'Original':<20} {'Balanced':<20}")
        print("-"*70)
        print(f"{'False Positives':<30} {orig_fp:<20} {metrics['fp']:<20}")
        print(f"{'False Alarm Rate':<30} {orig_fp/len(orig_df)*100:.2f}%{'':<14} {metrics['fp']/len(results_df)*100:.2f}%")
        print(f"{'True Positives':<30} {orig_tp:<20} {metrics['tp']:<20}")
        print(f"{'Precision':<30} {75.59:.2f}%{'':<14} {metrics['precision']*100:.2f}%")
        print(f"{'Recall':<30} {25.07:.2f}%{'':<14} {metrics['recall']*100:.2f}%")
        print(f"{'F1-Score':<30} {37.65:.2f}%{'':<14} {metrics['f1']*100:.2f}%")
        
        print(f"\n✓ False alarm reduction: {improvement} alarms ({improvement/orig_fp*100:.1f}% fewer)")
        print(f"{'':30} (trade-off: {recall_change} fewer true detections)")
    except:
        pass
    
    print("\n✓ Balanced simulation complete!")


if __name__ == '__main__':
    main()
