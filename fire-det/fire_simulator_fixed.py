"""
Fire Detection Algorithm Simulator - Fixed Version
Processes images in temporal sequences to properly evaluate the algorithm
"""

import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class FireDetector:
    """Simulates the ESP32-CAM FireDetector class"""
    
    # Thresholds from fire_detection.ino
    FIRE_PIXEL_RATIO = 0.004
    MIN_VALUE = 180
    MIN_SATURATION = 80
    MIN_RED_DOMINANCE = 30
    FLICKER_THRESH = 0.000005
    CONFIRM_FRAMES = 3
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
        return (hue <= 18) or (hue >= 162) or (19 <= hue <= 50)
    
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
                
                # Low saturation - avoid overflow by using float
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
    
    def reset(self):
        """Reset for new sequence"""
        self.history = np.zeros(self.HIST_LEN)
        self.hist_idx = 0
        self.consec_count = 0
        self.last_ratio = 0.0


def parse_yolo_label(label_path):
    """Parse YOLO format label file - returns True if contains fire (class 0)"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return False  # No label = no fire
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if lines:
                parts = lines[0].strip().split()
                # Class 0 = fire, anything else = no fire
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
    print(f"Processing {len(image_files)} images sequentially (WITH temporal continuity)...")
    
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
    """Analyze detection results and false alarm sources"""
    
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
    print("FIRE DETECTION PERFORMANCE METRICS")
    print("="*70)
    print(f"Total frames: {total}")
    print(f"True Positives (correct fire detection): {tp}")
    print(f"True Negatives (correct non-fire): {tn}")
    print(f"False Positives (false alarms): {fp}")
    print(f"False Negatives (missed fire): {fn}")
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Analyze false positives (false alarms)
    fp_mask = ((df['predicted']) & (~df['ground_truth']))
    fp_df = df[fp_mask]
    
    print("\n" + "="*70)
    print("FALSE ALARM ANALYSIS (False Positives)")
    print("="*70)
    print(f"Number of false alarms: {len(fp_df)}")
    
    if len(fp_df) > 0:
        print(f"\nAverage metrics for false alarms:")
        print(f"  Fire ratio: {fp_df['fire_ratio'].mean():.6f}")
        print(f"  Variance: {fp_df['variance'].mean():.8f}")
        print(f"  Confidence: {fp_df['confidence'].mean():.2f}")
        
        # Rejection statistics
        total_rejected = (fp_df['rejection_dark'] + 
                         fp_df['rejection_grey'] + 
                         fp_df['rejection_saturation'] + 
                         fp_df['rejection_hue']).sum()
        
        if total_rejected > 0:
            print(f"\nRejection breakdown across false alarms:")
            print(f"  Dark pixels (MIN_VALUE): {fp_df['rejection_dark'].sum()} ({fp_df['rejection_dark'].sum()/total_rejected*100:.1f}%)")
            print(f"  Grayscale (delta=0): {fp_df['rejection_grey'].sum()} ({fp_df['rejection_grey'].sum()/total_rejected*100:.1f}%)")
            print(f"  Low saturation: {fp_df['rejection_saturation'].sum()} ({fp_df['rejection_saturation'].sum()/total_rejected*100:.1f}%)")
            print(f"  Wrong hue: {fp_df['rejection_hue'].sum()} ({fp_df['rejection_hue'].sum()/total_rejected*100:.1f}%)")
    
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1,
        'total': total, 'fp_df': fp_df
    }


def create_table_1(df, metrics):
    """Create Table 1: Sources of false alarms and suppression mechanisms"""
    
    fp_df = metrics['fp_df']
    
    # Sources of false alarms and their suppression
    table_data = []
    
    # 1. Dark regions
    table_data.append({
        'False Alarm Source': 'Dark regions (noise, shadows)',
        'Optical Artifact': 'Low brightness pixels detected as fire',
        'Threshold Mechanism': f'MIN_VALUE ≥ {FireDetector.MIN_VALUE}',
        'Pixels Filtered': f'{fp_df["rejection_dark"].sum()}',
        'Effectiveness': 'Dark pixels rejected at first stage'
    })
    
    # 2. Grayscale artifacts
    table_data.append({
        'False Alarm Source': 'Grayscale artifacts (reflections)',
        'Optical Artifact': 'Monochromatic pixels (neutral color)',
        'Threshold Mechanism': 'δ = max(R,G,B) - min(R,G,B) > 0',
        'Pixels Filtered': f'{fp_df["rejection_grey"].sum()}',
        'Effectiveness': 'Grayscale pixels rejected (no color)'
    })
    
    # 3. Low saturation
    table_data.append({
        'False Alarm Source': 'Desaturated regions (pale colors)',
        'Optical Artifact': 'Low saturation mimicking fire-like hues',
        'Threshold Mechanism': f'MIN_SATURATION ≥ {FireDetector.MIN_SATURATION}',
        'Pixels Filtered': f'{fp_df["rejection_saturation"].sum()}',
        'Effectiveness': 'Low saturation pixels eliminated'
    })
    
    # 4. Wrong hue
    table_data.append({
        'False Alarm Source': 'Non-fire colored regions',
        'Optical Artifact': f'Hues outside flame range (0-50°, 162-180°)',
        'Threshold Mechanism': f'RED_DOMINANCE ≥ {FireDetector.MIN_RED_DOMINANCE}',
        'Pixels Filtered': f'{fp_df["rejection_hue"].sum()}',
        'Effectiveness': 'Non-flame hues filtered out'
    })
    
    # 5. Flicker suppression
    table_data.append({
        'False Alarm Source': 'Transient optical artifacts',
        'Optical Artifact': 'Single-frame anomalies without temporal consistency',
        'Threshold Mechanism': f'Variance ≥ {FireDetector.FLICKER_THRESH}',
        'Pixels Filtered': 'Temporal variance analysis',
        'Effectiveness': f'Requires variance > {FireDetector.FLICKER_THRESH} for confirmation'
    })
    
    # 6. Confirmation counter
    consec_fps = len(fp_df[fp_df['confidence'] < FireDetector.CONFIRM_FRAMES])
    table_data.append({
        'False Alarm Source': 'Frame-level false positives',
        'Optical Artifact': 'Single/few frames meeting fire criteria',
        'Threshold Mechanism': f'CONFIRM_FRAMES = {FireDetector.CONFIRM_FRAMES}',
        'Pixels Filtered': f'~{consec_fps} frames rejected',
        'Effectiveness': f'Requires ≥{FireDetector.CONFIRM_FRAMES} consecutive frame confirmations'
    })
    
    table_1_df = pd.DataFrame(table_data)
    
    print("\n" + "="*70)
    print("TABLE 1: SOURCES OF FALSE ALARMS AND SUPPRESSION MECHANISMS")
    print("="*70)
    print(table_1_df.to_string(index=False))
    
    # Save to CSV
    table_1_df.to_csv('fire_detection_table_1.csv', index=False)
    print("\n✓ Table 1 saved to: fire_detection_table_1.csv")
    
    return table_1_df


def main():
    print("\n" + "="*70)
    print("Fire Detection Algorithm Simulator")
    print("Processing with temporal continuity for realistic evaluation")
    print("="*70)
    
    # Run simulation
    results_df = run_simulation()
    
    # Save results
    results_df.to_csv('fire_detection_results_analysis.csv', index=False)
    print(f"\n✓ Results saved to: fire_detection_results_analysis.csv")
    
    # Analyze results
    metrics = analyze_results(results_df)
    
    # Create Table 1
    table_1 = create_table_1(results_df, metrics)
    
    # Summary statistics
    print("\n" + "="*70)
    print("ALGORITHM PARAMETERS")
    print("="*70)
    print(f"Fire Pixel Ratio Threshold (τ_ratio): {FireDetector.FIRE_PIXEL_RATIO}")
    print(f"Minimum Brightness (MIN_VALUE): {FireDetector.MIN_VALUE}")
    print(f"Minimum Saturation: {FireDetector.MIN_SATURATION}")
    print(f"Minimum Red Dominance: {FireDetector.MIN_RED_DOMINANCE}")
    print(f"Flicker Variance Threshold (τ_flicker): {FireDetector.FLICKER_THRESH:.2e}")
    print(f"Confirmation Frames (N_confirm): {FireDetector.CONFIRM_FRAMES}")
    print(f"History/Variance Window Length: {FireDetector.HIST_LEN} frames")
    
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    fire_images = results_df['ground_truth'].sum()
    non_fire_images = (~results_df['ground_truth']).sum()
    print(f"Fire images in dataset: {fire_images}")
    print(f"Non-fire images in dataset: {non_fire_images}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
