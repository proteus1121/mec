"""
Fire Hazard Detection Simulator - Based on XGBoost IoT Sensor Article
Simulates an IoT sensor network detecting fire hazards using machine learning.

Key Features:
- Synthetic sensor data generation (10 sensors)
- Feature engineering with derived features
- XGBoost model training and optimization
- Real-time fire hazard prediction
- Performance metrics and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import time

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class FireHazardSensor:
    """Simulates IoT sensors for fire detection"""
    
    def __init__(self, num_samples=5000):
        """
        Initialize sensor simulator with 10 sensor types
        """
        self.num_samples = num_samples
        self.sensor_names = [
            'Temperature', 'Humidity', 'Smoke', 'LPG', 'CH4',
            'CO', 'Flame', 'Motion', 'Illuminance', 'Pressure'
        ]
        
    def generate_safe_conditions(self, n_samples):
        """Generate sensor readings for SAFE conditions (~65% of data)"""
        data = np.zeros((n_samples, 10))
        data[:, 0] = np.random.normal(25, 5, n_samples)      # Temperature (°C)
        data[:, 1] = np.random.normal(50, 15, n_samples)     # Humidity (%)
        data[:, 2] = np.random.normal(0, 0.5, n_samples)     # Smoke (ppm)
        data[:, 3] = np.random.normal(100, 20, n_samples)    # LPG (ppm)
        data[:, 4] = np.random.normal(400, 100, n_samples)   # CH4 (ppm)
        data[:, 5] = np.random.normal(0.5, 0.2, n_samples)   # CO (ppm)
        data[:, 6] = np.random.normal(0, 0.1, n_samples)     # Flame (presence)
        data[:, 7] = np.random.normal(0, 0.1, n_samples)     # Motion (activity)
        data[:, 8] = np.random.normal(300, 100, n_samples)   # Illuminance (lux)
        data[:, 9] = np.random.normal(1013, 10, n_samples)   # Pressure (hPa)
        
        # Ensure physically realistic bounds
        data = np.maximum(data, 0)
        data[:, 0] = np.clip(data[:, 0], -10, 50)
        data[:, 1] = np.clip(data[:, 1], 0, 100)
        return data, np.zeros(n_samples, dtype=int)  # Label: 0 = Safe
    
    def generate_potentially_dangerous(self, n_samples):
        """Generate sensor readings for POTENTIALLY DANGEROUS conditions (~20%)"""
        data = np.zeros((n_samples, 10))
        data[:, 0] = np.random.normal(40, 8, n_samples)      # Higher temperature
        data[:, 1] = np.random.normal(35, 15, n_samples)     # Lower humidity (dry)
        data[:, 2] = np.random.normal(2, 1.5, n_samples)     # Moderate smoke
        data[:, 3] = np.random.normal(300, 100, n_samples)   # Higher LPG
        data[:, 4] = np.random.normal(600, 200, n_samples)   # Higher CH4
        data[:, 5] = np.random.normal(1.5, 0.5, n_samples)   # Higher CO
        data[:, 6] = np.random.normal(0.1, 0.15, n_samples)  # Occasional flame
        data[:, 7] = np.random.normal(0.3, 0.2, n_samples)   # Some motion
        data[:, 8] = np.random.normal(200, 100, n_samples)   # Lower illuminance
        data[:, 9] = np.random.normal(1010, 15, n_samples)   # Pressure variation
        
        data = np.maximum(data, 0)
        data[:, 0] = np.clip(data[:, 0], -10, 80)
        data[:, 1] = np.clip(data[:, 1], 0, 100)
        return data, np.ones(n_samples, dtype=int)  # Label: 1 = Potentially Dangerous
    
    def generate_fire_hazardous(self, n_samples):
        """Generate sensor readings for FIRE-HAZARDOUS conditions (~15%)"""
        data = np.zeros((n_samples, 10))
        # High temperature + high smoke = critical fire signature
        data[:, 0] = np.random.normal(60, 10, n_samples)     # Very high temp
        data[:, 1] = np.random.normal(20, 10, n_samples)     # Very low humidity
        data[:, 2] = np.random.normal(8, 3, n_samples)       # Very high smoke
        data[:, 3] = np.random.normal(600, 150, n_samples)   # Very high LPG
        data[:, 4] = np.random.normal(1000, 300, n_samples)  # Very high CH4
        data[:, 5] = np.random.normal(5, 1.5, n_samples)     # Very high CO
        data[:, 6] = np.random.normal(0.8, 0.15, n_samples)  # Strong flame signal
        data[:, 7] = np.random.normal(0.7, 0.2, n_samples)   # High motion
        data[:, 8] = np.random.normal(50, 50, n_samples)     # Very low illuminance
        data[:, 9] = np.random.normal(1005, 20, n_samples)   # Pressure anomaly
        
        data = np.maximum(data, 0)
        data[:, 0] = np.clip(data[:, 0], 20, 100)
        data[:, 1] = np.clip(data[:, 1], 0, 100)
        return data, np.full(n_samples, 2, dtype=int)  # Label: 2 = Fire Hazardous
    
    def generate_dataset(self):
        """Generate complete synthetic dataset with class distribution"""
        # Class distribution: 65% Safe, 20% Potentially Dangerous, 15% Fire-Hazardous
        n_safe = int(self.num_samples * 0.65)
        n_danger = int(self.num_samples * 0.20)
        n_fire = self.num_samples - n_safe - n_danger
        
        X_safe, y_safe = self.generate_safe_conditions(n_safe)
        X_danger, y_danger = self.generate_potentially_dangerous(n_danger)
        X_fire, y_fire = self.generate_fire_hazardous(n_fire)
        
        # Combine and shuffle
        X = np.vstack([X_safe, X_danger, X_fire])
        y = np.hstack([y_safe, y_danger, y_fire])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=self.sensor_names)
        df['Class'] = y
        
        return df


class FeatureEngineer:
    """Creates derived features for fire detection"""
    
    @staticmethod
    def engineer_features(df):
        """
        Create derived features based on article methodology:
        - Temperature × Smoke product (synergistic effect)
        - Gas ratios (CH4/CO)
        - Differences and interactions
        """
        df_eng = df.copy()
        
        # Critical interaction: Temperature × Smoke
        df_eng['Temp_Smoke_Product'] = df['Temperature'] * df['Smoke']
        
        # Gas concentration ratios
        df_eng['CH4_CO_Ratio'] = df['CH4'] / (df['CO'] + 0.01)  # Avoid division by zero
        df_eng['LPG_CH4_Ratio'] = df['LPG'] / (df['CH4'] + 0.01)
        
        # Hazardous gas indicator (sum of dangerous gases)
        df_eng['Hazardous_Gas_Index'] = df['LPG'] + df['CH4'] + df['CO']
        
        # Temperature anomaly
        df_eng['Temp_Anomaly'] = np.abs(df['Temperature'] - 25)  # Deviation from normal
        
        # Humidity stress indicator (low humidity during high temp is dangerous)
        df_eng['Humidity_Stress'] = (100 - df['Humidity']) * (df['Temperature'] / 25)
        
        # Smoke-Flame correlation
        df_eng['Smoke_Flame_Product'] = df['Smoke'] * df['Flame']
        
        return df_eng


class FireDetectionModel:
    """Main fire detection model using XGBoost"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.performance_metrics = {}
        
    def prepare_data(self, df):
        """Prepare and split data for training"""
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        self.feature_names = X.columns.tolist()
        
        # Split data: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Normalize features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model with optimized hyperparameters"""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
        print("="*60)
        
        # Optimized hyperparameters from article
        params = {
            'learning_rate': 0.04,        # Low learning rate
            'n_estimators': 850,          # More trees for slow learning
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.25,
            'reg_lambda': 1.0,
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        start_time = time.time()
        
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        return training_time
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Fire-hazard recall (critical metric) - manually calculate from class 2
        fire_true_positives = np.sum((y_test == 2) & (y_pred == 2))
        fire_total = np.sum(y_test == 2)
        fire_recall = fire_true_positives / fire_total if fire_total > 0 else 0
        
        # Multi-class ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        
        self.performance_metrics[dataset_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Fire_Recall': fire_recall,
            'ROC-AUC': roc_auc
        }
        
        print(f"\n{dataset_name} Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Fire-Hazard Recall: {fire_recall:.4f} ⚠️ (Critical Metric)")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        return y_pred, y_pred_proba
    
    def get_feature_importance(self):
        """Extract and rank feature importance"""
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return feature_imp
    
    def predict_real_time(self, sensor_readings):
        """
        Real-time prediction for new sensor readings
        
        Args:
            sensor_readings: dict or array of sensor values
        
        Returns:
            prediction: class (0=Safe, 1=Potentially Dangerous, 2=Fire-Hazardous)
            confidence: probability of the predicted class
        """
        if isinstance(sensor_readings, dict):
            sensor_readings = pd.DataFrame([sensor_readings])
        else:
            sensor_readings = pd.DataFrame([sensor_readings], columns=self.feature_names[:10])
        
        # Engineer features
        sensor_readings = FeatureEngineer.engineer_features(sensor_readings)
        
        # Normalize
        sensor_readings = self.scaler.transform(sensor_readings)
        
        # Predict
        pred = self.model.predict(sensor_readings)[0]
        proba = self.model.predict_proba(sensor_readings)[0]
        confidence = max(proba)
        
        return pred, confidence, proba


def visualize_results(model, X_test, y_test, feature_importance):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Feature Importance
    ax1 = plt.subplot(2, 3, 1)
    top_features = feature_importance.head(10)
    ax1.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
    ax1.set_xlabel('Importance (Gain)')
    ax1.set_title('Top 10 Feature Importance')
    ax1.invert_yaxis()
    
    # 2. Confusion Matrix
    ax2 = plt.subplot(2, 3, 2)
    y_pred = model.model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Safe', 'Potentially Dangerous', 'Fire-Hazardous'],
                yticklabels=['Safe', 'Potentially Dangerous', 'Fire-Hazardous'])
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    ax2.set_title('Confusion Matrix')
    
    # 3. Performance Metrics
    ax3 = plt.subplot(2, 3, 3)
    metrics = model.performance_metrics.get('Test', {})
    metric_names = list(metrics.keys())[:5]
    metric_values = [metrics[m] for m in metric_names]
    colors = ['#2ecc71' if v > 0.95 else '#f39c12' if v > 0.9 else '#e74c3c' 
              for v in metric_values]
    ax3.bar(range(len(metric_names)), metric_values, color=colors)
    ax3.set_xticks(range(len(metric_names)))
    ax3.set_xticklabels([m.replace('_', ' ') for m in metric_names], rotation=45, ha='right')
    ax3.set_ylabel('Score')
    ax3.set_ylim([0.85, 1.0])
    ax3.set_title('Performance Metrics')
    ax3.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target')
    ax3.legend()
    
    # 4. Class Distribution in Predictions
    ax4 = plt.subplot(2, 3, 4)
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    true_counts = pd.Series(y_test).value_counts().sort_index()
    x = np.arange(3)
    width = 0.35
    ax4.bar(x - width/2, true_counts.values, width, label='True', alpha=0.8)
    ax4.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Safe', 'Potentially Dangerous', 'Fire-Hazardous'])
    ax4.set_ylabel('Count')
    ax4.set_title('Class Distribution: True vs Predicted')
    ax4.legend()
    
    # 5. Confidence Distribution
    ax5 = plt.subplot(2, 3, 5)
    y_pred_proba = model.model.predict_proba(X_test)
    max_proba = np.max(y_pred_proba, axis=1)
    ax5.hist(max_proba, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Prediction Confidence')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Model Confidence Distribution')
    ax5.axvline(x=0.9, color='r', linestyle='--', alpha=0.7, label='High Confidence')
    ax5.legend()
    
    # 6. Fire-Hazard Detection Rate
    ax6 = plt.subplot(2, 3, 6)
    fire_true = (y_test == 2).sum()
    fire_detected = ((y_test == 2) & (y_pred == 2)).sum()
    fire_missed = fire_true - fire_detected
    fire_false_alarms = ((y_test != 2) & (y_pred == 2)).sum()
    
    categories = ['Correctly\nDetected', 'Missed', 'False\nAlarms']
    values = [fire_detected, fire_missed, fire_false_alarms]
    colors_fire = ['#2ecc71', '#e74c3c', '#f39c12']
    ax6.bar(categories, values, color=colors_fire, alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Count')
    ax6.set_title(f'Fire-Hazard Detection (Detection Rate: {fire_detected/fire_true:.1%})')
    
    plt.tight_layout()
    plt.savefig('c:\\Users\\Artem\\Documents\\calculations\\fire_detection_results.png', dpi=300, bbox_inches='tight')
    print("\n📊 Results saved to: fire_detection_results.png")
    plt.show()


def main():
    """Main simulation execution"""
    print("\n" + "🔥"*30)
    print("FIRE HAZARD DETECTION IoT SIMULATOR")
    print("Based on XGBoost Article - Hyperparameter Optimization Study")
    print("🔥"*30)
    
    # Step 1: Generate synthetic sensor data
    print("\n[1/5] Generating synthetic sensor data...")
    sensor = FireHazardSensor(num_samples=5000)
    df = sensor.generate_dataset()
    print(f"✓ Generated {len(df)} samples across 10 sensors")
    print(f"  Class distribution:\n{df['Class'].value_counts().sort_index()}")
    
    # Step 2: Feature engineering
    print("\n[2/5] Engineering derived features...")
    df_engineered = FeatureEngineer.engineer_features(df)
    print(f"✓ Created {df_engineered.shape[1]-1} features (original + derived)")
    print(f"  Features: {', '.join(df_engineered.columns[:-1].tolist()[:5])}...")
    
    # Step 3: Prepare data
    print("\n[3/5] Preparing training/validation/test splits...")
    model = FireDetectionModel()
    (X_train, X_val, X_test), (y_train, y_val, y_test) = model.prepare_data(df_engineered)
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    print(f"  SMOTE applied to handle class imbalance")
    
    # Step 4: Train model
    print("\n[4/5] Training XGBoost with optimized hyperparameters...")
    training_time = model.train(X_train, y_train, X_val, y_val)
    print("✓ Model trained successfully")
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating model performance...")
    print("\n" + "="*60)
    model.evaluate(X_train, y_train, "Training")
    model.evaluate(X_val, y_val, "Validation")
    y_pred_test, y_proba_test = model.evaluate(X_test, y_test, "Test")
    print("="*60)
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    print("\n📊 Feature Importance (Top 10):")
    print(feature_importance.head(10).to_string(index=False))
    
    # Real-time prediction examples
    print("\n" + "="*60)
    print("REAL-TIME PREDICTION EXAMPLES")
    print("="*60)
    
    # Example 1: Safe conditions
    safe_reading = {
        'Temperature': 22, 'Humidity': 55, 'Smoke': 0.1, 'LPG': 95,
        'CH4': 380, 'CO': 0.3, 'Flame': 0, 'Motion': 0.05,
        'Illuminance': 350, 'Pressure': 1013
    }
    pred, conf, proba = model.predict_real_time(safe_reading)
    class_names = ['Safe', 'Potentially Dangerous', 'Fire-Hazardous']
    print(f"\n✓ Safe Conditions Reading:")
    print(f"  Prediction: {class_names[pred]} (confidence: {conf:.2%})")
    print(f"  Probabilities: Safe={proba[0]:.2%}, Dangerous={proba[1]:.2%}, Fire={proba[2]:.2%}")
    
    # Example 2: Dangerous conditions
    dangerous_reading = {
        'Temperature': 45, 'Humidity': 30, 'Smoke': 3.5, 'LPG': 350,
        'CH4': 750, 'CO': 2.0, 'Flame': 0.2, 'Motion': 0.4,
        'Illuminance': 150, 'Pressure': 1008
    }
    pred, conf, proba = model.predict_real_time(dangerous_reading)
    print(f"\n⚠️  Potentially Dangerous Conditions Reading:")
    print(f"  Prediction: {class_names[pred]} (confidence: {conf:.2%})")
    print(f"  Probabilities: Safe={proba[0]:.2%}, Dangerous={proba[1]:.2%}, Fire={proba[2]:.2%}")
    
    # Example 3: Fire-hazardous conditions
    fire_reading = {
        'Temperature': 68, 'Humidity': 15, 'Smoke': 9.2, 'LPG': 750,
        'CH4': 1200, 'CO': 6.5, 'Flame': 0.9, 'Motion': 0.8,
        'Illuminance': 20, 'Pressure': 1000
    }
    pred, conf, proba = model.predict_real_time(fire_reading)
    alert_symbol = "🔥" if pred == 2 else "⚠️ "
    print(f"\n{alert_symbol} Fire-Hazardous Conditions Reading:")
    print(f"  Prediction: {class_names[pred]} (confidence: {conf:.2%})")
    print(f"  Probabilities: Safe={proba[0]:.2%}, Dangerous={proba[1]:.2%}, Fire={proba[2]:.2%}")
    
    # Visualization
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS...")
    print("="*60)
    visualize_results(model, X_test, y_test, feature_importance)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print("="*60)
    print("\n📈 Results Summary:")
    print(f"  Model Accuracy: {model.performance_metrics['Test']['Accuracy']:.4f}")
    print(f"  Fire Detection Recall: {model.performance_metrics['Test']['Fire_Recall']:.4f}")
    print(f"  Training Time: {training_time:.2f} seconds")
    print(f"  Total Features: {len(model.feature_names)}")


if __name__ == "__main__":
    main()
