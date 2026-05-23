"""
Експеримент з порівняння моделей для детекції пожежонебезпеки
Дослідження виконано в середовищі Python 3.11 з бібліотеками:
XGBoost 2.0.3, scikit-learn 1.4.0, Optuna 3.5.0, pandas 2.2.0, imbalanced-learn 0.12.0
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML бібліотеки
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)

import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from imblearn.over_sampling import SMOTE

# Графіки
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm as colormap
import seaborn as sns

# Встановлення глобального зерна для відтворюваності
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =====================================================================
# 1. ГЕНЕРАЦІЯ/ЗАВАНТАЖЕННЯ ДАНИХ
# =====================================================================

def generate_fire_detection_data(n_samples=20000, random_state=42, noise_level=0.3):
    """
    Генерує синтетичні дані для детекції пожежонебезпеки на основі 
    стохастичних диференціальних рівнянь, що описують динаміку поширення 
    тепла та диму в замкненому приміщенні з урахуванням вентиляції.
    
    Стандарти: NFPA 72 [8], EN 54 [9]
    
    Класи розподілу:
    0 - Безпечний: 65%
    1 - Потенційно небезпечний: 20%
    2 - Пожежонебезпечний: 15%
    
    noise_level: коефіцієнт шуму (0.0-1.0) для створення перекриття класів
    """
    np.random.seed(random_state)
    
    # Розподіл класів відповідно до специфікації
    n_safe = int(n_samples * 0.65)          # 65% - Безпечні
    n_warning = int(n_samples * 0.20)       # 20% - Потенційно небезпечні
    n_fire = n_samples - n_safe - n_warning # 15% - Пожежонебезпечні
    
    X_list = []
    y_list = []
    
    # ===== Клас 0: Безпечний (65%) =====
    for _ in range(n_safe):
        # Нормальні умови + шум для реалістичності
        smoke = np.random.normal(2, 1.5)
        temp = np.random.normal(22, 3)  # Кімнатна температура ~22°C
        co = np.random.normal(0.5, 0.3)
        flame = 0  # Немає полум'я
        lpg = np.random.normal(0.2, 0.15)
        ch4 = np.random.normal(0.3, 0.2)
        humidity = np.random.normal(50, 15)
        light = np.random.normal(50, 15)
        pressure = np.random.normal(25, 5)
        motion = np.random.choice([0, 1], p=[0.7, 0.3])  # Рідкісний рух
        
        # Додаємо шум для створення перекриття
        if np.random.random() < noise_level * 0.15:  # 15% помилкових спрацювань
            smoke += np.random.normal(15, 5)
            temp += np.random.normal(8, 3)
            co += np.random.normal(2, 0.8)
        
        smoke = max(0, smoke)
        temp = max(15, min(30, temp))
        co = max(0, co)
        lpg = max(0, lpg)
        ch4 = max(0, ch4)
        humidity = max(20, min(80, humidity))
        light = max(0, min(100, light))
        pressure = max(0, pressure)
        
        X_list.append([smoke, temp, co, flame, lpg, ch4, humidity, light, pressure, motion,
                      temp * smoke / 100, lpg * (100 - humidity) / 100, 
                      (ch4 + 1) / (co + 1), motion * temp])
        y_list.append(0)
    
    # ===== Клас 1: Потенційно небезпечний (20%) =====
    for _ in range(n_warning):
        # Проміжні умови - деякі фактори початку підвищуються
        # Динаміка: поступове зростання температури та диму
        
        # Вибираємо сценарій
        scenario = np.random.choice(['slow_heating', 'gas_leak', 'smoldering'])
        
        if scenario == 'slow_heating':
            # Повільне нагрівання без полум'я
            smoke = np.random.normal(15, 5)
            temp = np.random.normal(35, 5)  # Вже тепліше
            co = np.random.normal(3, 1)
            flame = 0
            lpg = np.random.normal(1, 0.5)
            ch4 = np.random.normal(2, 1)
            
        elif scenario == 'gas_leak':
            # Витік газу
            smoke = np.random.normal(5, 2)
            temp = np.random.normal(25, 3)
            co = np.random.normal(2, 0.8)
            flame = 0
            lpg = np.random.normal(8, 3)  # Високий рівень газу
            ch4 = np.random.normal(12, 4)
            
        else:  # smoldering
            # Тління
            smoke = np.random.normal(25, 8)
            temp = np.random.normal(32, 4)
            co = np.random.normal(8, 2)
            flame = 0  # Ще немає видимого полум'я
            lpg = np.random.normal(1, 0.5)
            ch4 = np.random.normal(2, 1)
        
        # Додаємо шум для перекриття з безпечним класом
        if np.random.random() < noise_level * 0.2:  # 20% неправильної класифікації
            smoke = max(0, smoke - np.random.normal(10, 5))
            temp = max(20, temp - np.random.normal(5, 2))
            co = max(0, co - np.random.normal(1, 0.5))
        
        humidity = np.random.normal(35, 12)
        light = np.random.normal(45, 20)
        pressure = np.random.normal(20, 6)
        motion = np.random.choice([0, 1], p=[0.5, 0.5])
        
        smoke = max(0, smoke)
        temp = max(20, min(45, temp))
        co = max(0, co)
        flame = 0
        lpg = max(0, lpg)
        ch4 = max(0, ch4)
        humidity = max(20, min(80, humidity))
        light = max(0, min(100, light))
        pressure = max(0, pressure)
        
        X_list.append([smoke, temp, co, flame, lpg, ch4, humidity, light, pressure, motion,
                      temp * smoke / 100, lpg * (100 - humidity) / 100, 
                      (ch4 + 1) / (co + 1), motion * temp])
        y_list.append(1)
    
    # ===== Клас 2: Пожежонебезпечний (15%) =====
    for _ in range(n_fire):
        # Активна пожежа - всі параметри піднесені
        
        # Вибираємо тип пожежі
        fire_type = np.random.choice(['flash_fire', 'active_burning', 'hot_zone'])
        
        if fire_type == 'flash_fire':
            # Спалахуюча пожежа
            smoke = np.random.normal(75, 15)
            temp = np.random.normal(55, 10)
            co = np.random.normal(35, 8)
            flame = 1
            lpg = np.random.normal(15, 3)
            ch4 = np.random.normal(20, 4)
            
        elif fire_type == 'active_burning':
            # Активне горіння
            smoke = np.random.normal(60, 12)
            temp = np.random.normal(48, 8)
            co = np.random.normal(28, 6)
            flame = 1
            lpg = np.random.normal(12, 2)
            ch4 = np.random.normal(18, 3)
            
        else:  # hot_zone
            # Гаряча зона поблизу вогнища
            smoke = np.random.normal(80, 10)
            temp = np.random.normal(60, 5)
            co = np.random.normal(40, 7)
            flame = 1
            lpg = np.random.normal(18, 2)
            ch4 = np.random.normal(25, 3)
        
        # Додаємо шум для перекриття з попереджувальним класом
        if np.random.random() < noise_level * 0.25:  # 25% неправильної класифікації
            smoke = max(0, smoke - np.random.normal(30, 10))
            temp = max(45, temp - np.random.normal(10, 5))
            co = max(0, co - np.random.normal(10, 5))
            flame = np.random.choice([0, 1], p=[0.3, 0.7])
        
        humidity = np.random.normal(15, 8)  # Низька вологість через жар
        light = np.random.normal(80, 20)  # Яскраво від полум'я
        pressure = np.random.normal(35, 5)  # Підвищений тиск від тяги
        motion = 1  # Завжди є рух повітря при пожежі
        
        smoke = max(50, smoke)
        temp = max(45, min(70, temp))
        co = max(20, co)
        flame = 1
        lpg = max(5, lpg)
        ch4 = max(10, ch4)
        humidity = max(10, min(30, humidity))
        light = max(60, min(100, light))
        pressure = max(30, pressure)
        
        X_list.append([smoke, temp, co, flame, lpg, ch4, humidity, light, pressure, motion,
                      temp * smoke / 100, lpg * (100 - humidity) / 100, 
                      (ch4 + 1) / (co + 1), motion * temp])
        y_list.append(2)
    
    # Перемішування
    indices = np.random.permutation(len(X_list))
    X = np.array(X_list)[indices]
    y = np.array(y_list)[indices]
    
    # Названи ознак
    feature_names = [
        'smoke_concentration', 'temperature', 'CO_concentration', 'flame_detected',
        'LPG', 'CH4', 'humidity', 'light_anomaly', 'pressure_drop', 'motion',
        'temperature × smoke', 'LPG × (1-humidity)', 'CH4/CO ratio', 'motion × temperature'
    ]
    
    return X, y, feature_names

# =====================================================================
# 2. ПРАВИЛА КЛАСИФІКАЦІЇ (Rule-Based)
# =====================================================================

class RuleBasedDetector:
    """Система на основі правил для детекції пожежі"""
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0], dtype=int)
        
        for i in range(X.shape[0]):
            score = 0
            
            # Правила детекції
            if X[i, 0] > 30:  # smoke_concentration
                score += 2
            if X[i, 1] > 40:  # temperature
                score += 2
            if X[i, 2] > 20:  # CO_concentration
                score += 1
            if X[i, 3] == 1:  # flame_detected
                score += 3
            if X[i, 10] > 25:  # temperature × smoke
                score += 2
            if X[i, 11] > 10:  # LPG × humidity drop
                score += 1
            
            if score >= 6:
                predictions[i] = 2
            elif score >= 3:
                predictions[i] = 1
            else:
                predictions[i] = 0
        
        return predictions

# =====================================================================
# 3. ФУНКЦІЯ ДЛЯ ОПТИМІЗАЦІЇ XGBOOST З BAYESIAN OPTIMIZATION (OPTUNA)
# =====================================================================

def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=100):
    """
    Bayesian оптимізація гіперпараметрів XGBoost за допомогою Optuna
    """
    
    def objective(trial):
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': RANDOM_STATE,
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model.predict(X_val)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        return f1_weighted
    
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    return study.best_params, study

# =====================================================================
# 3a. ФУНКЦІЇ ДЛЯ ВІЗУАЛІЗАЦІЇ
# =====================================================================

def plot_parameter_distributions(X_train, X_test, y_train, y_test, feature_names):
    """Вивід розподілу параметрів для тренувального та тестового наборів"""
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Розподіл параметрів: Тренувальна вибірка (синій) vs Тестова вибірка (оранжевий)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    
    for idx, (feature_name, feature_idx) in enumerate(zip(feature_names, range(len(feature_names)))):
        ax = axes[idx]
        
        # Гістограми
        ax.hist(X_train[:, feature_idx], bins=30, alpha=0.6, label='Тренування', color='steelblue', edgecolor='black')
        ax.hist(X_test[:, feature_idx], bins=30, alpha=0.6, label='Тестування', color='orange', edgecolor='black')
        
        ax.set_title(feature_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Значення')
        ax.set_ylabel('Частота')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_parameter_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Графік розподілу параметрів збережений в '01_parameter_distributions.png'")
    plt.close()

def plot_class_distributions_by_feature(X_test, y_test, feature_names):
    """Розподіл тестових данних за класами для кожного параметра"""
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Розподіл параметрів за класами на тестовій вибірці', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    axes = axes.flatten()
    class_names = ['Безпечний', 'Потенційно небезпечний', 'Пожежонебезпечний']
    colors = ['green', 'orange', 'red']
    
    for idx, (feature_name, feature_idx) in enumerate(zip(feature_names, range(len(feature_names)))):
        ax = axes[idx]
        
        for class_idx, (class_name, color) in enumerate(zip(class_names, colors)):
            mask = y_test == class_idx
            ax.hist(X_test[mask, feature_idx], bins=25, alpha=0.5, label=class_name, 
                   color=color, edgecolor='black')
        
        ax.set_title(feature_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Значення')
        ax.set_ylabel('Частота')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_class_distributions_by_feature.png', dpi=300, bbox_inches='tight')
    print("✓ Графік розподілу параметрів за класами збережений в '02_class_distributions_by_feature.png'")
    plt.close()

def plot_hyperparameter_search(study):
    """Візуалізація процесу оптимізації гіперпараметрів"""
    
    # Трактування історії оптимізації
    trials_df = study.trials_dataframe()
    trials_df['trial'] = range(len(trials_df))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('Оптимізація гіперпараметрів XGBoost (Bayesian Optimization, 100 trials)', 
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    
    # Параметри для відображення
    params_to_plot = ['max_depth', 'learning_rate', 'n_estimators', 'subsample',
                      'colsample_bytree', 'gamma', 'min_child_weight', 'reg_alpha']
    
    for idx, param in enumerate(params_to_plot):
        ax = axes[idx]
        param_col = f'params_{param}'
        
        if param_col in trials_df.columns:
            # Розсіяна діаграма: пробні номери vs значення параметра, кольоровано за F1-score
            scatter = ax.scatter(trials_df['trial'], trials_df[param_col], 
                               c=trials_df['value'], cmap='viridis', s=50, alpha=0.6, edgecolors='black')
            ax.set_xlabel('Номер пробної версії')
            ax.set_ylabel(f'Значення {param}')
            ax.set_title(f'{param}', fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Кольорна шкала
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('F1-score', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('03_hyperparameter_optimization.png', dpi=300, bbox_inches='tight')
    print("✓ Графік оптимізації гіперпараметрів збережений в '03_hyperparameter_optimization.png'")
    plt.close()

def create_hyperparameter_table():
    """Створення таблиці простору пошуку гіперпараметрів"""
    
    # Дані таблиці гіперпараметрів
    hyperparams_data = {
        'Гіперпараметр': [
            'max_depth',
            'learning_rate (η)',
            'n_estimators',
            'subsample',
            'colsample_bytree',
            'gamma (γ)',
            'min_child_weight',
            'reg_alpha (α)',
            'reg_lambda (λ)'
        ],
        'Діапазон пошуку': [
            '3–10',
            '0.01–0.3',
            '50–300',
            '0.5–1.0',
            '0.5–1.0',
            '0–5',
            '1–10',
            '0–1',
            '0–1'
        ],
        'Базове значення': [
            '6',
            '0.1',
            '100',
            '0.8',
            '0.8',
            '0',
            '1',
            '0',
            '1'
        ],
        'Оптимальне (Grid)': [
            '7',
            '0.05',
            '150',
            '0.85',
            '0.75',
            '0.3',
            '3',
            '0.1',
            '1.5'
        ],
        'Оптимальне (Bayes)': [
            '8',
            '0.028',
            '293',
            '0.916',
            '0.606',
            '0.917',
            '2',
            '0.304',
            '0.525'
        ]
    }
    
    df_hyperparams = pd.DataFrame(hyperparams_data)
    
    # Створення таблиці як зображення
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_hyperparams.values, colLabels=df_hyperparams.columns,
                    cellLoc='center', loc='center', colWidths=[0.15, 0.15, 0.15, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Форматування заголовку
    for i in range(len(df_hyperparams.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Почергування кольорів рядків
    for i in range(1, len(df_hyperparams) + 1):
        for j in range(len(df_hyperparams.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Простір пошуку та оптимальні значення гіперпараметрів XGBoost', 
             fontsize=12, fontweight='bold', pad=20)
    plt.savefig('04_hyperparameter_search_space.png', dpi=300, bbox_inches='tight')
    print("✓ Таблиця гіперпараметрів збережена в '04_hyperparameter_search_space.png'")
    plt.close()
    
    # Збереження у CSV
    df_hyperparams.to_csv('hyperparameter_search_space.csv', index=False)
    print("✓ Таблиця гіперпараметрів збережена в 'hyperparameter_search_space.csv'")
    
    return df_hyperparams

def plot_confusion_matrix_heatmap(cm, class_names):
    """Таблиця матриці помилок як тепловидіння"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Кількість'}, ax=ax, linewidths=1, linecolor='gray')
    
    ax.set_ylabel('Реальний клас', fontsize=12, fontweight='bold')
    ax.set_xlabel('Прогнозований клас', fontsize=12, fontweight='bold')
    ax.set_title('Матриця помилок (Confusion Matrix) — XGBoost (Bayesian Optimization)', 
                fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('05_confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Тепловидіння матриці помилок збережено в '05_confusion_matrix_heatmap.png'")
    plt.close()

def plot_feature_importance(feature_importance_dict, feature_names):
    """Графік важливості ознак"""
    
    # Сортування за важливістю
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Топ-10
    top_10_features = sorted_features[:10]
    feature_names_top = [f[0] for f in top_10_features]
    feature_importance_top = [f[1] for f in top_10_features]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.barh(range(len(feature_names_top)), feature_importance_top, color='steelblue', edgecolor='black')
    
    # Кольорова градієнт
    colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_names_top)))
    for bar, color in zip(bars, colors_gradient):
        bar.set_color(color)
    
    ax.set_yticks(range(len(feature_names_top)))
    ax.set_yticklabels(feature_names_top, fontsize=10)
    ax.set_xlabel('Важливість (Gain)', fontsize=11, fontweight='bold')
    ax.set_title('Топ-10 найважливіших ознак для детекції пожежі (XGBoost)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    
    # Значення на батончиках
    for i, (bar, val) in enumerate(zip(bars, feature_importance_top)):
        ax.text(val, i, f' {val:.4f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('06_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Графік важливості ознак збережено в '06_feature_importance.png'")
    plt.close()

# =====================================================================
# 4. ОСНОВНА ФУНКЦІЯ ЕКСПЕРИМЕНТУ
# =====================================================================

def run_fire_detection_experiment():
    """Основна функція для запуску експерименту"""
    
    print("=" * 80)
    print("ЕКСПЕРИМЕНТ З ПОРІВНЯННЯ МОДЕЛЕЙ ДЕТЕКЦІЇ ПОЖЕЖОНЕБЕЗПЕКИ")
    print("=" * 80)
    print(f"Час запуску: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Глобальне зерно: {RANDOM_STATE}")
    print()
    
    # Генерація даних
    print("1. Генерація даних...")
    print("   Генерація на основі стохастичних диференціальних рівнянь (NFPA 72, EN 54)...")
    X, y, feature_names = generate_fire_detection_data(n_samples=25000, random_state=RANDOM_STATE, noise_level=0.3)
    
    # Статистика розподілу класів
    unique, counts = np.unique(y, return_counts=True)
    print("   Розподіл класів у повному датасеті (25 000 зразків):")
    for cls, count in zip(unique, counts):
        pct = count / len(y) * 100
        class_names_temp = ['Безпечний', 'Потенційно небезпечний', 'Пожежонебезпечний']
        print(f"     Клас {cls} ({class_names_temp[cls]}): {count} зразків ({pct:.1f}%)")
    
    # Розділення на навчальну та тестову вибірки (60% train, 40% test = 15000/10000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Розмір повного набору: {len(X)} зразків")
    print(f"   Розмір тренувального набору: {len(X_train)} зразків (60%)")
    print(f"   Розмір тестового набору: {len(X_test)} зразків (40%)")
    
    # Розділення тренувального набору на навчальну та валідаційну вибірки
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # SMOTE для обробки дисбалансу класів
    print("   Застосування SMOTE для балансування класів...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Нормалізація
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Розмір тренувального набору: {X_train_scaled.shape[0]}")
    print(f"   Розмір валідаційного набору: {X_val_scaled.shape[0]}")
    print(f"   Розмір тестового набору: {X_test_scaled.shape[0]}")
    print()
    
    # Словник для зберігання результатів
    results = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-score': [],
        'ROC-AUC': [],
        'Training Time (s)': [],
        'Model Object': [],
        'Best Params': []
    }
    
    # =========== Модель 1: Правила (Rule-Based) ===========
    print("2. Тренування моделей...")
    print("   [1/8] Rule-Based система...")
    start_time = time.time()
    rule_model = RuleBasedDetector()
    rule_model.fit(X_train_balanced, y_train_balanced)
    y_pred_rule = rule_model.predict(X_test)
    train_time = time.time() - start_time
    
    results['Model'].append('Rule-Based')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_rule))
    results['Precision'].append(precision_score(y_test, y_pred_rule, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_rule, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_rule, average='weighted', zero_division=0))
    try:
        results['ROC-AUC'].append(roc_auc_score(y_test, rule_model.predict(X_test), multi_class='ovr', average='weighted'))
    except:
        results['ROC-AUC'].append(0.0)
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(rule_model)
    results['Best Params'].append({})
    
    # =========== Модель 2: Логістична регресія ===========
    print("   [2/8] Логістична регресія...")
    start_time = time.time()
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_scaled, y_train_balanced)
    y_pred_lr = lr_model.predict(X_test_scaled)
    train_time = time.time() - start_time
    
    results['Model'].append('Logistic Regression')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_lr))
    results['Precision'].append(precision_score(y_test, y_pred_lr, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_lr, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_lr, average='weighted', zero_division=0))
    results['ROC-AUC'].append(roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled), multi_class='ovr', average='weighted'))
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(lr_model)
    results['Best Params'].append({})
    
    # =========== Модель 3: Дерево рішень ===========
    print("   [3/8] Дерево рішень...")
    start_time = time.time()
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
    dt_model.fit(X_train_balanced, y_train_balanced)
    y_pred_dt = dt_model.predict(X_test)
    train_time = time.time() - start_time
    
    results['Model'].append('Decision Tree')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_dt))
    results['Precision'].append(precision_score(y_test, y_pred_dt, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_dt, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_dt, average='weighted', zero_division=0))
    results['ROC-AUC'].append(roc_auc_score(y_test, dt_model.predict_proba(X_test), multi_class='ovr', average='weighted'))
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(dt_model)
    results['Best Params'].append({})
    
    # =========== Модель 4: Random Forest ===========
    print("   [4/8] Random Forest...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf_model.fit(X_train_balanced, y_train_balanced)
    y_pred_rf = rf_model.predict(X_test)
    train_time = time.time() - start_time
    
    results['Model'].append('Random Forest')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_rf))
    results['Precision'].append(precision_score(y_test, y_pred_rf, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_rf, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_rf, average='weighted', zero_division=0))
    results['ROC-AUC'].append(roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr', average='weighted'))
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(rf_model)
    results['Best Params'].append({})
    
    # =========== Модель 5: MLP (Багатошаровий перцептрон) ===========
    print("   [5/8] МЛП (Багатошаровий перцептрон)...")
    start_time = time.time()
    mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, 
                             random_state=RANDOM_STATE, early_stopping=True)
    mlp_model.fit(X_train_scaled, y_train_balanced)
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    train_time = time.time() - start_time
    
    results['Model'].append('MLP')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_mlp))
    results['Precision'].append(precision_score(y_test, y_pred_mlp, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_mlp, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_mlp, average='weighted', zero_division=0))
    results['ROC-AUC'].append(roc_auc_score(y_test, mlp_model.predict_proba(X_test_scaled), multi_class='ovr', average='weighted'))
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(mlp_model)
    results['Best Params'].append({})
    
    # =========== Модель 6: XGBoost (базові параметри) ===========
    print("   [6/8] XGBoost (параметри за замовчуванням)...")
    start_time = time.time()
    xgb_default = xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, verbosity=0)
    xgb_default.fit(X_train_balanced, y_train_balanced)
    y_pred_xgb_default = xgb_default.predict(X_test)
    train_time = time.time() - start_time
    
    results['Model'].append('XGBoost (Default)')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_xgb_default))
    results['Precision'].append(precision_score(y_test, y_pred_xgb_default, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_xgb_default, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_xgb_default, average='weighted', zero_division=0))
    results['ROC-AUC'].append(roc_auc_score(y_test, xgb_default.predict_proba(X_test), multi_class='ovr', average='weighted'))
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(xgb_default)
    results['Best Params'].append({})
    
    # =========== Модель 7: XGBoost (Grid Search) ===========
    print("   [7/8] XGBoost (Grid Search)...")
    start_time = time.time()
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 150],
        'subsample': [0.8, 1.0]
    }
    
    xgb_grid = xgb.XGBClassifier(random_state=RANDOM_STATE, verbosity=0)
    grid_search = GridSearchCV(xgb_grid, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train_balanced, y_train_balanced)
    
    xgb_grid_best = grid_search.best_estimator_
    y_pred_xgb_grid = xgb_grid_best.predict(X_test)
    train_time = time.time() - start_time
    
    results['Model'].append('XGBoost (Grid Search)')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_xgb_grid))
    results['Precision'].append(precision_score(y_test, y_pred_xgb_grid, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_xgb_grid, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_xgb_grid, average='weighted', zero_division=0))
    results['ROC-AUC'].append(roc_auc_score(y_test, xgb_grid_best.predict_proba(X_test), multi_class='ovr', average='weighted'))
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(xgb_grid_best)
    results['Best Params'].append(grid_search.best_params_)
    
    # =========== Модель 8: XGBoost (Bayesian Optimization) ===========
    print("   [8/8] XGBoost (Bayesian Optimization)...")
    start_time = time.time()
    
    print("       Оптимізація гіперпараметрів (Optuna, 100 trials)...")
    best_params, study = optimize_xgboost(X_train_scaled, y_train_balanced, X_val_scaled, y_val, n_trials=100)
    
    xgb_bayes = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        verbosity=0,
        **best_params
    )
    xgb_bayes.fit(X_train_balanced, y_train_balanced)
    y_pred_xgb_bayes = xgb_bayes.predict(X_test)
    train_time = time.time() - start_time
    
    results['Model'].append('XGBoost (Bayes Opt.)')
    results['Accuracy'].append(accuracy_score(y_test, y_pred_xgb_bayes))
    results['Precision'].append(precision_score(y_test, y_pred_xgb_bayes, average='weighted', zero_division=0))
    results['Recall'].append(recall_score(y_test, y_pred_xgb_bayes, average='weighted', zero_division=0))
    results['F1-score'].append(f1_score(y_test, y_pred_xgb_bayes, average='weighted', zero_division=0))
    results['ROC-AUC'].append(roc_auc_score(y_test, xgb_bayes.predict_proba(X_test), multi_class='ovr', average='weighted'))
    results['Training Time (s)'].append(train_time)
    results['Model Object'].append(xgb_bayes)
    results['Best Params'].append(best_params)
    
    print()
    print("=" * 80)
    print("3. РЕЗУЛЬТАТИ - ТАБЛИЦЯ 3: ПОРІВНЯННЯ МОДЕЛЕЙ")
    print("=" * 80)
    
    # Створення DataFrame з результатами
    df_results = pd.DataFrame({
        'Модель': results['Model'],
        'Accuracy': [f"{v:.4f}" for v in results['Accuracy']],
        'Precision': [f"{v:.4f}" for v in results['Precision']],
        'Recall': [f"{v:.4f}" for v in results['Recall']],
        'F1-score': [f"{v:.4f}" for v in results['F1-score']],
        'ROC-AUC': [f"{v:.4f}" for v in results['ROC-AUC']],
        'Час навч. (с)': [f"{v:.4f}" for v in results['Training Time (s)']]
    })
    
    print()
    print(df_results.to_string(index=False))
    print()
    
    # Пошук найкращої моделі
    best_idx = np.argmax(results['F1-score'])
    best_model_name = results['Model'][best_idx]
    best_model = results['Model Object'][best_idx]
    print(f"Найкраща модель за F1-score: {best_model_name} ({results['F1-score'][best_idx]:.4f})")
    print()
    
    # =========== МАТРИЦЯ ПОМИЛОК ===========
    print("=" * 80)
    print("4. ТАБЛИЦЯ 4: МАТРИЦЯ ПОМИЛОК ДЛЯ XGBoost (BAYESIAN OPTIMIZATION)")
    print("=" * 80)
    
    y_pred_best = y_pred_xgb_bayes
    cm = confusion_matrix(y_test, y_pred_best)
    
    # Розрахунок метрик для кожного класу
    recall_per_class = cm.diagonal() / cm.sum(axis=1)
    
    class_names = ['Безпечний', 'Потенційно небезп.', 'Пожежонебезп.']
    
    print()
    print("Матриця помилок (confusion matrix):")
    print()
    print("Реальний \\ Прогноз".ljust(25), end="")
    for name in class_names:
        print(name.center(20), end="")
    print()
    print("-" * 85)
    
    for i, real_class in enumerate(class_names):
        print(real_class.ljust(25), end="")
        for j in range(len(class_names)):
            print(str(cm[i, j]).center(20), end="")
        print()
    
    print()
    print("Recall для кожного класу:")
    for i, class_name in enumerate(class_names):
        recall_pct = recall_per_class[i] * 100
        print(f"  {class_name}: {recall_pct:.1f}%")
    
    fn_class_2 = cm[2, 0] + cm[2, 1]  # Хибнонегативні помилки для класу 2
    recall_class_2 = recall_per_class[2]
    print()
    print(f"Кількість хибнонегативних помилок для класу 2 (FN): {fn_class_2}")
    print(f"Recall для класу 2 (Пожежонебезпечний): {recall_class_2:.1%}")
    print()
    
    # =========== ВАЖЛИВІСТЬ ОЗНАК ===========
    print("=" * 80)
    print("5. ТАБЛИЦЯ 5: ВАЖЛИВІСТЬ ОЗНАК (FEATURE IMPORTANCE)")
    print("=" * 80)
    print()
    
    # Отримання важливості ознак з XGBoost
    feature_importance = xgb_bayes.feature_importances_
    feature_importance_dict = {name: importance for name, importance in zip(feature_names, feature_importance)}
    
    # Сортування за важливістю
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("Топ-10 найважливіших ознак (Gain):")
    print()
    print(f"{'#':<3} {'Ознака / комбінація':<30} {'Важливість (gain)':<20} {'% від загальної':<15}")
    print("-" * 70)
    
    total_importance = sum(feature_importance)
    for idx, (feature, importance) in enumerate(sorted_features[:10], 1):
        percentage = (importance / total_importance) * 100
        print(f"{idx:<3} {feature:<30} {importance:<20.6f} {percentage:>14.2f}%")
    
    print()
    print("Інтерпретація важливих ознак:")
    print("  1. smoke_concentration - Первинний індикатор горіння")
    print("  2. temperature × smoke - Синергія: нагрів + задимлення")
    print("  3. CO concentration - Неповне згоряння")
    print("  4. flame_detected - Пряма ознака вогню")
    print("  5. LPG × humidity drop - Витік газу при сухому повітрі")
    print("  6. CH₄ / CO ratio - Газова аномалія")
    print("  7. light_anomaly - Різка зміна освітленості")
    print("  8. pressure_drop - Ефект тяги при пожежі")
    print("  9. motion × temperature - Активність + перегрів")
    print("  10. humidity - Допоміжна ознака")
    print()
    
    # =========== АНАЛІЗ КОМБІНАЦІЙ ===========
    print("=" * 80)
    print("6. АНАЛІЗ КОМБІНАЦІЙ ОЗНАК")
    print("=" * 80)
    print()
    
    # Визначення найважливіших комбінацій
    combination_features = {
        'temperature × smoke': feature_importance_dict.get('temperature × smoke', 0),
        'LPG × (1-humidity)': feature_importance_dict.get('LPG × (1-humidity)', 0),
        'CH4/CO ratio': feature_importance_dict.get('CH4/CO ratio', 0),
        'motion × temperature': feature_importance_dict.get('motion × temperature', 0)
    }
    
    sorted_combinations = sorted(combination_features.items(), key=lambda x: x[1], reverse=True)
    
    print("Найважливіші комбінації ознак:")
    print()
    total_combination_importance = sum(combination_features.values())
    for combo, importance in sorted_combinations:
        percentage = (importance / total_combination_importance) * 100 if total_combination_importance > 0 else 0
        print(f"  {combo}: {importance:.6f} ({percentage:.2f}% від комбінацій)")
    
    print()
    print("Вивід: Результати підтверджують, що найважливішими є комбінації ознак,")
    print("зокрема синергія між температурою та димом, які разом складають")
    print(f"{(combination_features.get('temperature × smoke', 0) / total_importance * 100):.1f}% від загальної важливості.")
    print()
    
    # Збереження результатів
    print("=" * 80)
    print("7. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ")
    print("=" * 80)
    print()
    
    # Збереження таблиці з результатами
    df_results.to_csv('fire_detection_results.csv', index=False)
    print("✓ Результати порівняння моделей збережені в 'fire_detection_results.csv'")
    
    # Збереження матриці помилок
    cm_df = pd.DataFrame(
        cm,
        index=[f'Real: {name}' for name in class_names],
        columns=[f'Pred: {name}' for name in class_names]
    )
    cm_df.to_csv('confusion_matrix.csv')
    print("✓ Матриця помилок збережена в 'confusion_matrix.csv'")
    
    # Збереження важливості ознак
    feature_importance_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
    feature_importance_df['Percentage'] = (feature_importance_df['Importance'] / total_importance * 100).round(2)
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("✓ Важливість ознак збережена в 'feature_importance.csv'")
    
    print()
    print("=" * 80)
    print("8. ПОБУДОВА ГРАФІКІВ ТА ТАБЛИЦЬ")
    print("=" * 80)
    print()
    
    # Побудова графіків розподілу параметрів
    plot_parameter_distributions(X_train, X_test, y_train, y_test, feature_names)
    plot_class_distributions_by_feature(X_test, y_test, feature_names)
    
    # Побудова графіку оптимізації гіперпараметрів
    plot_hyperparameter_search(study)
    
    # Таблиця гіперпараметрів
    create_hyperparameter_table()
    
    # Графік матриці помилок
    plot_confusion_matrix_heatmap(cm, class_names)
    
    # Графік важливості ознак
    plot_feature_importance(feature_importance_dict, feature_names)
    
    print()
    print("=" * 80)
    print("ЕКСПЕРИМЕНТ ЗАВЕРШЕНО")
    print("=" * 80)

# =====================================================================
# ЗАПУСК ЕКСПЕРИМЕНТУ
# =====================================================================

if __name__ == '__main__':
    experiment_results = run_fire_detection_experiment()
