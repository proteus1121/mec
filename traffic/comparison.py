# Comparison of temperature time series predictions

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from filterpy.kalman import KalmanFilter

# --- Generate temperature data ---
np.random.seed(2024)
N = 500

def smooth(x, w=5):
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

def inject_events(base, positions, magnitude, width):
    sig = base.copy()
    for p in map(int, positions):
        half = width // 2
        pulse = np.concatenate([
            np.linspace(0, magnitude, half),
            np.linspace(magnitude, 0, width - half)
        ])
        start = max(0, p - half)
        end = min(len(sig), start + len(pulse))
        sig[start:end] += pulse[:end - start]
    return sig

t = np.linspace(0, 6 * np.pi, N)
td = smooth(22 + 4 * np.sin(t/4) + 0.08 * np.random.randn(N), 3)
td = inject_events(td, [100, 270, 420], 7, 40)
td = np.clip(td, 16, 35)

# --- ARIMA prediction ---
# Fit ARIMA(2,1,2) on the entire series for in-sample predictions
model = ARIMA(td, order=(2, 1, 2))
model_fit = model.fit()
arima_pred = model_fit.fittedvalues

# --- Kalman filter prediction ---
# Simple 1D Kalman filter for position with constant velocity
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([td[0], 0.])  # initial state: position and velocity
kf.F = np.array([[1., 1.], [0., 1.]])  # state transition
kf.H = np.array([[1., 0.]])  # measurement function
kf.P *= 1000.  # initial uncertainty
kf.R = 0.1  # measurement noise
kf.Q = np.array([[0.01, 0.01], [0.01, 0.01]])  # process noise

kalman_pred = np.zeros(N)
kalman_pred[0] = td[0]

for i in range(1, N):
    kf.predict()
    kf.update(td[i])
    kalman_pred[i] = kf.x[0]

# --- Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1, 1]})

# Top plot: Real data and predictions
ax1.plot(td, label='Реальний ряд', linestyle='-', color='black', linewidth=1)
ax1.plot(arima_pred, label='Прогноз ARIMA', linestyle='--', color='black', linewidth=3)
ax1.plot(kalman_pred, label='Прогноз Калмана', linestyle='', color='black', linewidth=0,
         marker='o', markersize=4, markerfacecolor='none', markeredgewidth=1.5, markevery=5)

ax1.set_ylabel('Температура (°C)', fontsize=11)
ax1.set_title('Порівняння прогнозів: Реальний ряд, ARIMA та Фільтр Калмана', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 35)

# Middle plot: Difference real - ARIMA
diff_real_arima = np.array(td) - np.array(arima_pred)
ax2.fill_between(range(len(diff_real_arima)), diff_real_arima, alpha=0.5, color='black')
ax2.plot(diff_real_arima, color='black', linewidth=1.5, label='Реальний - ARIMA')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

ax2.set_ylabel('Різниця (°C)', fontsize=11)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# Bottom plot: Difference real - Kalman
diff_real_kalman = np.array(td) - np.array(kalman_pred)
ax3.fill_between(range(len(diff_real_kalman)), diff_real_kalman, alpha=0.5, color='black')
ax3.plot(diff_real_kalman, color='black', linewidth=1.5, label='Реальний - Калман')
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

ax3.set_xlabel('Час', fontsize=11)
ax3.set_ylabel('Різниця (°C)', fontsize=11)
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = "temperature_comparison.png"
plt.savefig(output_path, dpi=150, facecolor='white')
print(f"Plot saved to {output_path}")