# Re-run the computation and plot as requested

import numpy as np
import matplotlib.pyplot as plt

# --- Generate temperature data (same as script) ---
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

# --- Model params ---
WARMUP = 25
WINDOW = 30

def ar2_predict(history):
    win = np.array(history[-WINDOW:], dtype=float)
    if len(win) < 5:
        return history[-1]
    diff = np.diff(win)
    if len(diff) < 4:
        return history[-1]
    X = np.column_stack([diff[1:-1], diff[:-2], np.ones(len(diff)-2)])
    y = diff[2:]
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        d_pred = coef[0]*diff[-1] + coef[1]*diff[-2] + coef[2]
    except:
        d_pred = diff[-1]
    return float(win[-1] + d_pred)

def run(series, threshold):
    n = len(series)
    sent = np.zeros(n, bool)
    pred = np.full(n, np.nan)
    hist = list(series[:WARMUP])
    sent[:WARMUP] = True

    for i in range(WARMUP, n):
        p = ar2_predict(hist)
        pred[i] = p
        if abs(series[i] - p) > threshold:
            sent[i] = True
            hist.append(series[i])
        else:
            hist.append(p)

    trr = (1 - sent.sum()/n) * 100

    rec = np.where(sent, series, pred)
    rec = np.nan_to_num(rec, nan=series.mean())
    mae = np.mean(np.abs(series - rec))

    return trr, mae

# --- Sweep delta ---
deltas = np.linspace(0.01, 1.0, 40)
trr_list = []
mae_list = []

for d in deltas:
    trr, mae = run(td, d)
    trr_list.append(trr)
    mae_list.append(mae)

# Print some values for verification
print(f"deltas: {deltas[:5]} ... {deltas[-5:]}")
print(f"trr_list: {trr_list[:5]} ... {trr_list[-5:]}")
print(f"mae_list: {mae_list[:5]} ... {mae_list[-5:]}")

# Find indices for specific deltas
import numpy as np
idx_01 = np.argmin(np.abs(deltas - 0.1))
idx_02 = np.argmin(np.abs(deltas - 0.2))
idx_03 = np.argmin(np.abs(deltas - 0.3))
idx_10 = np.argmin(np.abs(deltas - 1.0))

print(f"At δ=0.1 ({deltas[idx_01]:.3f}): TRR={trr_list[idx_01]:.2f}%, MAE={mae_list[idx_01]:.3f}°C")
print(f"At δ=0.2 ({deltas[idx_02]:.3f}): TRR={trr_list[idx_02]:.2f}%, MAE={mae_list[idx_02]:.3f}°C")
print(f"At δ=0.3 ({deltas[idx_03]:.3f}): TRR={trr_list[idx_03]:.2f}%, MAE={mae_list[idx_03]:.3f}°C")
print(f"At δ=1.0 ({deltas[idx_10]:.3f}): TRR={trr_list[idx_10]:.2f}%, MAE={mae_list[idx_10]:.3f}°C")

# --- Plot ---
fig, ax1 = plt.subplots()

line1 = ax1.plot(deltas, trr_list, 'b-', label='TRR (%)')
ax1.set_xlabel("δ (°C)")
ax1.set_ylabel("TRR (%)", color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
line2 = ax2.plot(deltas, mae_list, 'r-', label='MAE (°C)')
ax2.set_ylabel("MAE (°C)", color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title("TRR та MAE від порогу δ (Температура)")

# Add legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# Save file
output_path = "trr_mae_vs_delta_temperature.png"
plt.savefig(output_path)

output_path