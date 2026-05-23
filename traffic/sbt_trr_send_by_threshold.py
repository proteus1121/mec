"""
SBT (Send By Threshold) з адаптивним порогом = 0.5% від діапазону датчика
— те саме що ARIMA та Kalman, для чесного порівняння.

Поріг для SBT: |x[i] - last_sent| > threshold
  де threshold = 0.5% * (max - min) датчика

Поріг для ARIMA/Kalman: |x[i] - predicted[i]| > threshold
  де threshold = 0.5% * (max - min) датчика

Різниця: SBT порівнює нове значення з ОСТАННІМ НАДІСЛАНИМ,
         ARIMA/Kalman — з ПРОГНОЗОМ МОДЕЛІ.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import os

np.random.seed(2024)
N = 500
THRESHOLD_PCT = 0.5   # % від (max - min) — як в ARIMA та Kalman


def adaptive_threshold(series):
    r = float(series.max() - series.min())
    return (THRESHOLD_PCT / 100.0) * r if r > 1e-6 else 0.5


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

hd = smooth(58 + 10 * np.cos(t/5) + 0.15 * np.random.randn(N), 4)
hd = inject_events(hd, [160, 370], -14, 50)
hd = np.clip(hd, 30, 88)

ld = smooth(218 + 8 * np.sin(t/6) + 0.5 * np.random.randn(N), 3)
ld = inject_events(ld, [130, 355], 280, 35)
ld = np.clip(ld, 180, 560)

cd = smooth(163 + 8 * np.sin(t/7 + 1) + 0.4 * np.random.randn(N), 3)
cd = inject_events(cd, [200, 415], 240, 30)
cd = np.clip(cd, 130, 450)

sd = smooth(95 + 10 * np.sin(t/5 + 2) + 0.3 * np.random.randn(N), 3)
sd = inject_events(sd, [148, 298, 470], 200, 25)
sd = np.clip(sd, 60, 340)

motion = np.zeros(N)
j = 0
while j < N:
    off = int(np.random.uniform(7, 22))
    on  = int(np.random.uniform(1, 7))
    if j + off < N:
        motion[j + off:j + off + on] = 1
    j += off + on

sensors = {
    "Температура (°C)": td,
    "Вологість (%)":    hd,
    "LPG (ppm)":        ld,
    "CH4 (ppm)":        cd,
    "Дим (одн.)":       sd,
    "Рух (0/1)":        motion,
}


def sbt_filter(series, threshold):
    """
    SBT: надсилаємо якщо |нове - останнє_надіслане| > threshold.
    Реконструкція: zero-order hold (тримаємо останнє надіслане).
    """
    n = len(series)
    sent_mask = np.zeros(n, dtype=bool)
    sent_mask[0] = True
    last_sent = series[0]

    for i in range(1, n):
        if abs(series[i] - last_sent) > threshold:
            sent_mask[i] = True
            last_sent = series[i]

    reconstructed = np.empty(n)
    last_value = series[0]
    for i in range(n):
        if sent_mask[i]:
            last_value = series[i]
        reconstructed[i] = last_value

    return sent_mask, reconstructed


print("=" * 80)
print("  SBT з адаптивним порогом = 0.5% від діапазону (чесне порівняння)")
print(f"  Поріг = {THRESHOLD_PCT}% від (max-min)  |  N={N}")
print("=" * 80)
print(f"{'Датчик':<22} {'Поріг (абс.)':>13} {'Надіслано':>10} {'TRR (%)':>8} {'MAE (%)':>8}")
print("-" * 80)

results = {}
for name, data in sensors.items():
    thr = adaptive_threshold(data)
    sm, reconstructed = sbt_filter(data, thr)
    sent = int(sm.sum())
    trr  = (1 - sent / N) * 100
    mae  = float(np.mean(np.abs(data - reconstructed)))
    rng  = float(data.max() - data.min())
    mae_pct = float(mae / rng * 100) if rng > 0 else 0.0
    results[name] = dict(
        data=data, sent_mask=sm, reconstructed=reconstructed,
        total=N, sent=sent, trr=trr, mae_pct=mae_pct, threshold=thr
    )
    print(f"{name:<22} {thr:>13.4f} {sent:>10} {trr:>8.1f} {mae_pct:>8.2f}")

reductions  = [v["trr"]     for v in results.values()]
rel_errors  = [v["mae_pct"] for v in results.values()]
print("-" * 80)
print(f"{'Середнє:':<22} {'':>13} {'':>10} {np.mean(reductions):>8.1f} {np.mean(rel_errors):>8.2f}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────
# ВІЗУАЛІЗАЦІЯ
# ─────────────────────────────────────────────────────────────────
C = dict(
    actual="#00d4ff", reconstructed="#ff6b35", sent="#00ff9f",
    bg="#0d1117", panel="#161b22", grid="#21262d",
    text="#e6edf3", subtext="#8b949e",
    bars=["#1f6feb","#238636","#d29922","#8957e5","#e85151","#39d353"],
)
plt.rcParams.update({
    "figure.facecolor": C["bg"], "axes.facecolor": C["panel"],
    "axes.edgecolor":   C["grid"], "axes.labelcolor": C["text"],
    "xtick.color":      C["subtext"], "ytick.color":  C["subtext"],
    "text.color":       C["text"],  "grid.color":     C["grid"],
    "grid.linewidth":   0.6, "font.family": "monospace", "font.size": 9,
})

sensor_names = list(results.keys())

fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
fig.suptitle(
    "IoT-система моніторингу середовища: метод SBT (Send By Threshold)\n"
    f"Адаптивний поріг MAE = {THRESHOLD_PCT}% від діапазону  •  "
    "Чесне порівняння з ARIMA та Kalman",
    fontsize=11.5, fontweight="bold", color=C["text"], y=0.987
)

gs_top = gridspec.GridSpec(3, 2, figure=fig,
                           top=0.93, bottom=0.42,
                           hspace=0.52, wspace=0.28,
                           left=0.06, right=0.97)

for idx, name in enumerate(sensor_names):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs_top[row, col])
    d  = results[name]
    x  = np.arange(N)

    ax.fill_between(x, d["data"], d["reconstructed"],
                    alpha=0.08, color=C["reconstructed"], zorder=1)
    ax.plot(x, d["data"], color=C["actual"], lw=1.0, alpha=0.9,
            label="Реальні дані", zorder=2)
    ax.plot(x, d["reconstructed"], color=C["reconstructed"],
            lw=0.8, alpha=0.65, ls="--", label="Реконструкція (ZOH)", zorder=3)
    ax.scatter(x[d["sent_mask"]], d["data"][d["sent_mask"]],
               s=7, color=C["sent"], zorder=4, alpha=0.75, label="Відправлено")

    ax.set_title(name, fontsize=9, color=C["text"], pad=4)
    ax.set_xlabel("Відлік", fontsize=7.5, color=C["subtext"])
    ax.grid(True, alpha=0.28)
    info = (f"↓{d['trr']:.1f}%  err={d['mae_pct']:.2f}%  "
            f"поріг={d['threshold']:.3f}")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=7.5, color=C["sent"], va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"], alpha=0.55))

    if idx == 0:
        ax.legend(loc="upper right", fontsize=7.5,
                  framealpha=0.35, facecolor=C["panel"])

gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                           top=0.37, bottom=0.06, wspace=0.35,
                           left=0.06, right=0.97)

# — Зменшення трафіку (TRR) —
ax_r = fig.add_subplot(gs_bot[0, 0])
short = [n.split(" ")[0] for n in sensor_names]
br = ax_r.barh(short, reductions, color=C["bars"], height=0.55, zorder=3)
# Показуємо очікуваний діапазон ARIMA/Kalman для порівняння
ax_r.axvline(54, color="#ff6b35", lw=1.3, ls="--", alpha=0.85, label="ARIMA min 54%")
ax_r.axvline(83, color="#ffca28", lw=1.3, ls="--", alpha=0.85, label="ARIMA max 83%")
ax_r.set_xlim(0, 100)
ax_r.xaxis.set_major_formatter(PercentFormatter())
ax_r.set_title("Зменшення трафіку (TRR)", fontsize=10, color=C["text"])
ax_r.grid(True, axis="x", alpha=0.32)
ax_r.legend(fontsize=7.5, framealpha=0.3, facecolor=C["panel"])
for b, v in zip(br, reductions):
    clr = C["sent"] if 40 <= v <= 90 else "#ff9944"
    ax_r.text(v + 0.5, b.get_y() + b.get_height()/2,
              f"{v:.1f}%", va="center", fontsize=8.5, color=clr)

# — Відносна похибка (MAE) —
ax_e = fig.add_subplot(gs_bot[0, 1])
be = ax_e.barh(short, rel_errors, color=C["bars"], height=0.55, zorder=3)
ax_e.axvline(3.1, color="#ff6b35", lw=1.5, ls="--", alpha=0.9, label="3.1% межа")
ax_e.xaxis.set_major_formatter(PercentFormatter())
ax_e.set_title("Відносна похибка (MAE)", fontsize=10, color=C["text"])
ax_e.set_xlim(0, max(max(rel_errors) * 1.45, 4))
ax_e.grid(True, axis="x", alpha=0.32)
ax_e.legend(fontsize=8, framealpha=0.3, facecolor=C["panel"])
for b, v in zip(be, rel_errors):
    clr = "#ff4444" if v > 3.1 else C["sent"]
    ax_e.text(v + 0.03, b.get_y() + b.get_height()/2,
              f"{v:.2f}%", va="center", fontsize=8.5, color=clr)

# — Зведена таблиця —
ax_t = fig.add_subplot(gs_bot[0, 2])
ax_t.axis("off")
rows = []
for name in sensor_names:
    d = results[name]
    ok_r = "✓" if 40 <= d["trr"] <= 90 else "~"
    ok_e = "✓" if d["mae_pct"] <= 3.1 else "!"
    rows.append([name.split(" ")[0],
                 f"{d['sent']}/{N}",
                 f"{d['trr']:.1f}% {ok_r}",
                 f"{d['mae_pct']:.2f}% {ok_e}"])
avg_sent = int(np.mean([results[n]["sent"] for n in sensor_names]))
rows.append(["Середнє", f"{avg_sent}/{N}",
             f"{np.mean(reductions):.1f}%",
             f"{np.mean(rel_errors):.2f}%"])

tbl = ax_t.table(cellText=rows,
                 colLabels=["Датчик", "Відправлено", "TRR", "MAE"],
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.1, 1.58)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(C["grid"])
    if r == 0:
        cell.set_facecolor("#1f6feb")
        cell.set_text_props(color="white", fontweight="bold")
    elif r == len(rows):
        cell.set_facecolor("#238636")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#1c2128" if r % 2 == 0 else C["panel"])
        cell.set_text_props(color=C["text"])
ax_t.set_title("Зведені результати", fontsize=10, color=C["text"], pad=8)

fig.text(
    0.5, 0.004,
    f"SBT: |x[i] − last_sent| > {THRESHOLD_PCT}% діапазону  •  "
    f"Реконструкція: zero-order hold  •  N={N}  •  Синтетичні IoT-дані",
    ha="center", fontsize=8, color=C["subtext"]
)

output_path = os.path.join(os.getcwd(), "iot_sbt_fair_result.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
plt.close()
print(f"\nГрафік збережено: {output_path}")