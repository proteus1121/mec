"""
Порівняння трафіку IoT-системи моніторингу середовища
із використанням предиктивної фільтрації Kalman.

Датчики: температура, вологість, LPG, CH4, дим, рух
Метод:   одновимірний Калман-фільтр з моделлю стану [x, dx]
         + адаптивний поріг MAE = 0.5% від діапазону датчика
Ціль:    зменшення трафіку при низькій відносній похибці
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from filterpy.kalman import KalmanFilter
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 1. ГЕНЕРАЦІЯ РЕАЛІСТИЧНИХ IoT-ДАНИХ
# ─────────────────────────────────────────────────────────────────

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

hd = smooth(58 + 10 * np.cos(t/5) + 0.15 * np.random.randn(N), 4)
hd = inject_events(hd, [160, 370], -14, 50)
hd = np.clip(hd, 30, 88)

ld = smooth(218 + 8 * np.sin(t/6) + 0.5 * np.random.randn(N), 3)
ld = inject_events(ld, [130, 355], 280, 35)
ld = np.clip(ld, 180, 560)

cd = smooth(163 + 8 * np.sin(t/7+1) + 0.4 * np.random.randn(N), 3)
cd = inject_events(cd, [200, 415], 240, 30)
cd = np.clip(cd, 130, 450)

sd = smooth(95 + 10 * np.sin(t/5+2) + 0.3 * np.random.randn(N), 3)
sd = inject_events(sd, [148, 298, 470], 200, 25)
sd = np.clip(sd, 60, 340)

motion = np.zeros(N)
j = 0
while j < N:
    off = int(np.random.uniform(7, 22))
    on = int(np.random.uniform(1, 7))
    if j + off < N:
        motion[j + off:j + off + on] = 1
    j += off + on
motion = motion[:N]

sensors = {
    "Температура (°C)": td,
    "Вологість (%)":    hd,
    "LPG (ppm)":        ld,
    "CH4 (ppm)":        cd,
    "Дим (одн.)":       sd,
    "Рух (0/1)":        motion,
}

# ─────────────────────────────────────────────────────────────────
# 2. KALMAN-ПРЕДИКТОР
# ─────────────────────────────────────────────────────────────────

THRESHOLD_PCT = 0.5   # % від (max-min) датчика
WARMUP        = 25    # стартовий буфер


def adaptive_threshold(series):
    r = float(series.max() - series.min())
    return (THRESHOLD_PCT / 100.0) * r if r > 1e-6 else 0.5


def make_kalman_filter(initial_value, initial_velocity=0.0, measurement_var=1.0):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[initial_value], [initial_velocity]], dtype=float)
    kf.F = np.array([[1., 1.], [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 5.0
    kf.R = np.array([[measurement_var]])
    q = max(measurement_var * 0.01, 1e-4)
    kf.Q = np.array([[q, 0.], [0., q]])
    return kf


def kalman_filter(series, threshold):
    n = len(series)
    sent_mask = np.zeros(n, dtype=bool)
    predictions = np.full(n, np.nan)
    errors = np.full(n, np.nan)

    history = list(series[:WARMUP])
    sent_mask[:WARMUP] = True

    if len(history) >= 2:
        init_vel = history[-1] - history[-2]
    else:
        init_vel = 0.0

    measurement_var = max(np.var(np.diff(series)), 0.01)
    kf = make_kalman_filter(history[-1], init_vel, measurement_var)

    for i in range(WARMUP, n):
        kf.predict()
        pred = float(kf.x[0, 0])
        actual = series[i]
        mae = abs(actual - pred)

        predictions[i] = pred
        errors[i] = mae

        if mae > threshold:
            sent_mask[i] = True
            kf.update(np.array([[actual]]))
            history.append(actual)
        else:
            history.append(actual)

    return sent_mask, predictions, errors


def relative_monitoring_error(series, sent_mask, predictions):
    reconstructed = np.where(sent_mask, series, predictions)
    s = pd.Series(reconstructed).ffill().bfill().values
    rng = float(series.max() - series.min())
    return float(np.mean(np.abs(series - s)) / rng * 100) if rng > 0 else 0.0


if __name__ == "__main__":
    print("=" * 72)
    print("  IoT Kalman-подібна предиктивна фільтрація трафіку")
    print(f"  Поріг = {THRESHOLD_PCT}% від діапазону | N={N} | Прогрів={WARMUP}")
    print("=" * 72)
    print(f"{'Датчик':<22} {'Поріг':>8} {'Надіслано':>10} {'Зменшення':>11} {'Похибка':>9}")
    print("-" * 72)

    results = {}
    for name, data in sensors.items():
        thr = adaptive_threshold(data)
        sm, preds, errs = kalman_filter(data, thr)
        sent = int(sm.sum())
        reduction = (1 - sent / N) * 100
        rel_err = relative_monitoring_error(data, sm, preds)
        results[name] = dict(
            data=data, sent_mask=sm, predictions=preds, errors=errs,
            total=N, sent=sent, reduction=reduction,
            rel_err=rel_err, threshold=thr
        )
        bar = "█" * int(reduction / 4)
        print(f"{name:<22} {thr:>8.3f} {sent:>10} {reduction:>9.1f}%  {rel_err:>7.2f}%  {bar}")

    reductions = [v["reduction"] for v in results.values()]
    rel_errors = [v["rel_err"] for v in results.values()]
    print("-" * 72)
    print(f"{'Середнє:':<22} {'':>8} {'':>10} {np.mean(reductions):>9.1f}%  {np.mean(rel_errors):>7.2f}%")
    print(f"{'Діапазон:':<22} {'':>8} {'':>10} {min(reductions):.1f}–{max(reductions):.1f}%  max {max(rel_errors):.2f}%")
    print("=" * 72)

    C = dict(
        actual="#00d4ff", predicted="#ff6b35", sent="#00ff9f",
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

    fig = plt.figure(figsize=(18, 14), facecolor=C["bg"])
    fig.suptitle(
        "IoT-система моніторингу середовища: предиктивна фільтрація трафіку на базі Kalman\n"
        f"Адаптивний поріг MAE = {THRESHOLD_PCT}% від діапазону датчика  •  "
        "Ціль: зменшення трафіку при низькій відносній похибці",
        fontsize=11.5, fontweight="bold", color=C["text"], y=0.987
    )

    gs_top = gridspec.GridSpec(3, 2, figure=fig,
                               top=0.93, bottom=0.42,
                               hspace=0.52, wspace=0.28,
                               left=0.06, right=0.97)

    sensor_names = list(results.keys())
    for idx, name in enumerate(sensor_names):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs_top[row, col])
        d = results[name]
        x = np.arange(N)
        v = ~np.isnan(d["predictions"])

        ax.fill_between(x[v], d["data"][v], d["predictions"][v],
                        alpha=0.08, color=C["predicted"], zorder=1)
        ax.plot(x, d["data"], color=C["actual"], lw=1.0, alpha=0.9,
                label="Реальні дані", zorder=2)
        ax.plot(x[v], d["predictions"][v], color=C["predicted"],
                lw=0.8, alpha=0.65, ls="--", label="Прогноз Kalman", zorder=3)
        ax.scatter(x[d["sent_mask"]], d["data"][d["sent_mask"]],
                   s=7, color=C["sent"], zorder=4, alpha=0.75, label="Передано")

        ax.set_title(name, fontsize=9, color=C["text"], pad=4)
        ax.set_xlabel("Відлік", fontsize=7.5, color=C["subtext"])
        ax.grid(True, alpha=0.28)
        info = f"↓{d['reduction']:.1f}%  err={d['rel_err']:.2f}%  поріг={d['threshold']:.3f}"
        ax.text(0.02, 0.97, info, transform=ax.transAxes,
                fontsize=7.5, color=C["sent"], va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=C["bg"], alpha=0.55))

        if idx == 0:
            ax.legend(loc="upper right", fontsize=7.5,
                      framealpha=0.35, facecolor=C["panel"])

    gs_bot = gridspec.GridSpec(1, 3, figure=fig,
                               top=0.37, bottom=0.06, wspace=0.35,
                               left=0.06, right=0.97)

    ax_r = fig.add_subplot(gs_bot[0, 0])
    short = [n.split(" ")[0] for n in sensor_names]
    br = ax_r.barh(short, reductions, color=C["bars"], height=0.55, zorder=3)
    ax_r.axvline(57, color="#ff6b35", lw=1.3, ls="--", alpha=0.85, label="57% мін.")
    ax_r.axvline(74, color="#ffca28", lw=1.3, ls="--", alpha=0.85, label="74% макс.")
    ax_r.set_xlim(0, 100)
    ax_r.xaxis.set_major_formatter(PercentFormatter())
    ax_r.set_title("Зменшення трафіку", fontsize=10, color=C["text"])
    ax_r.grid(True, axis="x", alpha=0.32)
    ax_r.legend(fontsize=8, framealpha=0.3, facecolor=C["panel"])
    for b, v in zip(br, reductions):
        clr = C["sent"] if 50 <= v <= 85 else "#ff9944"
        ax_r.text(v+0.5, b.get_y()+b.get_height()/2,
                  f"{v:.1f}%", va="center", fontsize=8.5, color=clr)

    ax_e = fig.add_subplot(gs_bot[0, 1])
    be = ax_e.barh(short, rel_errors, color=C["bars"], height=0.55, zorder=3)
    ax_e.axvline(3.1, color="#ff6b35", lw=1.5, ls="--", alpha=0.9, label="3.1% межа")
    ax_e.xaxis.set_major_formatter(PercentFormatter())
    ax_e.set_title("Відносна похибка моніторингу", fontsize=10, color=C["text"])
    ax_e.set_xlim(0, max(max(rel_errors)*1.45, 4))
    ax_e.grid(True, axis="x", alpha=0.32)
    ax_e.legend(fontsize=8, framealpha=0.3, facecolor=C["panel"])
    for b, v in zip(be, rel_errors):
        clr = "#ff4444" if v > 3.1 else C["sent"]
        ax_e.text(v+0.03, b.get_y()+b.get_height()/2,
                  f"{v:.2f}%", va="center", fontsize=8.5, color=clr)

    ax_t = fig.add_subplot(gs_bot[0, 2])
    ax_t.axis("off")
    rows = []
    for name in sensor_names:
        d = results[name]
        ok_r = "✓" if 50 <= d["reduction"] <= 85 else "~"
        ok_e = "✓" if d["rel_err"] <= 3.1 else "!"
        rows.append([name.split(" ")[0],
                     f"{d['sent']}/{N}",
                     f"{d['reduction']:.1f}% {ok_r}",
                     f"{d['rel_err']:.2f}% {ok_e}"])
    avg_sent = int(np.mean([results[n]["sent"] for n in sensor_names]))
    rows.append(["Середнє", f"{avg_sent}/{N}",
                 f"{np.mean(reductions):.1f}%", f"{np.mean(rel_errors):.2f}%"])

    tbl = ax_t.table(cellText=rows,
                      colLabels=["Датчик","Надіслано","Зменш.","Похибка"],
                      loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.58)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C["grid"])
        if r == 0:
            cell.set_facecolor("#1f6feb"); cell.set_text_props(color="white", fontweight="bold")
        elif r == len(rows):
            cell.set_facecolor("#238636"); cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#1c2128" if r%2==0 else C["panel"])
            cell.set_text_props(color=C["text"])
    ax_t.set_title("Зведені результати", fontsize=10, color=C["text"], pad=8)

    fig.text(0.5, 0.004,
             f"Kalman-фільтр з моделлю [x, dx]  •  MAE={THRESHOLD_PCT}% від діапазону  "
             f"•  Прогрів={WARMUP}  •  N={N}  •  Синтетичні IoT-дані",
             ha="center", fontsize=8, color=C["subtext"])

    output_path = "iot_kalman_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    plt.close()
    print(f"\nГрафік збережено: {output_path}")
