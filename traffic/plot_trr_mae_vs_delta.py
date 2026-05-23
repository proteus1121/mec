"""
Побудова графіків залежності TRR і MAE від порогу δ
для ARIMA(2,1,2) реконструкції кожного сенсорного каналу.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def ar2_predict(history, window=30):
    win = np.array(history[-window:], dtype=float)
    if len(win) < 5:
        return history[-1]
    diff = np.diff(win)
    if len(diff) < 4:
        return history[-1]
    X = np.column_stack([diff[1:-1], diff[:-2], np.ones(len(diff)-2)])
    y = diff[2:]
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        d_pred = coef[0] * diff[-1] + coef[1] * diff[-2] + coef[2]
    except Exception:
        d_pred = diff[-1]
    return float(win[-1] + d_pred)


def arima_like_filter(series, threshold, warmup=25, window=30):
    n = len(series)
    sent_mask = np.zeros(n, dtype=bool)
    predictions = np.full(n, np.nan)
    history = list(series[:warmup])
    sent_mask[:warmup] = True

    for i in range(warmup, n):
        pred = ar2_predict(history, window=window)
        actual = series[i]
        mae = abs(actual - pred)
        predictions[i] = pred

        if mae > threshold:
            sent_mask[i] = True
            history.append(actual)
        else:
            history.append(pred)

    return sent_mask, predictions


def reconstruct(series, sent_mask, predictions):
    rec = np.where(sent_mask, series, predictions)
    return pd.Series(rec).ffill().bfill().values


def actual_threshold(series, delta_fraction):
    rng = float(series.max() - series.min())
    return delta_fraction * rng if rng > 1e-9 else delta_fraction


def trr_mae_for_threshold(series, delta_fraction, warmup=25, window=30):
    threshold = actual_threshold(series, delta_fraction)
    sent_mask, predictions = arima_like_filter(series, threshold, warmup, window)
    reconstructed = reconstruct(series, sent_mask, predictions)
    trr = (1 - sent_mask.sum() / len(series)) * 100
    mae = float(np.mean(np.abs(series - reconstructed)))
    rng = float(series.max() - series.min())
    mae_pct = float(mae / rng * 100) if rng > 0 else 0.0
    return trr, mae_pct, threshold


def make_sensor_data():
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
        on = int(np.random.uniform(1, 7))
        if j + off < N:
            motion[j + off:j + off + on] = 1
        j += off + on

    return {
        "a) Температура (°C)": td,
        "b) Вологість (%)":    hd,
        "c) LPG (ppm)":        ld,
        "d) CH4 (ppm)":        cd,
        "e) Дим (одн.)":       sd,
        "f) Рух (0/1)":        motion,
    }


def unit_from_sensor_name(sensor_name):
    if "°C" in sensor_name:
        return "°C"
    if "%" in sensor_name:
        return "%"
    if "ppm" in sensor_name:
        return "ppm"
    if "одн." in sensor_name:
        return "одн."
    if "0/1" in sensor_name:
        return ""
    return ""


def plot_trr_mae(sensor_name, delta_percents, trr_list, mae_list, ax):
    ax1 = ax
    ax2 = ax1.twinx()
    ax1.plot(delta_percents, trr_list, color="black", linestyle="-", label="TRR (%)")
    ax2.plot(delta_percents, mae_list, color="black", linestyle="--", label="MAE (%)")

    # Добавляем вертикальную пунктирную линию при δ = 0.5%
    ax1.axvline(x=0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="δ = 0.5%")
    
    # Находим точки пересечения при δ = 0.5%
    trr_at_05 = np.interp(0.5, delta_percents, trr_list)
    mae_at_05 = np.interp(0.5, delta_percents, mae_list)
    
    # Отмечаем точки пересечения
    ax1.scatter([0.5], [trr_at_05], color="black", s=100, zorder=5, label="_nolegend_")
    ax2.scatter([0.5], [mae_at_05], color="black", s=100, zorder=5, label="_nolegend_")

    ax1.set_xlabel("δ (%)")
    ax1.set_ylabel("TRR (%)", color="black")
    ax2.set_ylabel("MAE (%)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax1.set_title(sensor_name)
    ax1.grid(True, alpha=0.25)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=7)


if __name__ == "__main__":
    sensors = make_sensor_data()
    delta_fractions = np.linspace(0.001, 0.02, 40)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    for ax, (name, series) in zip(axes, sensors.items()):
        trr_list = []
        mae_list = []
        delta_percent_values = []
        for delta in delta_fractions:
            trr, mae_pct, _ = trr_mae_for_threshold(series, delta)
            trr_list.append(trr)
            mae_list.append(mae_pct)
            delta_percent_values.append(delta * 100)
        plot_trr_mae(name, delta_percent_values, trr_list, mae_list, ax)

    fig.suptitle(
        "Залежність TRR та MAE від порогу δ для ARIMA(2,1,2)\n", fontsize=14)
    output_path = "trr_mae_vs_delta_all_sensors.png"
    plt.savefig(output_path, dpi=150)
    print(f"Графік збережено: {output_path}")
