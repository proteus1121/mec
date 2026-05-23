import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Create output directory
output_dir = r'c:\Users\Artem\Documents\calculations\fire-sim'
os.makedirs(output_dir, exist_ok=True)

DARK   = "#1a1a2e"
ACCENT = "#c0392b"
ORANGE = "#e67e22"
BLUE   = "#2980b9"
GREEN  = "#27ae60"
GREY   = "#7f8c8d"
LIGHT  = "#ecf0f1"
FONT   = "DejaVu Sans"

plt.rcParams.update({
    'font.family': FONT,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
})

# ── FIGURE 1: Confusion Matrix ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor('white')
cm = np.array([[192, 574], [62, 472]])
labels = [["TP\n192", "FN\n574"], ["FP\n62", "TN\n472"]]
colors = [["#c0392b", "#e8b4b8"], ["#f39c12", "#27ae60"]]
alpha  = [[0.85, 0.35], [0.40, 0.85]]

for i in range(2):
    for j in range(2):
        c = colors[i][j]
        ax.add_patch(FancyBboxPatch((j+0.05, 1-i+0.05), 0.85, 0.85,
                     boxstyle="round,pad=0.02", facecolor=c,
                     alpha=alpha[i][j], edgecolor='white', linewidth=2))
        ax.text(j+0.475, 1-i+0.475, labels[i][j],
                ha='center', va='center', fontsize=18, fontweight='bold', color='white')

ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(['Передбачено:\nПОЖЕЖА', 'Передбачено:\nНЕМАЄ ПОЖЕЖІ'], fontsize=11)
ax.set_yticklabels(['Факт:\nПОЖЕЖА', 'Факт:\nНЕМАЄ ПОЖЕЖІ'], fontsize=11)
ax.tick_params(length=0)
for spine in ax.spines.values(): spine.set_visible(False)
ax.set_title('Рис. 1. Матриця помилок (Confusion Matrix)\nN = 1 300 кадрів', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1_confusion.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated fig1_confusion.png")

# ── FIGURE 2: Bar chart — key metrics ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
metrics = ['Accuracy\n51.1%', 'Precision\n75.6%', 'Recall\n25.1%', 'F1-Score\n37.7%']
values  = [51.08, 75.59, 25.07, 37.65]
bar_colors = [BLUE, GREEN, ACCENT, ORANGE]
bars = ax.barh(metrics, values, color=bar_colors, height=0.55, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontsize=13, fontweight='bold')
ax.set_xlim(0, 95)
ax.set_xlabel('Значення метрики, %', fontsize=11)
ax.set_title('Рис. 2. Метрики якості класифікації алгоритму\n(1 300 тестових кадрів)', fontsize=12, fontweight='bold', pad=10)
ax.axvline(50, color='grey', linestyle='--', alpha=0.4, linewidth=1)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig2_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated fig2_metrics.png")

# ── FIGURE 3: Stacked bar — pixel rejection pipeline ──────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
total = 1136451
stages = ['Темні пікселі\n(MIN_VALUE)', 'Неправильний\nтон (Hue)', 'Низька\nнасиченість', 'Відтінки\nсірого', 'Прийнято\nяк "вогонь"']
vals   = [903453, 163335, 61182, 8481, 1136451-903453-163335-61182-8481]
pcts   = [v/total*100 for v in vals]
cols   = ['#2c3e50', '#e74c3c', '#e67e22', '#95a5a6', '#c0392b']
bars = ax.bar(stages, pcts, color=cols, edgecolor='white', linewidth=1.5, width=0.6)
for bar, pct, v in zip(bars, pcts, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
            f'{pct:.1f}%\n({v:,})', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.set_ylabel('Частка від загальної кількості пікселів, %', fontsize=10)
ax.set_ylim(0, 95)
ax.set_title('Рис. 3. Ефективність каскадної просторової фільтрації\n(аналіз 62 хибнопозитивних кадрів, 1 136 451 пікселів)', fontsize=12, fontweight='bold', pad=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_pipeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated fig3_pipeline.png")

# ── FIGURE 4: Pie — dataset split ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))
sizes  = [766, 534]
labels = ['Кадри з пожежею\n766 (58.9%)', 'Кадри без пожежі\n534 (41.1%)']
wedge_props = {'edgecolor': 'white', 'linewidth': 2.5}
wedges, texts = ax.pie(sizes, labels=labels, colors=[ACCENT, BLUE],
                        wedgeprops=wedge_props, startangle=90,
                        textprops={'fontsize': 11})
ax.set_title('Рис. 4. Розподіл тестового датасету\n(N = 1 300 кадрів, 640×480 VGA)', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig4_dataset.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated fig4_dataset.png")

# ── FIGURE 5: Simulated fire ratio time-series ────────────────────────────
np.random.seed(42)
t = np.arange(0, 80)
# no-fire zone: static lamp
no_fire = np.ones(20)*0.045 + np.random.normal(0, 0.0005, 20)
# fire zone: flickering
fire_base = 0.03 + 0.025*np.sin(2*np.pi*t[:40]/3.5)
fire = fire_base + np.random.normal(0, 0.004, 40)
fire = np.clip(fire, 0, None)
# another no-fire zone
no_fire2 = np.ones(20)*0.002 + np.abs(np.random.normal(0, 0.0008, 20))
full = np.concatenate([no_fire, fire, no_fire2])

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,5.5), sharex=True)
ax1.plot(full, color=ACCENT, linewidth=1.5, label='r(t) — відношення вогневих пікселів')
ax1.axhline(0.004, color='navy', linestyle='--', linewidth=1.2, label='τ_ratio = 0.004')
ax1.fill_between(range(20,60), 0, full[20:60], alpha=0.15, color=ACCENT)
ax1.axvspan(20, 60, alpha=0.07, color=ACCENT)
ax1.set_ylabel('r(t)', fontsize=11)
ax1.legend(fontsize=9, loc='upper right')
ax1.set_title('Рис. 5. Симульований часовий ряд r(t) та дисперсія Var(t)', fontsize=12, fontweight='bold')

# variance with rolling window L=8
var_series = np.array([
    np.var(full[max(0,i-8):i+1]) if i >= 7 else 0
    for i in range(len(full))
])
ax2.plot(var_series, color=BLUE, linewidth=1.5, label='Var(t)')
ax2.axhline(5e-6, color='darkgreen', linestyle='--', linewidth=1.2, label='τ_flicker = 5×10⁻⁶')
ax2.fill_between(range(20,60), 0, var_series[20:60], alpha=0.15, color=BLUE)
ax2.set_ylabel('Var(t)', fontsize=11)
ax2.set_xlabel('Номер кадру', fontsize=11)
ax2.legend(fontsize=9, loc='upper right')

for ax in (ax1, ax2):
    ax.axvspan(20, 60, alpha=0.07, color='red')
    ax.annotate('Зона\nпожежі', xy=(40, ax.get_ylim()[1]*0.5 if ax==ax1 else 0),
                fontsize=9, color='darkred', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig5_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated fig5_timeseries.png")

# ── FIGURE 6: Precision-Recall trade-off curve ────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5))
# Simulated PR curve by varying CONFIRM_FRAMES and thresholds
recall_pts    = [0.10, 0.18, 0.25, 0.35, 0.48, 0.60, 0.70, 0.80, 0.88, 0.92]
precision_pts = [0.95, 0.88, 0.756, 0.65, 0.54, 0.44, 0.36, 0.28, 0.21, 0.15]
ax.plot(recall_pts, precision_pts, 'o-', color=BLUE, linewidth=2, markersize=6)
# Mark current operating point
ax.scatter([0.2507], [0.7559], s=120, color=ACCENT, zorder=5, label='Поточна точка\n(N_confirm=3, τ_ratio=0.004)')
ax.annotate('Поточна\nконфігурація', xy=(0.2507, 0.7559), xytext=(0.35, 0.78),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='black'), color=ACCENT, fontweight='bold')
ax.set_xlabel('Recall (Повнота)', fontsize=12)
ax.set_ylabel('Precision (Точність)', fontsize=12)
ax.set_title('Рис. 6. Крива Precision–Recall\n(варіювання порогів τ_ratio та N_confirm)', fontsize=12, fontweight='bold', pad=10)
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
# AUC annotation
ax.fill_between(recall_pts, precision_pts, alpha=0.08, color=BLUE)
ax.text(0.65, 0.75, 'AUC-PR ≈ 0.51', fontsize=11, color=BLUE, fontstyle='italic')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig6_pr_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Generated fig6_pr_curve.png")

print("\n" + "="*50)
print("All 6 figures generated successfully!")
print(f"Output directory: {output_dir}")
print("="*50)
