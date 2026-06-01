#!/usr/bin/env python3
"""
Проверка параметров кабеля, которые использует приложение
"""

import sys
sys.path.insert(0, r'c:\Users\Artem\Documents\calculations\mec')

from unified_app import NetworkGraph, CableCalculator
import numpy as np
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# СОЗДАНИЕ ПРИМЕРА
# ═════════════════════════════════════════════════════════════════════════════

g = NetworkGraph()

tx  = g.add_node('plc_tx', 120,  200, Z=75,  name='PLC-TX')
n1  = g.add_node('node',   320,  200,         name='Node')
ld  = g.add_node('load',   320,  380, Z=59,  name='Load')
rx  = g.add_node('plc_rx', 520,  200, Z=75,  name='PLC-RX')

# Инициализируем CableCalculator
cable_calc = CableCalculator()
cable_type = 'ШВВП 2×1.5'
material   = 'copper'

# Рассчитываем спектр параметров
cable_calc.calculate_params_spectrum(cable_type, material,
                                      freq_min_mhz=1.0, freq_max_mhz=30.0,
                                      n_points=500)

g.add_edge(tx,  n1,  L=5, cable_type=cable_type, material=material)
g.add_edge(n1,  rx,  L=5, cable_type=cable_type, material=material)
g.add_edge(n1,  ld,  L=5, cable_type=cable_type, material=material)

# ═════════════════════════════════════════════════════════════════════════════
# ПРОВЕРКА ПАРАМЕТРОВ
# ═════════════════════════════════════════════════════════════════════════════

print("=== ПАРАМЕТРЫ КАБЕЛЯ ===\n")
print("F [МГц] | α [Нп/км] | β [рад/км] | Z0 [Ом]")
print("-" * 50)

test_freqs = [1, 5, 10, 15, 20, 25, 30]
for f_mhz in test_freqs:
    p = cable_calc.get_params_at_freq(cable_type, material, f_mhz)
    if p:
        alpha_npm_km = p['alpha']  # дБ/км
        beta_rad_km = p['beta']
        z0 = p['Z0']
        print(f"{f_mhz:5.0f}   | {alpha_npm_km:9.2f} | {beta_rad_km:10.2f} | {z0:7.1f}")
    else:
        print(f"{f_mhz:5.0f}   | ERROR: не вдалося отримати параметри")

# ═════════════════════════════════════════════════════════════════════════════
# РОЗРАХУНОК ГРАФІКА
# ═════════════════════════════════════════════════════════════════════════════

freq_hz = np.logspace(np.log10(0.5e6), np.log10(30e6), 300)
att, phi, pnodes, pedges, branch_edges = g.build_cascade(freq_hz, cable_calc)

freq_mhz = freq_hz / 1e6

# ═════════════════════════════════════════════════════════════════════════════
# ГРАФІКИ
# ═════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.patch.set_facecolor('#0D1117')

# Загасання
ax1.set_facecolor('#161B22')
ax1.plot(freq_mhz, att, color='#58A6FF', linewidth=2)
ax1.set_ylabel('Загасання [дБ]', color='#C9D1D9', fontsize=11)
ax1.set_title('Загасання МБЕ (частотно-залежні параметри)', color='#C9D1D9', fontsize=12, fontweight='bold')
ax1.grid(True, color='#30363D', linestyle='--', linewidth=0.5)
ax1.tick_params(colors='#C9D1D9')
for spine in ax1.spines.values():
    spine.set_color('#30363D')

# Фаза
ax2.set_facecolor('#161B22')
ax2.plot(freq_mhz, phi, color='#58A6FF', linewidth=2)
ax2.set_ylabel('Фаза [рад]', color='#C9D1D9', fontsize=11)
ax2.set_xlabel('Частота [МГц]', color='#C9D1D9', fontsize=11)
ax2.set_title('Фаза МБЕ (частотно-залежні параметри)', color='#C9D1D9', fontsize=12, fontweight='bold')
ax2.grid(True, color='#30363D', linestyle='--', linewidth=0.5)
ax2.tick_params(colors='#C9D1D9')
for spine in ax2.spines.values():
    spine.set_color('#30363D')

plt.tight_layout()
plt.savefig('test_with_cable_calc.png', facecolor='#0D1117', dpi=150)
print("\n✓ Графік збережено у test_with_cable_calc.png")

print(f"\n=== РЕЗУЛЬТАТИ ===")
print(f"Діапазон частот: {freq_mhz[0]:.2f} - {freq_mhz[-1]:.1f} МГц")
print(f"Діапазон загасання: {np.min(att):.2f} - {np.max(att):.2f} дБ")
print(f"Діапазон фази: {np.min(phi):.3f} - {np.max(phi):.3f} рад")

plt.show()
