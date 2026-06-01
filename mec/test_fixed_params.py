#!/usr/bin/env python3
"""
Простой тест - расчет МБЕ с фиксированными параметрами кабеля
"""

import numpy as np
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════════════════════
# ПАРАМЕТРИ МЕРЕЖІ
# ═════════════════════════════════════════════════════════════════════════════

L1_seg = 5.0     # м (TX -> Node)
L2_seg = 5.0     # м (Node -> RX)  
L_branch = 5.0   # м (Node -> Load)
Zs = 75.0        # Ом (TX)
ZL = 75.0        # Ом (RX)
Z_load = 59.0    # Ом (Load)

# Фіксовані параметри кабелю ШВВП 2×1.5 на 24 МГц
alpha = 16.600460      # Нп/км (загасання)
beta = 1087.281047     # рад/км (фаза)
Z0 = 66.9              # Ом (характеристичний імпеданс)

# Гамма = α + jβ (константна для всіх частот)
gamma = complex(alpha, beta)

# ═════════════════════════════════════════════════════════════════════════════
# РОЗРАХУНОК
# ═════════════════════════════════════════════════════════════════════════════

# Діапазон частот (по логарифмічній шкалі від 0.01 до 1000 МГц)
freq_hz = np.logspace(np.log10(0.01e6), np.log10(1000e6), 500)
freq_mhz = freq_hz / 1e6

att_list = []
phi_list = []

for fi, f_hz in enumerate(freq_hz):
    # Гиперболические функции от постоянной гаммы
    gl1 = gamma * L1_seg
    gl2 = gamma * L2_seg
    gl_br = gamma * L_branch
    
    # Функции гиперболических косинуса и синуса (с обработкой переполнения)
    def safe_hyperbolic(gl):
        gl_re = np.real(gl)
        if gl_re > 100:
            exp_val = np.exp(gl_re)
            return exp_val / 2, exp_val / 2  # cosh, sinh
        else:
            return np.cosh(gl), np.sinh(gl)
    
    cosh_gl1, sinh_gl1 = safe_hyperbolic(gl1)
    cosh_gl2, sinh_gl2 = safe_hyperbolic(gl2)
    cosh_gl_br, sinh_gl_br = safe_hyperbolic(gl_br)
    
    # Вхідне сопротивление ветви (Load на конце ветвления)
    Zvid_branch = Z0 * (Z_load * cosh_gl_br + Z0 * sinh_gl_br) / (Z0 * cosh_gl_br + Z_load * sinh_gl_br)
    
    # ABCD матриці для сегментів та ветвлення
    # A1 (первый сегмент TX -> Node)
    A1_1 = cosh_gl1
    B1_1 = Z0 * sinh_gl1
    C1_1 = sinh_gl1 / Z0
    D1_1 = cosh_gl1
    
    # A2 (T-ветвление на Node)
    A2_1 = 1.0
    B2_1 = 0.0
    C2_1 = 1.0 / Zvid_branch
    D2_1 = 1.0
    
    # A1 (второй сегмент Node -> RX)
    A1_2 = cosh_gl2
    B1_2 = Z0 * sinh_gl2
    C1_2 = sinh_gl2 / Z0
    D1_2 = cosh_gl2
    
    # Множимо матриці: A_total = A1_1 * A2_1 * A1_2
    # Спочатку A1_1 * A2_1
    A_temp = A1_1 * A2_1 + B1_1 * C2_1
    B_temp = A1_1 * B2_1 + B1_1 * D2_1
    C_temp = C1_1 * A2_1 + D1_1 * C2_1
    D_temp = C1_1 * B2_1 + D1_1 * D2_1
    
    # Потім (A1_1 * A2_1) * A1_2
    A = A_temp * A1_2 + B_temp * C1_2
    B = A_temp * B1_2 + B_temp * D1_2
    C = C_temp * A1_2 + D_temp * C1_2
    D = C_temp * B1_2 + D_temp * D1_2
    
    # Передавальна функція: H = 2*ZL / (A*ZL + B + C*Zs*ZL + D*Zs)
    denom = A * ZL + B + C * Zs * ZL + D * Zs
    if np.abs(denom) > 1e-200 and np.isfinite(denom):
        H = (2.0 * ZL) / denom
        absH = np.abs(H)
        if absH > 0 and np.isfinite(absH):
            att = -20.0 * np.log10(absH)
            phi = np.angle(H)
        else:
            att = 0.0
            phi = 0.0
    else:
        att = 0.0
        phi = 0.0
    
    att_list.append(att)
    phi_list.append(phi)

att_array = np.array(att_list)
phi_array = np.array(phi_list)

# ═════════════════════════════════════════════════════════════════════════════
# ГРАФІКИ
# ═════════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.patch.set_facecolor('#0D1117')

# Загасання
ax1.set_facecolor('#161B22')
ax1.plot(freq_mhz, att_array, color='#58A6FF', linewidth=1.5, label=f'Загасання')
ax1.set_ylabel('Загасання [дБ]', color='#C9D1D9', fontsize=11)
ax1.set_title('Загасання МБЕ', color='#C9D1D9', fontsize=12, fontweight='bold')
ax1.grid(True, color='#30363D', linestyle='--', linewidth=0.5)
ax1.tick_params(colors='#C9D1D9')
ax1.set_xscale('log')
for spine in ax1.spines.values():
    spine.set_color('#30363D')
ax1.legend(loc='upper left', facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')

# Фаза
ax2.set_facecolor('#161B22')
ax2.plot(freq_mhz, phi_array, color='#58A6FF', linewidth=1.5, label='Фаза')
ax2.set_ylabel('Фаза [рад]', color='#C9D1D9', fontsize=11)
ax2.set_xlabel('Частота [МГц]', color='#C9D1D9', fontsize=11)
ax2.set_title('Фаза МБЕ', color='#C9D1D9', fontsize=12, fontweight='bold')
ax2.grid(True, color='#30363D', linestyle='--', linewidth=0.5)
ax2.tick_params(colors='#C9D1D9')
ax2.set_xscale('log')
for spine in ax2.spines.values():
    spine.set_color('#30363D')
ax2.legend(loc='upper left', facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')

plt.tight_layout()
plt.savefig('test_fixed_params.png', facecolor='#0D1117', dpi=150)
print("✓ Графік збережено у test_fixed_params.png")

print("\n=== РЕЗУЛЬТАТИ ===")
print(f"Діапазон частот: {freq_mhz[0]:.2f} - {freq_mhz[-1]:.0f} МГц")
print(f"Параметри кабелю:")
print(f"  α = {alpha} Нп/км")
print(f"  β = {beta} рад/км")
print(f"  Z0 = {Z0} Ом")
print(f"Мережа:")
print(f"  Zs = {Zs} Ом (TX)")
print(f"  ZL = {ZL} Ом (RX)")
print(f"  Z_load = {Z_load} Ом (Load)")
print(f"\nДіапазон загасання: {np.min(att_array):.2f} - {np.max(att_array):.2f} дБ")
print(f"Діапазон фази: {np.min(phi_array):.3f} - {np.max(phi_array):.3f} рад")

plt.show()
