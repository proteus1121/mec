#!/usr/bin/env python3
"""
Расчет МБЕ с частотно-зависимыми параметрами кабеля
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ═════════════════════════════════════════════════════════════════════════════
# ФУНКЦИИ ДЛЯ РАСЧЕТА ПАРАМЕТРОВ КАБЕЛЯ
# ═════════════════════════════════════════════════════════════════════════════

VX  = np.array([0,    0.5,     1,       1.5,    2,      2.5,    3,     3.5,   4,     4.5,   5,     7,     10   ])
VF  = np.array([0,    3.26e-4, 3.19e-3, 0.0258, 0.0782, 0.1756, 0.318, 0.492, 0.678, 0.862, 1.042, 1.743, 2.799])
VH  = np.array([0.0417,0.042,  0.053,   0.092,  0.169,  0.263,  0.348, 0.416, 0.466, 0.503, 0.530, 0.596, 0.643])
VQ  = np.array([1,    0.9998,  0.997,   0.987,  0.961,  0.913,  0.845, 0.766, 0.686, 0.616, 0.556, 0.400, 0.286])
VXG = np.array([0.5,  1,       1.5,     2,      2.5,    3,      3.5,   4,     4.5,   5,     7,     10   ])
VG  = np.array([9.75e-4,0.01519,0.0691, 0.1724, 0.295,  0.405,  0.499, 0.584, 0.669, 0.755, 1.109, 1.641])

_iF = interp1d(VX,  VF, kind='linear', bounds_error=False, fill_value=(0, None))
_iH = interp1d(VX,  VH, kind='linear', bounds_error=False, fill_value=(0.0417, 0.643))
_iQ = interp1d(VX,  VQ, kind='linear', bounds_error=False, fill_value=(1.0,    None))
_iG = interp1d(VXG, VG, kind='linear', bounds_error=False, fill_value=(0,      None))

def F_fn(x):
    x = np.asarray(x, dtype=float)
    r = np.where(x <= 0.4, 0.0,
        np.where(x <= 10,  _iF(np.clip(x, 0, 10)),
                            x / (2.0 * np.sqrt(2)) - 1.0))
    return r

def H_fn(x):
    x = np.asarray(x, dtype=float)
    r = np.where(x < 0.5,  0.0417,
        np.where(x <= 10,  _iH(np.clip(x, 0, 10)),
                            0.75))
    return r

def Q_fn(x):
    x = np.asarray(x, dtype=float)
    r = np.where(x < 0.7,  1.0,
        np.where(x <= 10,  _iQ(np.clip(x, 0, 10)),
                            2*np.sqrt(2)/np.where(x>0, x, 1e-10)))
    return r

def G_fn(xg):
    xg = np.asarray(xg, dtype=float)
    r = np.where(xg < 0.5,  xg**2 / 64,
        np.where(xg <= 10,  _iG(np.clip(xg, 0.5, 10)),
                             np.sqrt(2)*(xg - 1)*0.125))
    return r

def eps_f(f):
    return 2.45 + 2.73 * 2**(-1.54e-7 * f)

def krml_f(f, d1_mm=1.53):
    return 0.0105 * d1_mm * np.sqrt(f)

def calc_params(freq_mhz):
    """Расчет параметров кабеля на заданной частоте"""
    f_hz = freq_mhz * 1e6
    
    # Параметры ШВВП 2×1.5
    d1_mm = 1.53
    a1_mm = 2.53
    r1_mm = 0.765
    psi = 0.6
    rho = 0.0175  # медь
    mu = 1.0
    d_wire_mm = 0.238
    n_wires = 30.0
    
    # R01
    R01 = 2000 * 4 * rho / (d_wire_mm**2 * n_wires * np.pi)
    
    # Первичные параметры
    krml = krml_f(f_hz, d1_mm)
    d_rat = (d1_mm / a1_mm) ** 2
    
    F_v = F_fn(krml)
    H_v = H_fn(krml)
    Q_v = Q_fn(krml)
    G_v = G_fn(krml)
    
    denom_H = 1 - H_v * d_rat
    if np.abs(denom_H) < 1e-9:
        denom_H = 1e-9
    
    R1 = R01 * (1 + F_v + rho/0.0175 * G_v * d_rat / denom_H)
    L1 = (4 * np.log(2 * a1_mm / d1_mm) + mu * Q_v) * 1e-4
    C1 = eps_f(f_hz) * 1e-6 / (36 * np.log(a1_mm * psi / r1_mm))
    omega = 2 * np.pi * f_hz
    G1 = omega * C1 * 0.018
    
    # Вторичные параметры
    modZ = np.sqrt(R1**2 + (omega * L1)**2)
    modY = np.sqrt(G1**2 + (omega * C1)**2)
    cross = L1 * C1 * omega**2 - R1 * G1
    
    inner = modZ * modY
    alpha_np = np.sqrt(np.maximum(0, 0.5 * (inner - cross)))
    beta_rad = np.sqrt(np.maximum(0, 0.5 * (inner + cross)))
    
    Z_num = R1 + 1j * omega * L1
    Z_den = G1 + 1j * omega * C1
    if np.abs(Z_den) < 1e-30:
        Z_den = 1e-30 + 0j
    Z0_complex = np.sqrt(Z_num / Z_den)
    Z0 = np.abs(Z0_complex)
    
    return alpha_np, beta_rad, Z0

# ═════════════════════════════════════════════════════════════════════════════
# ТАБЛИЦА ПАРАМЕТРОВ
# ═════════════════════════════════════════════════════════════════════════════

freq_table = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
alpha_table = []
beta_table = []
Z0_table = []

for f in freq_table:
    a, b, z = calc_params(f)
    alpha_table.append(a)
    beta_table.append(b)
    Z0_table.append(z)

alpha_table = np.array(alpha_table)
beta_table = np.array(beta_table)
Z0_table = np.array(Z0_table)

# Интерполяция параметров (логарифмическая шкала)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

f_log = np.log10(freq_table)
alpha_log = np.log10(np.maximum(alpha_table, 1e-10))
beta_log = np.log10(np.maximum(beta_table, 1e-10))

interp_alpha = interp1d(f_log, alpha_log, kind='linear', bounds_error=False, fill_value='extrapolate')
interp_beta = interp1d(f_log, beta_log, kind='linear', bounds_error=False, fill_value='extrapolate')
interp_Z0 = interp1d(f_log, Z0_table, kind='linear', bounds_error=False, fill_value='extrapolate')

# ═════════════════════════════════════════════════════════════════════════════
# ПАРАМЕТРИ МЕРЕЖІ
# ═════════════════════════════════════════════════════════════════════════════

L1_seg = 5.0     # м (TX -> Node)
L2_seg = 5.0     # м (Node -> RX)  
L_branch = 5.0   # м (Node -> Load)
Zs = 75.0        # Ом (TX)
ZL = 75.0        # Ом (RX)
Z_load = 59.0    # Ом (Load)

# ═════════════════════════════════════════════════════════════════════════════
# РОЗРАХУНОК НА ВСІХ ЧАСТОТАХ
# ═════════════════════════════════════════════════════════════════════════════

freq_hz = np.logspace(np.log10(0.01e6), np.log10(1000e6), 500)
freq_mhz = freq_hz / 1e6
f_log_arr = np.log10(freq_mhz)

att_list = []
phi_list = []

for fi, f_mhz in enumerate(freq_mhz):
    # Интерполируем параметры для этой частоты
    f_lg = np.log10(f_mhz)
    
    alpha = 10.0 ** interp_alpha(f_lg)
    beta = 10.0 ** interp_beta(f_lg)
    Z0 = interp_Z0(f_lg)
    
    # Гамма = α + jβ
    gamma = complex(alpha, beta)
    
    # Гиперболические функции
    gl1 = gamma * L1_seg
    gl2 = gamma * L2_seg
    gl_br = gamma * L_branch
    
    def safe_hyperbolic(gl):
        gl_re = np.real(gl)
        if gl_re > 100:
            exp_val = np.exp(gl_re)
            return exp_val / 2, exp_val / 2
        else:
            return np.cosh(gl), np.sinh(gl)
    
    cosh_gl1, sinh_gl1 = safe_hyperbolic(gl1)
    cosh_gl2, sinh_gl2 = safe_hyperbolic(gl2)
    cosh_gl_br, sinh_gl_br = safe_hyperbolic(gl_br)
    
    # Вхідне сопротивление ветви
    Zvid_branch = Z0 * (Z_load * cosh_gl_br + Z0 * sinh_gl_br) / (Z0 * cosh_gl_br + Z_load * sinh_gl_br)
    
    # ABCD матриці
    A1_1 = cosh_gl1
    B1_1 = Z0 * sinh_gl1
    C1_1 = sinh_gl1 / Z0
    D1_1 = cosh_gl1
    
    A2_1 = 1.0
    B2_1 = 0.0
    C2_1 = 1.0 / Zvid_branch
    D2_1 = 1.0
    
    A1_2 = cosh_gl2
    B1_2 = Z0 * sinh_gl2
    C1_2 = sinh_gl2 / Z0
    D1_2 = cosh_gl2
    
    # Множимо: (A1_1 * A2_1) * A1_2
    A_temp = A1_1 * A2_1 + B1_1 * C2_1
    B_temp = A1_1 * B2_1 + B1_1 * D2_1
    C_temp = C1_1 * A2_1 + D1_1 * C2_1
    D_temp = C1_1 * B2_1 + D1_1 * D2_1
    
    A = A_temp * A1_2 + B_temp * C1_2
    B = A_temp * B1_2 + B_temp * D1_2
    C = C_temp * A1_2 + D_temp * C1_2
    D = C_temp * B1_2 + D_temp * D1_2
    
    # Передавальна функція
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
ax1.plot(freq_mhz, att_array, color='#58A6FF', linewidth=1.5)
ax1.set_ylabel('Загасання [дБ]', color='#C9D1D9', fontsize=11)
ax1.set_title('Загасання МБЕ', color='#C9D1D9', fontsize=12, fontweight='bold')
ax1.grid(True, color='#30363D', linestyle='--', linewidth=0.5)
ax1.tick_params(colors='#C9D1D9')
ax1.set_xscale('log')
for spine in ax1.spines.values():
    spine.set_color('#30363D')

# Фаза
ax2.set_facecolor('#161B22')
ax2.plot(freq_mhz, phi_array, color='#58A6FF', linewidth=1.5)
ax2.set_ylabel('Фаза [рад]', color='#C9D1D9', fontsize=11)
ax2.set_xlabel('Частота [МГц]', color='#C9D1D9', fontsize=11)
ax2.set_title('Фаза МБЕ', color='#C9D1D9', fontsize=12, fontweight='bold')
ax2.grid(True, color='#30363D', linestyle='--', linewidth=0.5)
ax2.tick_params(colors='#C9D1D9')
ax2.set_xscale('log')
for spine in ax2.spines.values():
    spine.set_color('#30363D')

plt.tight_layout()
plt.savefig('test_frequency_dependent.png', facecolor='#0D1117', dpi=150)
print("✓ Графік збережено у test_frequency_dependent.png")

print("\n=== РЕЗУЛЬТАТИ ===")
print(f"Діапазон частот: {freq_mhz[0]:.2f} - {freq_mhz[-1]:.0f} МГц")
print(f"Діапазон загасання: {np.min(att_array):.2f} - {np.max(att_array):.2f} дБ")
print(f"Діапазон фази: {np.min(phi_array):.3f} - {np.max(phi_array):.3f} рад")

plt.show()
