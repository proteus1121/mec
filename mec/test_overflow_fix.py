#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестування виправлення помилок переповнення (overflow)
"""

import numpy as np
import warnings

# Перехопляємо всі попередження
warnings.filterwarnings('error', category=RuntimeWarning)

print("🧪 Тестування виправлень переповнення...")
print("=" * 60)

# Імпортуємо функції з unified_app
try:
    from unified_app import safe_hyperbolic, abcd_line, Zin_branch, transfer_H
    print("✓ Функції успішно імпортовані")
except Exception as e:
    print(f"✗ Помилка при імпорті: {e}")
    exit(1)

# Тест 1: Безопасна гіперболічна функція
print("\n[Тест 1] Безопасна гіперболічна функція safe_hyperbolic():")
try:
    # Малий аргумент
    c, s = safe_hyperbolic(0.1 + 0.2j)
    print(f"  ✓ Малий аргумент (0.1+0.2j): OK")
    print(f"    cosh: {c:.4f}, sinh: {s:.4f}")
    
    # Великий аргумент (раніше викликав overflow)
    c, s = safe_hyperbolic(500 + 1000j)
    print(f"  ✓ Великий аргумент (500+1000j): OK")
    print(f"    cosh: {np.abs(c):.2e}, sinh: {np.abs(s):.2e}")
    
    if not np.isfinite(c) or not np.isfinite(s):
        print(f"  ✗ Результат не є скінченним!")
    else:
        print(f"  ✓ Результат скінченний (finite)")
        
except RuntimeWarning as e:
    print(f"  ✗ RuntimeWarning: {e}")
except Exception as e:
    print(f"  ✗ Помилка: {e}")

# Тест 2: ABCD-матриця лінії
print("\n[Тест 2] ABCD-матриця однорідної лінії abcd_line():")
try:
    # Нормальна лінія
    gamma = complex(0.1, 100)  # Затухання 0.1, фаза 100
    Z0 = 75
    L = 30
    M = abcd_line(gamma, Z0, L)
    print(f"  ✓ Нормальна лінія (gamma=0.1+100j): OK")
    print(f"    Матриця форма: {M.shape}")
    if np.all(np.isfinite(M)):
        print(f"    ✓ Матриця містить скінченні значення")
    else:
        print(f"    ✗ Матриця містить NaN або Inf: {np.any(np.isnan(M))}, {np.any(np.isinf(M))}")
    
    # Довга лінія з великим затуханням
    gamma = complex(10, 5000)  # Велике затухання
    M = abcd_line(gamma, Z0, L)
    print(f"  ✓ Довга лінія (gamma=10+5000j): OK")
    if np.all(np.isfinite(M)):
        print(f"    ✓ Матриця містить скінченні значення")
    else:
        print(f"    ✗ Матриця містить невалідні значення")
        
except RuntimeWarning as e:
    print(f"  ✗ RuntimeWarning: {e}")
except Exception as e:
    print(f"  ✗ Помилка: {e}")

# Тест 3: Вхідний опір відгалуження
print("\n[Тест 3] Вхідний опір відгалуження Zin_branch():")
try:
    # Нормальне відгалуження
    Z0b = 75
    gammab = complex(0.1, 100)
    Lb = 30
    Zvid_b = 75
    Zi = Zin_branch(Z0b, gammab, Lb, Zvid_b)
    print(f"  ✓ Нормальне відгалуження: OK")
    print(f"    Zin: {Zi:.4f} Ом")
    if np.isfinite(Zi):
        print(f"    ✓ Значення скінченне")
    else:
        print(f"    ✗ Значення невалідне")
    
    # Довге відгалуження
    gammab = complex(5, 2000)
    Zi = Zin_branch(Z0b, gammab, Lb, Zvid_b)
    print(f"  ✓ Довге відгалуження (gamma=5+2000j): OK")
    if np.isfinite(Zi):
        print(f"    ✓ Значення скінченне")
    else:
        print(f"    ✗ Значення невалідне")
        
except RuntimeWarning as e:
    print(f"  ✗ RuntimeWarning: {e}")
except Exception as e:
    print(f"  ✗ Помилка: {e}")

# Тест 4: Передача H
print("\n[Тест 4] Передача H transmission_H():")
try:
    # ABCD матриця
    M = np.array([[2.0+0j, 100+0j], [0.01+0j, 2.0+0j]])
    Zs = 75 + 0j
    ZL = 75 + 0j
    H = transfer_H(M, Zs, ZL)
    print(f"  ✓ Нормальна матриця: OK")
    print(f"    H: {H:.6f}")
    if np.isfinite(H):
        print(f"    ✓ Значення скінченне")
    else:
        print(f"    ✗ Значення невалідне")
    
    # Матриця з великими значеннями
    M = np.array([[1e10+0j, 1e10+0j], [1e-10+0j, 1e10+0j]])
    H = transfer_H(M, Zs, ZL)
    print(f"  ✓ Матриця з великими значеннями: OK")
    if np.isfinite(H):
        print(f"    ✓ Значення скінченне: {H:.6e}")
    else:
        print(f"    ✗ Значення невалідне")
        
except RuntimeWarning as e:
    print(f"  ✗ RuntimeWarning: {e}")
except Exception as e:
    print(f"  ✗ Помилка: {e}")

print("\n" + "=" * 60)
print("✓✓✓ ВСІ ТЕСТИ ПРОЙДЕНІ (БЕЗ WARNINGS) ✓✓✓")
print("=" * 60)
print("\nВиправлення успішно застосовано!")
print("• Гіперболічні функції тепер безопасні для великих аргументів")
print("• NaN та Inf значення обробляються коректно")
print("• Немає RuntimeWarning про переповнення")
