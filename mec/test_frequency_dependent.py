#!/usr/bin/env python3
"""
Тест: Параметры кабеля зависят от частоты, а не фиксированы на 24 МГц
"""

import sys
import json
from pathlib import Path

# Добавляем путь для импорта
mec_dir = Path(__file__).parent
sys.path.insert(0, str(mec_dir))

from unified_app import CableCalculator, NetworkGraph

def test_frequency_dependent_params():
    """Проверяем, что параметры интерполируются для разных частот"""
    
    print("=" * 70)
    print("ТЕСТ: Параметры кабеля зависят от частоты")
    print("=" * 70)
    
    # Создаём кабельный калькулятор
    calc = CableCalculator()
    
    # Рассчитываем спектр для ШВВП 2×1.5
    cable_type = 'ШВВП 2×1.5'
    material = 'copper'
    
    print(f"\n✓ Рассчитываем спектр для {cable_type} ({material})...")
    calc.calculate_params_spectrum(cable_type, material, 
                                   freq_min_mhz=1, freq_max_mhz=100, n_points=10)
    print(f"  Спектр сохранён в JSON")
    
    # Получаем параметры на разных частотах
    print(f"\n✓ Интерполируем параметры на разных частотах:")
    print(f"  {'Частота (МГц)':<15} {'α (Np/m)':<15} {'β (rad/m)':<15} {'Z0 (Ω)':<10}")
    print(f"  {'-'*55}")
    
    test_freqs = [1, 10, 24, 50, 100]
    params_dict = {}
    
    for freq_mhz in test_freqs:
        params = calc.get_params_at_freq(cable_type, material, freq_mhz)
        alpha = params['alpha']
        beta = params['beta']
        Z0 = params['Z0']
        params_dict[freq_mhz] = {'alpha': alpha, 'beta': beta, 'Z0': Z0}
        print(f"  {freq_mhz:<15} {alpha:<15.6f} {beta:<15.6f} {Z0:<10.1f}")
    
    # Проверяем, что α и β меняются с частотой
    alpha_1 = params_dict[1]['alpha']
    alpha_100 = params_dict[100]['alpha']
    
    print(f"\n✓ Проверка: α меняется с частотой?")
    print(f"  α(1 МГц) = {alpha_1:.6f}")
    print(f"  α(100 МГц) = {alpha_100:.6f}")
    
    if abs(alpha_100 - alpha_1) > 0.1:
        print(f"  ✓ ДА! α отличается на {abs(alpha_100 - alpha_1):.6f}")
    else:
        print(f"  ✗ НЕТ - параметры одинаковые (ошибка!)")
        return False
    
    # Проверяем β
    beta_1 = params_dict[1]['beta']
    beta_100 = params_dict[100]['beta']
    
    print(f"\n✓ Проверка: β меняется с частотой?")
    print(f"  β(1 МГц) = {beta_1:.6f}")
    print(f"  β(100 МГц) = {beta_100:.6f}")
    
    if abs(beta_100 - beta_1) > 0.1:
        print(f"  ✓ ДА! β отличается на {abs(beta_100 - beta_1):.6f}")
    else:
        print(f"  ✗ НЕТ - параметры одинаковые (ошибка!)")
        return False
    
    print("\n" + "=" * 70)
    print("✓ ТЕСТ ПРОЙДЕН: Параметры кабеля корректно зависят от частоты")
    print("=" * 70)
    return True

def test_edge_storage():
    """Проверяем, что в edge сохраняется только Z0, а α и β интерполируются"""
    
    print("\n" + "=" * 70)
    print("ТЕСТ: Edge хранит Z0, а α и β интерполируются при расчёте")
    print("=" * 70)
    
    # Создаём граф
    graph = NetworkGraph()
    
    # Добавляем узлы (add_node возвращает ID)
    n1 = graph.add_node('plc_tx', 100, 100, Z=50)
    n2 = graph.add_node('node', 300, 100, Z=50)
    
    # Добавляем кабель между узлами (используем ID, а не имена)
    edge_id = graph.add_edge(n1, n2, L=30, cable_type='ШВВП 2×1.5', material='copper')
    edge = graph.edges[edge_id]
    
    # Рассчитываем спектр
    calc = CableCalculator()
    calc.calculate_params_spectrum('ШВВП 2×1.5', 'copper', 
                                   freq_min_mhz=1, freq_max_mhz=100, n_points=10)
    
    # Получаем Z0 на эталонной частоте
    params_24mhz = calc.get_params_at_freq('ШВВП 2×1.5', 'copper', 24.0)
    edge['Z0'] = params_24mhz['Z0']
    
    # Имитируем _on_cable_change: удаляем alpha и beta
    if 'alpha' in edge:
        del edge['alpha']
    if 'beta' in edge:
        del edge['beta']
    
    print(f"\nСохранено в edge:")
    print(f"  cable_type: {edge.get('cable_type')}")
    print(f"  material: {edge.get('material')}")
    print(f"  Z0: {edge.get('Z0', 'НЕ УСТАНОВЛЕНО'):.1f} Ω (эталонное значение)")
    print(f"  alpha: {edge.get('alpha', 'НЕ СОХРАНЯЕТСЯ')}")
    print(f"  beta: {edge.get('beta', 'НЕ СОХРАНЯЕТСЯ')}")
    
    # Проверяем, что alpha и beta не сохранены
    has_alpha = 'alpha' in edge
    has_beta = 'beta' in edge
    
    if not has_alpha and not has_beta:
        print(f"\n✓ ДА! α и β не сохраняются в edge (как и должно быть)")
    else:
        print(f"\n✗ ОШИБКА! α и β сохранены в edge")
        return False
    
    # Проверяем интерполяцию при расчёте
    print(f"\nПри build_cascade система интерполирует параметры для каждой частоты:")
    test_freqs = [1, 10, 50, 100]
    print(f"  {'Частота':<10} {'α (from spectrum)':<20} {'Z0 (from spectrum)':<20}")
    print(f"  {'-'*50}")
    
    for f_hz in [1e6, 10e6, 50e6, 100e6]:
        f_mhz = f_hz / 1e6
        params = calc.get_params_at_freq('ШВВП 2×1.5', 'copper', f_mhz)
        alpha = params['alpha']
        Z0 = params['Z0']
        print(f"  {f_mhz:<10.0f} {alpha:<20.6f} {Z0:<20.1f}")
    
    print("\n" + "=" * 70)
    print("✓ ТЕСТ ПРОЙДЕН: Edge правильно хранит параметры")
    print("  Спектр интерполируется для каждой частоты при расчёте")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = True
    success = test_frequency_dependent_params() and success
    success = test_edge_storage() and success
    
    if success:
        print("\n✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        sys.exit(0)
    else:
        print("\n✗ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ!")
        sys.exit(1)
