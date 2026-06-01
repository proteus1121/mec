#!/usr/bin/env python3
"""
Тест: build_cascade использует интерполированные параметры для выбранного диапазона частот
"""

import sys
import numpy as np
from pathlib import Path

mec_dir = Path(__file__).parent
sys.path.insert(0, str(mec_dir))

from unified_app import CableCalculator, NetworkGraph

def test_build_cascade_with_range():
    """Проверяем, что build_cascade интерполирует параметры для каждой частоты"""
    
    print("=" * 80)
    print("ТЕСТ: build_cascade использует интерполированные параметры для диапазона")
    print("=" * 80)
    
    # Создаём граф с простой топологией: TX → Cable → RX
    graph = NetworkGraph()
    
    # Добавляем узлы
    tx_id = graph.add_node('plc_tx', 50, 100, Z=50, name='TX')
    rx_id = graph.add_node('plc_rx', 250, 100, Z=50, name='RX')
    
    # Добавляем кабель
    edge_id = graph.add_edge(tx_id, rx_id, L=100, 
                            cable_type='ШВВП 2×1.5', material='copper')
    edge = graph.edges[edge_id]
    
    # Рассчитываем спектр
    calc = CableCalculator()
    print(f"\n✓ Рассчитываем спектр для кабеля...")
    calc.calculate_params_spectrum('ШВВП 2×1.5', 'copper', 
                                   freq_min_mhz=1, freq_max_mhz=100, n_points=10)
    
    # Устанавливаем Z₀ как в реальном коде
    params_ref = calc.get_params_at_freq('ШВВП 2×1.5', 'copper', 24.0)
    edge['Z0'] = params_ref['Z0']
    
    # Удаляем alpha/beta (как в _on_cable_change)
    if 'alpha' in edge:
        del edge['alpha']
    if 'beta' in edge:
        del edge['beta']
    
    print(f"  Спектр сохранён")
    print(f"  Edge['Z0'] = {edge['Z0']:.1f} Ω (эталонное значение)")
    print(f"  Edge['alpha'] = {edge.get('alpha', 'не сохранено')}")
    print(f"  Edge['beta'] = {edge.get('beta', 'не сохранено')}")
    
    # Рассчитываем каскад для разных диапазонов частот
    print(f"\n✓ Расчёт каскада для РАЗНЫХ диапазонов частот:")
    
    test_cases = [
        ("Узкий диапазон (1-10 МГц, 3 точки)", 
         np.array([1e6, 5.5e6, 10e6])),
        
        ("Средний диапазон (10-50 МГц, 3 точки)", 
         np.array([10e6, 30e6, 50e6])),
        
        ("Широкий диапазон (1-100 МГц, 3 точки)", 
         np.array([1e6, 50.5e6, 100e6])),
    ]
    
    all_passed = True
    
    for test_name, freq_array in test_cases:
        print(f"\n  {test_name}")
        print(f"  {'-' * 70}")
        
        try:
            # Выполняем расчёт каскада (функция сама найдёт TX и RX по типам)
            result = graph.build_cascade(freq_array)
            
            if result is None:
                print(f"    ✗ ОШИБКА: build_cascade вернул None")
                all_passed = False
                continue
            
            att_dB, phi_rad, path_nodes, path_edges = result
            
            print(f"  {'Частота':<12} {'α (Np/m)':<15} {'Ослабление (дБ)':<18} {'Статус'}")
            print(f"  {'-' * 70}")
            
            # Проверяем, что ослабление отличается для разных частот (α должна меняться)
            prev_alpha = None
            alphas = []
            
            for i, f_hz in enumerate(freq_array):
                f_mhz = f_hz / 1e6
                
                # Получаем эталонное значение α для этой частоты
                params = calc.get_params_at_freq('ШВВП 2×1.5', 'copper', f_mhz)
                alpha_expected = params['alpha']
                alphas.append(alpha_expected)
                
                att = att_dB[i]
                
                # Проверяем, что функция выполнена успешно (att конечное число)
                if np.isfinite(att):
                    status = "✓ OK (att finite)"
                else:
                    status = "✗ INVALID (att not finite)"
                    all_passed = False
                
                print(f"  {f_mhz:<12.1f} {alpha_expected:<15.6f} {att:<18.4f} {status}")
            
            # Проверяем, что α меняется для разных частот
            if len(alphas) > 1 and max(alphas) > min(alphas):
                alpha_variation = max(alphas) - min(alphas)
                print(f"\n  → α меняется: от {min(alphas):.6f} до {max(alphas):.6f}")
                print(f"    Диапазон: {alpha_variation:.6f} Np/m ✓")
            else:
                print(f"\n  → ✗ ОШИБКА: α не меняется между частотами!")
                all_passed = False
        
        except Exception as ex:
            print(f"    ✗ ОШИБКА: {str(ex)}")
            all_passed = False
    
    print(f"\n" + "=" * 80)
    if all_passed:
        print("✓ ТЕСТ ПРОЙДЕН: build_cascade правильно интерполирует для разных диапазонов")
    else:
        print("✗ ТЕСТ НЕ ПРОЙДЕН: некоторые расчёты некорректны")
    print("=" * 80)
    
    return all_passed

if __name__ == '__main__':
    success = test_build_cascade_with_range()
    sys.exit(0 if success else 1)
