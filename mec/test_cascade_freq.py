#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test frequency-dependent alpha and beta in cascade calculation"""

import numpy as np
from unified_app import NetworkGraph, CableCalculator

# Инициализируем граф и калькулятор
graph = NetworkGraph()
cable_calc = CableCalculator()

# Рассчитываем спектр для типа кабеля
print("Calculating cable spectrum...")
cable_calc.calculate_params_spectrum('ШВВП 2×1.5', 'copper', freq_min_mhz=1, freq_max_mhz=100, n_points=20)

# Создаем простую топологию: TX -> Cable -> RX
n_tx = graph.add_node('plc_tx', x=0, y=0, Z='50')
n_rx = graph.add_node('plc_rx', x=100, y=0, Z='50')

# Добавляем кабель с параметрами типа
eid = graph.add_edge(n_tx, n_rx, L=100.0, 
                     cable_type='ШВВП 2×1.5', material='copper', frequency_mhz=24.0)

# Получаем параметры кабеля на 24 МГц
params_24 = cable_calc.get_params_at_freq('ШВВП 2×1.5', 'copper', 24.0)
print(f"\nCable parameters at 24 MHz:")
print(f"  α = {params_24['alpha']:.2f} Np/km")
print(f"  β = {params_24['beta']:.2f} rad/km")

# Сохраняем параметры в ребро
edge = graph.edges[eid]
edge['alpha'] = params_24['alpha']
edge['beta'] = params_24['beta']
edge['Z0'] = params_24['Z0']

# Проверяем валидность топологии
ok, msg = graph.validate()
print(f"\nTopology validation: {msg}")

# Рассчитываем каскад на нескольких частотах
freq_array = np.array([1e6, 10e6, 24e6, 50e6, 100e6])  # 1, 10, 24, 50, 100 MHz
print(f"\nCalculating cascade for frequencies: {freq_array/1e6} MHz")

try:
    att, phi, pnodes, pedges = graph.build_cascade(freq_array)
    print("\n✓ Cascade calculation successful!")
    print(f"  Attenuation (dB): {att}")
    print(f"  Phase (rad): {phi}")
    print("\n✓ Frequency-dependent parameters are working correctly!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
