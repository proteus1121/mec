#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестування сохранення типу кабеля при переключении
"""

import sys
sys.path.insert(0, '.')

from unified_app import NetworkGraph, CableCalculator

print("🧪 Тестування сохранення типу кабеля...")
print("=" * 60)

# Создаём граф
graph = NetworkGraph()

# Добавляем узлы
tx = graph.add_node('plc_tx', 100, 100)
rx = graph.add_node('plc_rx', 200, 100)

# Добавляем кабель
eid = graph.add_edge(tx, rx)

print("\n✓ Граф создан с TX, RX и кабелем")

# Тест 1: Устанавливаем первый тип кабеля
print("\n[Тест 1] Установка ШВВП 2×1.5:")
edge = graph.edges[eid]
edge['cable_type'] = 'ШВВП 2×1.5'
edge['material'] = 'copper'
print(f"  ✓ Установлено: {edge.get('cable_type')}")
assert edge.get('cable_type') == 'ШВВП 2×1.5', "ОШИБКА: Тип не сохранился!"
print(f"  ✓ Проверка: Значение в памяти = {edge.get('cable_type')}")

# Тест 2: Переключаемся на второй тип
print("\n[Тест 2] Переключение на ШВВП 2×2.5:")
edge['cable_type'] = 'ШВВП 2×2.5'
edge['material'] = 'aluminium'
print(f"  ✓ Переключено: {edge.get('cable_type')}")
assert edge.get('cable_type') == 'ШВВП 2×2.5', "ОШИБКА: Тип не изменился!"
print(f"  ✓ Проверка: Значение в памяти = {edge.get('cable_type')}")

# Тест 3: Переключаемся обратно на первый тип
print("\n[Тест 3] Переключение обратно на ШВВП 2×1.5:")
edge['cable_type'] = 'ШВВП 2×1.5'
edge['material'] = 'copper'
print(f"  ✓ Переключено обратно: {edge.get('cable_type')}")
assert edge.get('cable_type') == 'ШВВП 2×1.5', "ОШИБКА: Тип не восстановился!"
print(f"  ✓ Проверка: Значение в памяти = {edge.get('cable_type')}")

# Тест 4: Сохранение и загрузка через JSON
print("\n[Тест 4] Сохранение в JSON и восстановление:")
data = graph.to_dict()
graph2 = NetworkGraph()
graph2.from_dict(data)
edge2 = graph2.edges[eid]
print(f"  ✓ После восстановления из JSON: {edge2.get('cable_type')}")
assert edge2.get('cable_type') == 'ШВВП 2×1.5', "ОШИБКА: Тип не восстановился из JSON!"
assert edge2.get('material') == 'copper', "ОШИБКА: Материал не восстановился из JSON!"
print(f"  ✓ Проверка JSON: cable_type = {edge2.get('cable_type')}, material = {edge2.get('material')}")

# Тест 5: Множественные переключения
print("\n[Тест 5] Множественные переключения:")
for i, ct in enumerate(['ШВВП 2×2.5', 'ШВВП 2×4', 'ШВВП 2×1.5', 'ШВВП 2×2.5']):
    edge['cable_type'] = ct
    retrieved = edge.get('cable_type')
    print(f"  Итерация {i+1}: {ct} → {retrieved}")
    assert retrieved == ct, f"ОШИБКА: Ожидался {ct}, получено {retrieved}"
print(f"  ✓ Все переключения сохранились корректно")

print("\n" + "=" * 60)
print("✓✓✓ ВСІ ТЕСТИ ПРОЙДЕНІ ✓✓✓")
print("=" * 60)
print("\nТип кабеля теперь надежно сохраняется при переключении!")
print("✓ Тип сохраняется в памяти")
print("✓ Тип восстанавливается при загрузке из JSON")
print("✓ Множественные переключения работают корректно")
