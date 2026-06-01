"""Test cable type saving and persistence"""

import json
from unified_app import NetworkGraph, CableCalculator

# Тест: создаём граф с кабелями и сохраняем
g = NetworkGraph()
cc = CableCalculator()

# Добавляем узлы
tx = g.add_node('plc_tx', 100, 100, Z=75, name='TX')
rx = g.add_node('plc_rx', 300, 100, Z=75, name='RX')

# Добавляем кабель с типом
params = cc.calculate_params_at_freq('ШВВП 2×1.5', 'copper', 24.0)
eid = g.add_edge(tx, rx, L=30, alpha=params['alpha'], beta=params['beta'], Z0=params['Z0'],
                 cable_type='ШВВП 2×1.5', material='copper', frequency_mhz=24.0)

# Проверяем, что сохраняется
edge_data = g.edges[eid]
print('✓ Кабель добавлен:')
print(f"  cable_type: {edge_data.get('cable_type')}")
print(f"  material: {edge_data.get('material')}")
print(f"  alpha: {edge_data.get('alpha'):.6f}")
print(f"  Z0: {edge_data.get('Z0'):.2f}")

# Сериализуем и десериализуем
scheme_dict = g.to_dict()
print('\n✓ Схема сериализована')

g2 = NetworkGraph()
g2.from_dict(scheme_dict)
print('✓ Схема десериализована')

# Проверяем восстановленные данные
for eid, edge in g2.edges.items():
    print(f"\n✓ Восстановленный кабель:")
    print(f"  cable_type: {edge.get('cable_type')}")
    print(f"  material: {edge.get('material')}")
    print(f"  frequency_mhz: {edge.get('frequency_mhz')}")
    print(f"  alpha: {edge.get('alpha'):.6f}")
    print(f"  Z0: {edge.get('Z0'):.2f}")

# Тест сохранения в файл
print('\n✓ Тестирование сохранения в файл:')
with open('test_scheme.json', 'w', encoding='utf-8') as f:
    json.dump(scheme_dict, f, indent=2, ensure_ascii=False)
print('  Сохранено в test_scheme.json')

with open('test_scheme.json', 'r', encoding='utf-8') as f:
    loaded_scheme = json.load(f)
print('  Загружено из файла')

# Восстанавливаем из файла
g3 = NetworkGraph()
g3.from_dict(loaded_scheme)
for eid, edge in g3.edges.items():
    print(f"\n✓ Кабель из файла:")
    print(f"  cable_type: {edge.get('cable_type')}")
    print(f"  material: {edge.get('material')}")
    assert edge.get('cable_type') == 'ШВВП 2×1.5', "cable_type не сохранился!"
    assert edge.get('material') == 'copper', "material не сохранился!"

print('\n✓✓✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ ✓✓✓')
print('Тип кабеля и материал успешно сохраняются и восстанавливаются!')
