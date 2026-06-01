"""Test script for unified_app functionality"""

from unified_app import CableCalculator
import json

def test_cable_calculator():
    cc = CableCalculator()
    
    # Test calculation
    params = cc.calculate_params_at_freq('ШВВП 2×1.5', 'copper', 24.0)
    if params:
        print('✓ Calculation successful!')
        print(f"Cable: {params['cable_type']}")
        print(f"Material: {params['material']}")
        print(f"Frequency: {params['frequency_mhz']} MHz")
        print(f"Alpha: {params['alpha']:.6f} Нп/м")
        print(f"Beta: {params['beta']:.6f} рад/м")
        print(f"Z0: {params['Z0']:.1f} Ом")
    else:
        print('✗ Calculation failed')
    
    # Test database
    print(f"✓ Database path: {cc.db_path}")
    print(f"✓ Available cables: {cc.get_cable_list()}")
    print(f"✓ Available materials: {cc.get_materials()}")

if __name__ == '__main__':
    test_cable_calculator()
