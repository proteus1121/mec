#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unified_app import CableCalculator

cc = CableCalculator()

print("Testing spectrum calculation...")
result = cc.calculate_params_spectrum('ШВВП 2×1.5', 'copper', freq_min_mhz=1, freq_max_mhz=100, n_points=10)
print(f"✓ Spectrum calculated: {len(result['freq_mhz'])} points")
print(f"  Frequencies (first 3): {[f'{f:.2f}' for f in result['freq_mhz'][:3]]}")
print(f"  Frequencies (last 3): {[f'{f:.2f}' for f in result['freq_mhz'][-3:]]}")

print("\nTesting interpolation...")
for test_freq in [1, 10, 24, 50, 100, 150]:
    params = cc.get_params_at_freq('ШВВП 2×1.5', 'copper', test_freq)
    print(f"  f={test_freq:3} MHz: α={params['alpha']:7.2f} β={params['beta']:8.2f}")

print("\n✓ All tests passed!")
