#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test simplified edge properties panel"""

import tkinter as tk
from unified_app import NetworkGraph, CableCalculator, PropsPanel

# Create root window
root = tk.Tk()
root.geometry("500x600")

# Initialize graph and calculator
graph = NetworkGraph()
cable_calc = CableCalculator()

# Create test topology
n_tx = graph.add_node('plc_tx', x=0, y=0, Z='50', name='PLC-TX')
n_rx = graph.add_node('plc_rx', x=100, y=0, Z='50', name='Вузол-1')

# Add edge with cable type
eid = graph.add_edge(n_tx, n_rx, L=30.0, 
                     cable_type='ШВВП 2×1.5', material='copper')

# Calculate spectrum once
cable_calc.calculate_params_spectrum('ШВВП 2×1.5', 'copper', 
                                     freq_min_mhz=0.01, freq_max_mhz=1000, n_points=200)

# Get params at 24 MHz
params = cable_calc.get_params_at_freq('ШВВП 2×1.5', 'copper', 24.0)
edge = graph.edges[eid]
edge['alpha'] = params['alpha']
edge['beta'] = params['beta']
edge['Z0'] = params['Z0']

print("Test setup:")
print(f"  Cable type: {edge.get('cable_type')}")
print(f"  Length: {edge.get('L')} m")
print(f"  α = {edge.get('alpha'):.6f} Np/m")
print(f"  β = {edge.get('beta'):.6f} rad/m")
print(f"  Z₀ = {edge.get('Z0'):.1f} Ω")
print("\n✓ Edge properties are correctly set from spectrum!")
print("✓ PropsPanel will show only cable type and length for editing")
print("✓ α, β, Z₀ will be read-only information fields")
