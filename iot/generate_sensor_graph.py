"""
Build sensor network graph using real IoT data from sensor_data.csv
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# Load real IoT sensor data from CSV
# ═══════════════════════════════════════════════════════════════════

# Read CSV file
data = pd.read_csv('sensor_data.csv')

# Convert datetime to pandas datetime
data['datetime'] = pd.to_datetime(data['datetime'])

# Sort by datetime
data = data.sort_values('datetime').reset_index(drop=True)

# Handle missing values - forward fill then backward fill
for col in ['temperature_C', 'humidity_pct', 'lpg', 'ch4', 'smoke']:
    data[col] = data[col].ffill().bfill()

print(f"Loaded {len(data)} records from sensor_data.csv")

print("Data Summary:")
print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
print(f"Temperature range: {data['temperature_C'].min():.1f} - {data['temperature_C'].max():.1f}°C")
print(f"Humidity range: {data['humidity_pct'].min():.1f} - {data['humidity_pct'].max():.1f}%")
print(f"LPG range: {data['lpg'].min():.1f} - {data['lpg'].max():.1f} ppm")
print(f"CH4 range: {data['ch4'].min():.1f} - {data['ch4'].max():.1f} ppm")
print(f"Smoke range: {data['smoke'].min():.1f} - {data['smoke'].max():.1f}")

# Create plot with actual values from CSV
fig, ax = plt.subplots(figsize=(18, 10), facecolor='white')

# Define sensors with black/white/gray colors and line styles
sensors_data = [
    {'name': 'temperature_C', 'label': 'Temperature (°C)', 'color': '#000000', 'linestyle': '-', 'linewidth': 2.5},
    {'name': 'lpg', 'label': 'LPG (ppm)', 'color': '#404040', 'linestyle': '--', 'linewidth': 2.5},
    {'name': 'ch4', 'label': 'CH4 (ppm)', 'color': '#808080', 'linestyle': '-.', 'linewidth': 2.5},
    {'name': 'smoke', 'label': 'Smoke (ppm)', 'color': '#C0C0C0', 'linestyle': ':', 'linewidth': 2.5}
]

# Plot each sensor with actual values
for sensor in sensors_data:
    col_data = data[sensor['name']]
    
    ax.plot(data.index, col_data,
            label=sensor['label'],
            linestyle=sensor['linestyle'],
            linewidth=sensor['linewidth'],
            color=sensor['color'],
            alpha=0.85,
            marker='o',
            markersize=3,
            markeredgewidth=0.5,
            markeredgecolor=sensor['color'],
            markerfacecolor=sensor['color'],
            markevery=max(1, len(data)//40))
    
    # Fill area under curve with transparency
    ax.fill_between(data.index, col_data, alpha=0.12, color=sensor['color'])

# Format plot
ax.set_ylabel('Sensor Values', fontsize=13, fontweight='bold')
ax.set_title('Графік зміни параметрів довкілля протягом 26-27 вересня, 2025',
             fontsize=15, fontweight='bold', pad=20)

# Enhanced legend with semi-transparent background
ax.legend(loc='best', fontsize=12, framealpha=0.95, edgecolor='black', 
          fancybox=True, shadow=True, frameon=True)

# Grid with transparency
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
ax.set_axisbelow(True)

# Format x-axis
n_ticks = 15
tick_positions = np.linspace(0, len(data)-1, n_ticks, dtype=int)
ax.set_xticks(tick_positions)
date_labels = [data['datetime'].iloc[i].strftime('%m-%d\n%H:%M') for i in tick_positions]
ax.set_xticklabels(date_labels, rotation=0, ha='center', fontsize=9)

# Add background color
ax.set_facecolor('#FAFAFA')

# Add spines styling
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.5)
    spine.set_alpha(0.7)

plt.tight_layout()

# Save figure
output_path = "sensor_network_colored.png"
plt.savefig(output_path, dpi=200, facecolor='white', edgecolor='none', bbox_inches='tight')
print(f"\n✓ Graph saved to {output_path}")
