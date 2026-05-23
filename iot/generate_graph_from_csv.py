"""
Build sensor network graph using real IoT data from sensor_data.csv
with transparency and improved visibility
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load real data from CSV
data = pd.read_csv('sensor_data.csv')

# Convert datetime string to datetime object
data['datetime'] = pd.to_datetime(data['datetime'])

# Sort by datetime
data = data.sort_values('datetime').reset_index(drop=True)

print("Data loaded:")
print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
print(f"Records: {len(data)}")
print(f"\nData summary:")
print(data.describe())

# Fill missing values with interpolation
for col in ['temperature_C', 'humidity_pct', 'lpg', 'ch4', 'smoke']:
    data[col] = data[col].interpolate(method='linear', limit_direction='both')

# Create plot with improved visibility
fig, ax = plt.subplots(figsize=(18, 10))

# Define line styles and colors with transparency
line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
line_widths = [2, 2.5, 2.5, 2.5, 3]
alphas = [0.9, 0.8, 0.75, 0.8, 0.85]
grays = [0.0, 0.25, 0.45, 0.65, 0.15]  # Different gray levels for better distinction

# Normalize each sensor to 0-100 scale for visualization
sensors_info = [
    ('temperature_C', 'Temperature (°C)', 0.0, 0.9),
    ('humidity_pct', 'Humidity (%)', 0.25, 0.8),
    ('lpg', 'LPG (ppm)', 0.45, 0.75),
    ('ch4', 'CH4 (ppm)', 0.65, 0.8),
    ('smoke', 'Smoke (од.)', 0.15, 0.85),
]

for (col_name, label, gray, alpha), linestyle, linewidth in zip(
    sensors_info, line_styles, line_widths):
    
    col_data = data[col_name]
    min_val = col_data.min()
    max_val = col_data.max()
    
    # Normalize to 0-100
    normalized = (col_data - min_val) / (max_val - min_val + 1e-6) * 90 + 5
    
    # Plot with transparency
    ax.plot(data.index, normalized, 
            label=label, 
            linestyle=linestyle, 
            linewidth=linewidth,
            color=str(gray),
            alpha=alpha,
            antialiased=True)
    
    # Add semi-transparent fill under the curve
    ax.fill_between(data.index, normalized, alpha=alpha*0.15, color=str(gray))

# Format plot
ax.set_xlabel('Time (Sep 26-27, 2025)', fontsize=13, fontweight='bold')
ax.set_ylabel('Sensor Values (Normalized Scale 0-100)', fontsize=13, fontweight='bold')
ax.set_title('Smart Sensor Network - Real IoT Time Series Data\nSep 26-27, 2025', 
             fontsize=15, fontweight='bold', pad=20)

# Enhanced legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black', 
          fancybox=True, shadow=True, ncol=1)

# Grid with transparency
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
ax.set_ylim(0, 105)
ax.set_xlim(0, len(data)-1)

# Format x-axis with dates
n_ticks = 16
tick_positions = np.linspace(0, len(data)-1, n_ticks, dtype=int)
ax.set_xticks(tick_positions)
date_labels = [data['datetime'].iloc[i].strftime('%m-%d %H:%M') for i in tick_positions]
ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=9)

# Improve appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

plt.tight_layout()

# Save figure with high DPI
output_path = "sensor_network_real_data.png"
plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none', bbox_inches='tight')
print(f"\nGraph saved to {output_path}")

# Print statistics
print("\n" + "="*60)
print("SENSOR DATA STATISTICS")
print("="*60)
for col_name, label, _, _ in sensors_info:
    if col_name in data.columns:
        print(f"\n{label}:")
        print(f"  Min:  {data[col_name].min():.2f}")
        print(f"  Max:  {data[col_name].max():.2f}")
        print(f"  Mean: {data[col_name].mean():.2f}")
        print(f"  Std:  {data[col_name].std():.2f}")
