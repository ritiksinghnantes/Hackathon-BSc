import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import os

print("1. Loading and cleaning data...")
df = pd.read_parquet("data/meteonet_spatial_imputed.parquet", engine="pyarrow")

# Format columns
df['date'] = pd.to_datetime(df['date'])
df['t'] = pd.to_numeric(df['t'], errors='coerce')
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['number_sta'] = df['number_sta'].astype(str)
df = df.dropna(subset=['t', 'lat', 'lon'])

# Convert Kelvin to Celsius if needed
if df['t'].mean() > 200:
    df['t'] = df['t'] - 273.15

# Apply the 25k robust station mask
valid_station_mask = df.groupby('number_sta')['t'].transform('count') > 25000
df = df[valid_station_mask].copy()

print("2. Calculating daily amplitude per station...")
df['day'] = df['date'].dt.date

# Get daily Min/Max for EVERY station
daily_stats = df.groupby(['day', 'number_sta', 'lat', 'lon'])['t'].agg(
    T_max='max',
    T_min='min'
).reset_index()

daily_stats['T_amplitude'] = daily_stats['T_max'] - daily_stats['T_min']

print("3. Calculating 14-day rolling average per station...")
# Sort chronologically so the rolling window moves forward in time correctly
daily_stats = daily_stats.sort_values(by=['number_sta', 'day'])

# Apply a 14-day rolling average to the amplitude, isolated per station
daily_stats['T_amp_14d_avg'] = daily_stats.groupby('number_sta')['T_amplitude'].transform(
    lambda x: x.rolling(window=14, min_periods=1).mean()
)

# Drop any potential NaN values that snuck through
daily_stats = daily_stats.dropna(subset=['T_amp_14d_avg'])

print("4. Preparing the map geometry...")
# Load the 10m physical coastline for the background map
coastline_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
coast_gdf = gpd.read_file(coastline_url)

# Set up the folder to hold the animation frames
output_dir = "animation_frames"
os.makedirs(output_dir, exist_ok=True)

# Find the absolute min and max of our rolling average to lock the colorbar scale
# (If we don't lock it, the colors will change meaning on every frame!)
vmin = np.floor(daily_stats['T_amp_14d_avg'].quantile(0.01)) # Typically around 2-3°C
vmax = np.ceil(daily_stats['T_amp_14d_avg'].quantile(0.99))  # Typically around 12-15°C

unique_days = sorted(daily_stats['day'].unique())
total_frames = len(unique_days)

print(f"5. Generating {total_frames} map frames...")

for i, current_day in enumerate(unique_days):
    # Filter data for just this specific day
    day_data = daily_stats[daily_stats['day'] == current_day]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#f4f4f9')
    ax.set_facecolor('#e0f3ff') # Light blue ocean background
    
    # Plot the coastline
    coast_gdf.plot(ax=ax, color='#333333', linewidth=1.2)
    
    # Plot the weather stations using a colormap (Plasma looks great for heat/temperature)
    scatter = ax.scatter(
        day_data['lon'], 
        day_data['lat'], 
        c=day_data['T_amp_14d_avg'], 
        cmap='plasma', 
        s=80,             # Dot size
        edgecolor='black', 
        linewidth=0.5, 
        alpha=0.9,
        vmin=vmin, 
        vmax=vmax,
        zorder=5          # Ensure dots render on top of the coastline
    )
    
    # Zoom the map in on Northwestern France
    ax.set_xlim([-5.5, 2.5])
    ax.set_ylim([46.0, 51.5])
    
    # Clean up the visual aesthetics
    ax.set_title(f"14-Day Average Temperature Amplitude\nDate: {current_day}", fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add the locked colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label('Temperature Delta ($T_{max} - T_{min}$) in °C', fontsize=12, labelpad=10)
    
    # Save the frame with leading zeros so they sort correctly alphabetically (frame_0001, frame_0002...)
    filename = os.path.join(output_dir, f"frame_{i:04d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig) # Close the figure to free up system memory
    
    # Print progress every 50 frames so you know it hasn't frozen
    if i % 50 == 0 or i == total_frames - 1:
        print(f"   Rendered {i}/{total_frames} frames...")

print(f"\nSuccess! All frames have been saved to the '{output_dir}' folder.")