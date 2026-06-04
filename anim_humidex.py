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
df['td'] = pd.to_numeric(df['td'], errors='coerce') # Need Dew Point for Humidex
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['number_sta'] = df['number_sta'].astype(str)

# Drop rows missing crucial data
df = df.dropna(subset=['t', 'td', 'lat', 'lon'])

# Convert Kelvin to Celsius for Temperature 
# (We keep a separate Kelvin series for the vapor pressure calculation)
if df['t'].mean() > 200:
    df['t_celsius'] = df['t'] - 273.15
    df['td_kelvin'] = df['td'] # Dew point needs to be in Kelvin for the e formula
else:
    df['t_celsius'] = df['t']
    df['td_kelvin'] = df['td'] + 273.15

# Apply the 25k robust station mask
valid_station_mask = df.groupby('number_sta')['t'].transform('count') > 25000
df = df[valid_station_mask].copy()

print("2. Calculating Humidex and Daily Delta...")
# Calculate vapor pressure (e)
# Formula: e = 6.11 * exp(5417.7530 * ((1/273.16) - (1/Td_kelvin)))
df['e'] = 6.11 * np.exp(5417.7530 * ((1 / 273.16) - (1 / df['td_kelvin'])))

# Calculate Humidex (H)
df['humidex'] = df['t_celsius'] + (5/9) * (df['e'] - 10)

# Calculate the Delta: How much hotter does it feel compared to the actual temp?
# We clip at 0 so we only look at instances where it feels hotter, ignoring winter windchill effects
df['felt_delta'] = (df['humidex'] - df['t_celsius']).clip(lower=0)

df['day'] = df['date'].dt.date

# Get the daily average of the Delta for EVERY station
daily_stats = df.groupby(['day', 'number_sta', 'lat', 'lon'])['felt_delta'].agg(
    Daily_Delta='mean'
).reset_index()

print("3. Calculating 14-day rolling average per station...")
# Sort chronologically
daily_stats = daily_stats.sort_values(by=['number_sta', 'day'])

# Apply a 14-day rolling average to the felt delta
daily_stats['Delta_14d_avg'] = daily_stats.groupby('number_sta')['Daily_Delta'].transform(
    lambda x: x.rolling(window=14, min_periods=1).mean()
)

daily_stats = daily_stats.dropna(subset=['Delta_14d_avg'])

print("4. Preparing the map geometry...")
coastline_url = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
coast_gdf = gpd.read_file(coastline_url)

output_dir = "animation_frames"
os.makedirs(output_dir, exist_ok=True)

# Lock the colorbar scale based on the delta 
vmin = 0.0  # Lowest delta is 0 (feels exactly like actual temp)
vmax = np.ceil(daily_stats['Delta_14d_avg'].quantile(0.99)) # Find a robust upper limit

unique_days = sorted(daily_stats['day'].unique())
total_frames = len(unique_days)

print(f"5. Generating {total_frames} map frames...")

for i, current_day in enumerate(unique_days):
    day_data = daily_stats[daily_stats['day'] == current_day]
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#f4f4f9')
    ax.set_facecolor('#e0f3ff') 
    
    coast_gdf.plot(ax=ax, color='#333333', linewidth=1.2)
    
    # Switched colormap to 'Reds' to visually imply "added heat/stiffness"
    scatter = ax.scatter(
        day_data['lon'], 
        day_data['lat'], 
        c=day_data['Delta_14d_avg'], 
        cmap='Reds', 
        s=80,             
        edgecolor='black', 
        linewidth=0.5, 
        alpha=0.9,
        vmin=vmin, 
        vmax=vmax,
        zorder=5          
    )
    
    ax.set_xlim([-5.5, 2.5])
    ax.set_ylim([46.0, 51.5])
    
    ax.set_title(f"14-Day Average: Humidex Delta (Felt Temp - Actual Temp)\nDate: {current_day}", fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.036, pad=0.04)
    cbar.set_label('Humidex Delta in °C', fontsize=12, labelpad=10)
    
    filename = os.path.join(output_dir, f"frame_{i:04d}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if i % 50 == 0 or i == total_frames - 1:
        print(f"   Rendered {i}/{total_frames} frames...")

print(f"\nSuccess! All frames have been saved to the '{output_dir}' folder.")