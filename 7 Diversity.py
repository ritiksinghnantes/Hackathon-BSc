import pandas as pd
import folium
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    # 1. Load the data
    print("Loading Parquet file...")
    data_path = os.path.join(os.getcwd(), "meteonet_spatial_imputed.parquet")
    df = pd.read_parquet(data_path, engine='pyarrow')

    # PART 1: CALCULATE DIVERSITY SCORE
    print("Calculating station data diversity...")
    meteo_cols = ['t', 'td', 'hu', 'ff', 'dd', 'precip', 'psl']
    
    # Create a boolean DataFrame: True if the value exists, False if it is NaN
    valid_readings = df[meteo_cols].notna()
    valid_readings['number_sta'] = df['number_sta']
    
    # Group by station.
    # .any() checks if the station has AT LEAST ONE valid reading for a specific column.
    # .sum(axis=1) counts how many columns came back as True (Max score: 7)
    diversity_df = valid_readings.groupby('number_sta').any().sum(axis=1).reset_index()
    diversity_df.columns = ['number_sta', 'diversity_score']

    # PART 2: FOLIUM MAP (Colored by Diversity)
    print("Extracting unique spatial coordinates for the map...")
    coords_df = df.groupby('number_sta')[['lat', 'lon']].first().reset_index()
    stations_df = pd.merge(coords_df, diversity_df, on='number_sta')

    print("Configuring purple color gradients...")
    min_div = stations_df['diversity_score'].min()
    max_div = stations_df['diversity_score'].max()
    
    # Use Matplotlib's sequential purple colormap
    cmap = plt.get_cmap('Purples')

    lat_mean = stations_df['lat'].mean()
    lon_mean = stations_df['lon'].mean()
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=6)

    for _, row in stations_df.iterrows():
        # Normalize the score
        if max_div > min_div:
            raw_norm = (row['diversity_score'] - min_div) / (max_div - min_div)
            
            # The bottom of the 'Purples' colormap is pure white, which vanishes on the map.
            # We scale the math so the minimum value starts at 0.3 (visible light purple) 
            # and goes up to 1.0 (dark purple).
            norm_div = 0.3 + (raw_norm * 0.7)
        else:
            norm_div = 0.7
            
        hex_color = mcolors.to_hex(cmap(norm_div))

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            # We add a thin dark gray outline so the very light purple dots don't vanish into the map background
            color="#555555", 
            weight=1,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            popup=f"Station ID: {row['number_sta']}<br>Sensors Working: {row['diversity_score']} / {len(meteo_cols)}"
        ).add_to(m)

    map_filename = "7b diversity_map.html"
    m.save(map_filename)
    print(f"Map successfully saved to '{map_filename}'. Open it in your browser.")

if __name__ == "__main__":
    main()