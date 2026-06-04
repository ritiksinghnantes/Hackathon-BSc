import pandas as pd
import folium
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    # 1. Load the data
    print("Loading Parquet file...")
    data_path = os.path.join(os.getcwd(), "data", "my_meteonet_data.parquet")
    df = pd.read_parquet(data_path, engine='pyarrow')

    # PART 1: CALCULATE FREQUENCIES
    print("Calculating station frequencies...")
    station_frequencies = df['number_sta'].value_counts()
    
    # Convert to DataFrame for the map merge
    freq_df = station_frequencies.reset_index()
    freq_df.columns = ['number_sta', 'frequency']

    # PART 2: FOLIUM MAP (Conditional Outline)
    print("Extracting unique spatial coordinates for the map...")
    coords_df = df.groupby('number_sta')[['lat', 'lon']].first().reset_index()
    stations_df = pd.merge(coords_df, freq_df, on='number_sta')

    print("Configuring color gradients and generating map...")
    min_freq = stations_df['frequency'].min()
    max_freq = stations_df['frequency'].max()
    cmap = plt.get_cmap('coolwarm') 

    lat_mean = stations_df['lat'].mean()
    lon_mean = stations_df['lon'].mean()
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=6)

    # VARIABLES
    coverage_radius_meters = 15000 
    threshold = 25000

    for _, row in stations_df.iterrows():
        # Normalize between 0 and 1 for the colormap
        if max_freq > min_freq:
            norm_freq = (row['frequency'] - min_freq) / (max_freq - min_freq)
        else:
            norm_freq = 0.5
            
        hex_color = mcolors.to_hex(cmap(norm_freq))

        # 1. The Large Hollow Outline Circle - ONLY for unreliable stations (< 25,000)
        if row['frequency'] < threshold:
            folium.Circle(
                location=[row["lat"], row["lon"]],
                radius=coverage_radius_meters, 
                color=hex_color,               
                weight=2,                      
                fill=False                     # Hollow inside
            ).add_to(m)

        # 2. The Solid Center Dot - FOR ALL STATIONS
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,                      
            color=hex_color,
            weight=1,
            fill=True,                     
            fill_color=hex_color,
            fill_opacity=1.0,              
            popup=f"Station ID: {row['number_sta']}<br>Observations: {row['frequency']}"
        ).add_to(m)

    map_filename = "3 all_stations_frequency_map.html"
    m.save(map_filename)
    print(f"Map successfully saved to '{map_filename}'")

    # PART 3: BAR CHART (With Threshold Line)
    print("\nGenerating the frequency bar chart...")
    
    total_unique_stations = df['number_sta'].nunique()
    valid_stations = station_frequencies[station_frequencies >= threshold].index
    
    print(f"Total Unique Stations: {total_unique_stations}")
    print(f"Stations >= {threshold} obs: {len(valid_stations)}")
    print(f"Stations < {threshold} obs: {total_unique_stations - len(valid_stations)}")

    plt.figure(figsize=(14, 6)) 
    
    # Plot the bar chart directly from the Series
    station_frequencies.plot(kind='bar', color='#1f77b4', edgecolor='black', alpha=0.8)

    # Add the horizontal threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Reliability Threshold ({threshold})')

    # Formatting
    plt.title(f"Observation Frequency by Station\n(Total: {total_unique_stations} | Valid: {len(valid_stations)})", fontsize=15)
    plt.xlabel("Station ID (number_sta)")
    plt.ylabel("Frequency (Total Observations)")
    
    # Rotate the x-axis labels so they don't overlap
    plt.xticks(rotation=90, fontsize=3) 
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()

    plt.tight_layout()
    chart_filename = "3 station_frequencies_threshold.png"
    plt.savefig(chart_filename, dpi=600)
    print(f"Bar chart successfully saved to '{chart_filename}'")

if __name__ == "__main__":
    main()