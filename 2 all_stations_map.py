import pandas as pd
import folium
import os

def main():
    # 1. Load the data
    print("Loading Parquet file...")
    data_path = os.path.join(os.getcwd(), "data", "my_meteonet_data.parquet")
    df = pd.read_parquet(data_path, engine='pyarrow')

    # 2. Extract UNIQUE stations to avoid overplotting and memory crashes
    # A station's location doesn't change, so we just grab its first recorded lat/lon
    print("Extracting unique spatial coordinates...")
    stations_df = df.groupby('number_sta')[['lat', 'lon']].first().reset_index()
    print(stations_df)
    # 3. Create the Folium Map
    # Center it dynamically based on the actual average coordinates of all stations
    lat_mean = stations_df['lat'].mean()
    lon_mean = stations_df['lon'].mean()
    m = folium.Map(location=[lat_mean, lon_mean], zoom_start=6)

    # 4. Plot every unique station
    print(f"Plotting {len(stations_df)} unique stations on the map...")
    for _, row in stations_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            color="#3186cc", # Standard clear blue
            fill=True,
            fill_color="#3186cc",
            fill_opacity=0.8,
            popup=f"Station: {row['number_sta']}"
        ).add_to(m)

    # 5. Save the map
    output_filename = "2 all_stations_map.html"
    m.save(output_filename)
    print(f"Map successfully saved to {output_filename}. Open it in your web browser.")

if __name__ == "__main__":
    main()