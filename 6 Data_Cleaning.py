import pandas as pd
import numpy as np
import os
from scipy.spatial import cKDTree
from math import radians, cos, sin, asin, sqrt

# Haversine formula to accurately calculate km distance from lat/lon
def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c # Radius of earth in kilometers

def main():
    print("Loading Parquet file...")
    data_path = os.path.join(os.getcwd(), "data/my_meteonet_data.parquet")
    df = pd.read_parquet(data_path, engine='pyarrow')
    
    # Ensure date is a proper datetime object so we can align timestamps perfectly
    df['date'] = pd.to_datetime(df['date'])

    # 1. Extract Unique Stations and Build KDTree
    print("Finding nearest neighbor for each station...")
    stations = df.groupby('number_sta')[['lat', 'lon', 'height_sta']].first().reset_index()
    
    R = 6371 # Earth radius
    stations['x'] = R * np.cos(np.radians(stations['lat'])) * np.cos(np.radians(stations['lon']))
    stations['y'] = R * np.cos(np.radians(stations['lat'])) * np.sin(np.radians(stations['lon']))
    stations['z'] = R * np.sin(np.radians(stations['lat']))
    
    tree = cKDTree(stations[['x', 'y', 'z']].values)
    distances, indices = tree.query(stations[['x', 'y', 'z']].values, k=2)

    vars_to_copy = ['t', 'td', 'hu']
    print("Executing Spatial Imputation...")
    total_imputed = {var: 0 for var in vars_to_copy + ['psl']}

    # 2. Iterate through each station's closest neighbor
    for i, row in stations.iterrows():
        station_A_id = row['number_sta']
        
        # Extract Station A data
        df_A = df[df['number_sta'] == station_A_id]
        
        # EFFICIENCY CHECK: If Station A has NO missing values for these columns, skip it immediately!
        cols_to_check = vars_to_copy + (['psl'] if 'psl' in df.columns else [])
        if df_A[cols_to_check].isna().sum().sum() == 0:
            continue # Move to the next station to save time
        
        neighbor_idx = indices[i][1]
        station_B_id = stations.iloc[neighbor_idx]['number_sta']
        
        # Calculate true Haversine distance
        dist_km = haversine_distance(
            row['lon'], row['lat'], 
            stations.iloc[neighbor_idx]['lon'], stations.iloc[neighbor_idx]['lat']
        )
        
        # Only proceed if the stations are closer than 15km
        if dist_km <= 15.0:
            df_B = df[df['number_sta'] == station_B_id]
            
            # merge_asof requires the dataframes to be sorted by the timestamp
            df_A = df_A.sort_values('date')
            df_B = df_B.sort_values('date')
            
            # FUZZY TIME MATCHING: Align df_A and df_B based on the NEAREST timestamp.
            # We enforce a maximum tolerance of 1 hour. If Station B's reading is more than 1 hour away, it won't copy.
            aligned = pd.merge_asof(
                df_A[['date'] + cols_to_check], 
                df_B[['date'] + cols_to_check], 
                on='date', 
                direction='nearest', 
                tolerance=pd.Timedelta(hours=1), 
                suffixes=('', '_B')
            )
            
            # Set the index back to the date so our boolean masks map perfectly back to the main dataframe
            aligned.set_index('date', inplace=True)
            
            # 3. Impute Standard Variables (Temp, Dew Point, Humidity)
            for var in vars_to_copy:
                # Mask: Station A is missing the value, BUT Station B has a value within the 1-hour tolerance
                mask = aligned[var].isna() & aligned[f'{var}_B'].notna()
                impute_count = mask.sum()
                
                if impute_count > 0:
                    df.loc[(df['number_sta'] == station_A_id) & (df['date'].isin(aligned[mask].index)), var] = aligned.loc[mask, f'{var}_B'].values
                    total_imputed[var] += impute_count

            # 4. Special Imputation for Pressure (Requires Height check)
            if 'psl' in df.columns:
                height_diff = abs(row['height_sta'] - stations.iloc[neighbor_idx]['height_sta'])

                mask_psl = aligned['psl'].isna() & aligned['psl_B'].notna()
                psl_count = mask_psl.sum()
                    
                if psl_count > 0:
                    df.loc[(df['number_sta'] == station_A_id) & (df['date'].isin(aligned[mask_psl].index)), 'psl'] = aligned.loc[mask_psl, 'psl_B'].values
                    total_imputed['psl'] += psl_count

    print("\n--- SPATIAL IMPUTATION RESULTS ---")
    for var, count in total_imputed.items():
        print(f"Values recovered for {var}: {count:,}")
        
    print("\nSaving spatially imputed Parquet file...")
    df.to_parquet("meteonet_spatial_imputed.parquet", engine='pyarrow')
    print("Done.")

if __name__ == "__main__":
    main()