import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_station_downtime_chart():
    # 1. Load Data
    print("Loading Parquet file...")
    data_path = os.path.join(os.getcwd(), "data/my_meteonet_data.parquet")
    df = pd.read_parquet(data_path, engine='pyarrow')
    df['date'] = pd.to_datetime(df['date'])

    # 2. Isolate Low-Frequency Stations
    station_frequencies = df['number_sta'].value_counts()
    low_freq_stations = station_frequencies.index.tolist()

    df_low = df[df['number_sta'].isin(low_freq_stations)].copy()
    
    # 3. Time Gap Calculation
    df_low = df_low.sort_values(by=['number_sta', 'date'])
    df_low['time_gap'] = df_low.groupby('number_sta')['date'].diff()
    df_low['gap_hours'] = df_low['time_gap'].dt.total_seconds() / 3600.0

    # 4. Aggregating the analysis per station
    analysis = []
    for sta, group in df_low.groupby('number_sta'):
        first_reading = group['date'].min()
        last_reading = group['date'].max()
        
        # Calculate Total Lifespan in Days
        lifespan_days = (last_reading - first_reading).total_seconds() / 86400.0
        
        # Calculate Max Downtime in Days
        valid_gaps = group['gap_hours'].dropna()
        max_gap_days = valid_gaps.max() / 24.0 if len(valid_gaps) > 0 else 0
            
        analysis.append({
            'Station': str(sta),
            'Total Lifespan (Days)': lifespan_days,
            'Max Downtime (Days)': max_gap_days
        })
        
    results_df = pd.DataFrame(analysis)
    
    # Sort the dataframe by Total Lifespan so the chart looks like a clean descending staircase
    results_df = results_df.sort_values(by='Total Lifespan (Days)', ascending=False)
    results_df.set_index('Station', inplace=True)
    
    # 5. Plotting the Grouped Bar Chart
    print("Generating Chart...")
    
    # We plot both columns. Pandas will automatically put them side-by-side.
    ax = results_df[['Total Lifespan (Days)', 'Max Downtime (Days)']].plot(
        kind='bar', 
        figsize=(18, 6), 
        color=['#1f77b4', '#d62728'], 
        width=0.8,
        edgecolor='black',
        alpha=0.85
    )
    
    # Formatting
    plt.title("Station Health: Total Lifespan vs. Maximum Continuous Downtime", fontsize=16)
    plt.ylabel("Time (Days)")
    plt.xlabel("Station ID")
    
    # Rotate the x-axis labels 90 degrees so the Station IDs are readable
    plt.xticks(rotation=90, fontsize=2) 
    
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("8 station_lifespan_downtime.png", dpi=600)
    print("Bar chart successfully saved as 'station_lifespan_downtime.png'")
    plt.show()

if __name__ == "__main__":
    generate_station_downtime_chart()