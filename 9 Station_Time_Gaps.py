import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_station_gaps_linear():
    print("Loading Parquet file...")
    data_path = os.path.join(os.getcwd(), "data/my_meteonet_data.parquet")
    df = pd.read_parquet(data_path, engine='pyarrow')
    df['date'] = pd.to_datetime(df['date'])

    # 1. Isolate the low-frequency stations
    print("Filtering low-frequency stations...")
    station_frequencies = df['number_sta'].value_counts()
    low_freq_stations = station_frequencies[station_frequencies < 25000].index.tolist()

    df_low = df[df['number_sta'].isin(low_freq_stations)].copy()
    
    # 2. Sort chronologically per station and calculate the gap
    df_low = df_low.sort_values(by=['number_sta', 'date'])
    df_low['time_gap'] = df_low.groupby('number_sta')['date'].diff()
    df_low['gap_hours'] = df_low['time_gap'].dt.total_seconds() / 3600.0

    # 3. Aggregate the average and median gaps per station
    print("Calculating Average and Median gaps...")
    analysis = []
    for sta, group in df_low.groupby('number_sta'):
        valid_gaps = group['gap_hours'].dropna()
        if len(valid_gaps) > 0:
            avg_gap = valid_gaps.mean()
            median_gap = valid_gaps.median()
        else:
            avg_gap = 0
            median_gap = 0
            
        analysis.append({
            'Station': str(sta),
            'Average Gap (Hours)': avg_gap,
            'Median Gap (Hours)': median_gap
        })
        
    results_df = pd.DataFrame(analysis)
    
    # Sort by Average Gap for a clean, readable chart (descending staircase)
    results_df = results_df.sort_values(by='Average Gap (Hours)', ascending=False)
    results_df.set_index('Station', inplace=True)
    
    # 4. Plotting the Bar Chart
    print("Generating Chart...")
    
    # Plotting Average (Orange) and Median (Green)
    ax = results_df[['Average Gap (Hours)', 'Median Gap (Hours)']].plot(
        kind='bar', 
        figsize=(18, 6), 
        color=['#ff7f0e', '#2ca02c'], 
        width=0.8,
        edgecolor='black',
        alpha=0.85
    )
    
    plt.title("Average vs. Median Time Gap Between Readings (Linear Scale)", fontsize=16)
    plt.ylabel("Time Gap in Hours (Linear)")
    plt.xlabel("Station ID")
    
    # Format X-axis so station IDs are vertical and legible
    plt.xticks(rotation=90, fontsize=6) 
    
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("9 station_time_gaps_linear.png", dpi=300)
    print("Bar chart saved as 'station_time_gaps_linear.png'")
    plt.show()

if __name__ == "__main__":
    plot_station_gaps_linear()