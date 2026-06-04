import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. Read the file
    print("Loading Parquet file...")
    df = pd.read_parquet('data/my_meteonet_data.parquet', engine='pyarrow')

    # 2. Identify reliable vs unreliable stations
    print("Categorizing stations by frequency...")
    station_counts = df['number_sta'].value_counts()
    
    # Create lists of the station IDs that fall into each bucket
    high_freq_stations = station_counts[station_counts >= 25000].index
    low_freq_stations = station_counts[station_counts < 25000].index

    # 3. Create two separate DataFrames based on the station lists
    df_high = df[df['number_sta'].isin(high_freq_stations)]
    df_low = df[df['number_sta'].isin(low_freq_stations)]

    # 4. Define the meteorological columns we care about
    cols_to_check = ['t', 'precip', 'hu', 'ff', 'psl']
    
    # 5. Calculate the percentage of NaNs for each column in both DataFrames
    # isna().mean() calculates the mathematical ratio of NaNs. We multiply by 100 for percentages.
    high_missing_pct = df_high[cols_to_check].isna().mean() * 100
    low_missing_pct = df_low[cols_to_check].isna().mean() * 100

    # Combine the results into a single summary table for easy viewing
    summary = pd.DataFrame({
        '>= 25,000 Obs (225 stations)': high_missing_pct,
        '< 25,000 Obs (62 stations)': low_missing_pct
    })
    
    # Rename index for better readability
    summary.index = ['Temperature (t)', 'Precipitation (precip)', 'Humidity (hu)', 'Wind Speed (ff)', 'Pressure (psl)']

    print("\n--- PERCENTAGE OF MISSING DATA (NaNs) ---")
    print("This shows the % of rows where the sensor failed to record the specific variable:")
    print(summary.round(2))

    # 6. Plotting the Comparison
    print("\nGenerating bar chart...")
    # We transpose the summary (.T) to group the bars by the columns instead of the categories
    ax = summary.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#d62728'], edgecolor='black', alpha=0.8)
    
    plt.title("Data Quality: Percentage of Missing Values\n(High Frequency vs Low Frequency Stations)", fontsize=14)
    plt.ylabel("Percentage of Missing Data (%)")
    plt.xlabel("Meteorological Variable")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Automatically add the exact percentage number on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    
    plt.tight_layout()
    plt.savefig("5 data_quality_comparison.png", dpi=300)
    print("Bar chart saved as 'data_quality_comparison.png'")
    plt.show()

if __name__ == "__main__":
    main()