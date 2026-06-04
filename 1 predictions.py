import pandas as pd
import matplotlib.pyplot as plt

# 1. Read and format the data
print("Loading Parquet file...")
df = pd.read_parquet('data/my_meteonet_data.parquet', engine='pyarrow')
df['date'] = pd.to_datetime(df['date'])

# 2. Extract strictly 2016-2018 and calculate daily average temperature
daily_temp = df.groupby(df['date'].dt.date)['t'].mean()
daily_temp.index = pd.to_datetime(daily_temp.index)

# 3. Create the Forecasting Model (Historical Average)
print("Generating Naive Forecast for 2018...")

# Isolate the training data (2016 & 2017)
train_data = daily_temp.loc['2016-01-01':'2017-12-31'].to_frame()
train_data['dayofyear'] = train_data.index.dayofyear
print(train_data)
# Calculate the mean temperature for each day of the year across 2016-2017
historical_avg = train_data.groupby('dayofyear')['t'].mean()
print(historical_avg)

# 4. Apply the forecast to 2018
# Create a blank index of every day in 2018
dates_2018 = pd.date_range(start="2018-01-01", end="2018-12-31", freq="D")
predicted_2018 = pd.Series(index=dates_2018, dtype=float)

# Map the historical average to the 2018 dates based on the day of the year
# (e.g., predicted_2018's Feb 5th = historical_avg's Feb 5th)
predicted_2018.loc[:] = historical_avg.loc[dates_2018.dayofyear].values

# 5. Plotting the results
print("Generating plot...")
plt.figure(figsize=(12, 6))

# Plot the training data (2016-2017)
plt.plot(train_data.index, train_data['t'], color='#d62728', 
         label='Actual Data (2016-2017)', linewidth=1.5, alpha=0.7)

# Plot the actual 2018 data for comparison (greyed out)
plt.plot(daily_temp.loc['2018-01-01':'2018-12-31'].index, 
         daily_temp.loc['2018-01-01':'2018-12-31'].values, 
         color='black', label='Actual Data (2018)', linewidth=1, alpha=0.3)

# Plot our Prediction for 2018
plt.plot(predicted_2018.index, predicted_2018.values, 
         color='#1f77b4', label='Predicted Data (2018)', linewidth=2.5)

# Add a vertical line to explicitly separate Training vs Testing periods
plt.axvline(pd.to_datetime('2018-01-01'), color='gray', linestyle='--', alpha=0.8)

# Aesthetics
plt.title("Historical Average Forecast: Predicting 2018 based on 2016-2017", fontsize=14)
plt.ylabel("Temperature (K)")
plt.xlabel("Date")
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig("1 meteonet_2018_forecast.png", dpi=300)
print("Forecast plot saved.")
plt.show()