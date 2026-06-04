import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_parquet(
    'data/my_meteonet_data.parquet',
    engine='pyarrow'
)

print(df.head())

# Aggregate by day

print("Aggregating data by day to prevent overplotting...")

df['date'] = pd.to_datetime(df['date'])

daily_avg = df.groupby(df['date'].dt.date).mean(numeric_only=True)
daily_avg.index = pd.to_datetime(daily_avg.index)
daily_avg['t'] = daily_avg['t']-273.15
daily_avg['td'] = daily_avg['td']-273.15

# Compute Humidex (perceived temperature)

# Temperature and dew point in Celsius
T = daily_avg['t']
Td = daily_avg['td']

# Environment Canada vapor pressure formula
e = 6.11 * np.exp(
    5417.7530 * (
        (1 / 273.16)
        - (1 / (Td + 273.15))
    )
)

# Humidex
daily_avg['humidex'] = T + (5 / 9) * (e - 10)

daily_avg['temp-diff'] = daily_avg['humidex'] - daily_avg['t']

print(daily_avg.head())

# Variables to plot

variables = [
    ('t', 'Temperature (°C)', '#d62728'),
    ('td', 'Dew Point (°C)', '#cdd627'),
    ('humidex', 'Perceived Temperature / Humidex (°C)', '#ff7f0e'),
    ('temp-diff', 'Difference', '#7661bd'),
    ('precip', 'Precipitation (mm)', '#1f77b4'),
    ('hu', 'Humidity (%)', '#2ca02c'),
    ('ff', 'Wind Speed (m/s)', '#9467bd'),
    ('psl', 'Pressure (Pa)', '#8c564b')
]

# Create plots

fig, axes = plt.subplots(
    nrows=len(variables),
    ncols=1,
    figsize=(14, 18),
    sharex=True
)

fig.suptitle(
    "Daily Average Meteorological Variables (2016–2018)",
    fontsize=16
)

for ax, (col, ylabel, color) in zip(axes, variables):

    if col in daily_avg.columns:

        values = daily_avg[col]

        ax.plot(
            daily_avg.index,
            values,
            color=color,
            linewidth=1.2
        )

        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    else:
        ax.text(
            0.5,
            0.5,
            f"Column '{col}' not found",
            ha='center',
            va='center',
            transform=ax.transAxes
#since the graph is empty (because the data is missing), there is no data axis! ax.transAxes overrides this and tells Matplotlib to use a relative bounding box for the whole chart instead. 
        )

axes[-1].set_xlabel("Date")

plt.tight_layout()
plt.subplots_adjust(top=0.96)

plt.savefig(
    "1 meteonet_timeseries_with_humidex_and_pt.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()