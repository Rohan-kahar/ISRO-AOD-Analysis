#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from io import StringIO

# Set plot style for research paper quality figures
sns.set_context("paper", font_scale=1.5)

# Load the data
df = pd.read_csv('Shastri_Nagar.csv')

# Convert date to datetime and set as index
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df.set_index('date', inplace=True)

# Create a year-month column for monthly analysis
df['year_month'] = df.index.strftime('%Y-%m')

# Calculate thresholds for the entire dataset
thresh_90 = np.percentile(df['AOD'], 90)
thresh_95 = np.percentile(df['AOD'], 95)
thresh_99 = np.percentile(df['AOD'], 99)

# Function to analyze threshold exceedances by month
def analyze_monthly_thresholds(dataframe, threshold_90, threshold_95, threshold_99):
    monthly_stats = {}
   
    for month, group in dataframe.groupby('year_month'):
        exceedances_90 = (group['AOD'] > threshold_90).sum()
        exceedances_95 = (group['AOD'] > threshold_95).sum()
        exceedances_99 = (group['AOD'] > threshold_99).sum()
        total_days = len(group)
       
        monthly_stats[month] = {
            'total_days': total_days,
            'exceedances_90': exceedances_90,
            'exceedances_95': exceedances_95,
            'exceedances_99': exceedances_99,
            'percent_90': (exceedances_90 / total_days) * 100,
            'percent_95': (exceedances_95 / total_days) * 100,
            'percent_99': (exceedances_99 / total_days) * 100,
            'mean_aod': group['AOD'].mean(),
            'max_aod': group['AOD'].max()
        }
   
    return pd.DataFrame.from_dict(monthly_stats, orient='index')

# Analyze monthly threshold exceedances
monthly_analysis = analyze_monthly_thresholds(df, thresh_90, thresh_95, thresh_99)

# Calculate monthly thresholds (changing thresholds for each month)
monthly_thresholds = df.groupby('year_month')['AOD'].agg(
    thresh_90_monthly=lambda x: np.percentile(x, 90),
    thresh_95_monthly=lambda x: np.percentile(x, 95),
    thresh_99_monthly=lambda x: np.percentile(x, 99)
)

# Note months with insufficient data for reliable percentile calculation
low_data_months = monthly_thresholds.index[df.groupby('year_month').size() < 5].tolist()
if low_data_months:
    print("\nMonths with fewer than 5 data points (less reliable percentiles):")
    for month in low_data_months:
        print(f"- {month}: {df.groupby('year_month').size()[month]} data points")

# Create a heatmap showing monthly threshold variations
monthly_data = pd.pivot_table(
    df,
    values='AOD',
    index=df.index.year,
    columns=df.index.month,
    aggfunc='mean'
)

plt.figure(figsize=(14, 8))
sns.heatmap(monthly_data, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5)
plt.title('Ashok Vihar Monthly Average AOD Values( 2023 - May 2025)', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Year')
plt.tight_layout()

# Replace month numbers with month names
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Oct', 'Nov', 'Dec']
plt.xticks(np.arange(0.5, 8.5), month_names)

# Save the heatmap
plt.savefig('aod_monthly_heatmap.png', dpi=300, bbox_inches='tight')
                
# Monthly threshold variations visualization
plt.figure(figsize=(15, 8))

# Plot monthly 90th percentile variations
plt.plot(monthly_thresholds.index, monthly_thresholds['thresh_90_monthly'],
         'o-', color='blue', markersize=6, label='Monthly 90th Percentile')
plt.axhline(y=thresh_90, color='blue', linestyle='--',
            label=f'Global 90th Percentile ({thresh_90:.2f})')

# Plot monthly 95th percentile variations
plt.plot(monthly_thresholds.index, monthly_thresholds['thresh_95_monthly'],
         'o-', color='red', markersize=6, label='Monthly 95th Percentile')
plt.axhline(y=thresh_95, color='red', linestyle='--',
            label=f'Global 95th Percentile ({thresh_95:.2f})')

# Plot monthly 99th percentile variations
plt.plot(monthly_thresholds.index, monthly_thresholds['thresh_99_monthly'],
         'o-', color='darkred', markersize=6, label='Monthly 99th Percentile')
plt.axhline(y=thresh_99, color='darkred', linestyle='--',
            label=f'Global 99th Percentile ({thresh_99:.2f})')

plt.xlabel('Month')
plt.ylabel('AOD Threshold Value')
plt.title('Ashok Vihar Monthly Variations in AOD Threshold Values(Nov 2023 - May 2025)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper center')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the monthly threshold variations plot
plt.savefig('aod_monthly_threshold_variations.png', dpi=300, bbox_inches='tight')





# Monthly max and min AOD
monthly_extremes = df.groupby('year_month')['AOD'].agg(['max', 'min']).reset_index()

plt.figure(figsize=(15, 7))
plt.plot(monthly_extremes['year_month'], monthly_extremes['max'], 'o-', color='red', label='Max AOD')
plt.plot(monthly_extremes['year_month'], monthly_extremes['min'], 'o-', color='blue', label='Min AOD')

# Annotate max values
for i, row in monthly_extremes.iterrows():
    plt.text(i, row['max'] + 0.02, f"{row['max']:.2f}", ha='center', va='bottom', fontsize=10, color='red')

# Annotate min values
for i, row in monthly_extremes.iterrows():
    plt.text(i, row['min'] - 0.02, f"{row['min']:.2f}", ha='center', va='top', fontsize=10, color='blue')

plt.xlabel('Month')
plt.ylabel('AOD Value')
plt.title('Ashok Vihar Monthly Maximum and Minimum AOD Values(Nov 2023 - May 2025)', fontsize=16)
plt.xticks(ticks=range(len(monthly_extremes)), labels=monthly_extremes['year_month'], rotation=90)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper center')
plt.tight_layout()

# Save the figure
plt.savefig('aod_monthly_max_min_plot_annotated.png', dpi=300, bbox_inches='tight')




monthly_mode = df.groupby([df.index.year, df.index.month])['AOD'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
).unstack()

plt.figure(figsize=(14, 8))
sns.heatmap(monthly_mode, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=.5)
plt.title('Ashok Vihar Nagar Monthly Most Occurring AOD Values(Nov 2023 - May 2025)', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Year')
plt.tight_layout()

# Set month labels
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Oct', 'Nov', 'Dec']
plt.xticks(np.arange(0.5, 8.5), month_names)

# Save the heatmap
plt.savefig('aod_monthly_mode_heatmap.png', dpi=300, bbox_inches='tight')

