#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import gaussian_kde
from scipy.integrate import cumtrapz
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Step 0: Load shapefile and filter for required states
india_states = gpd.read_file(r"C:\Users\admin\rohan\IndiaStateDistShapeFile\STATE_BDY.shp")  
# Update this path
target_states = ['UTTARPRADESH', 'BIHAR', 'HARYANA', 'DELHI', 'PUNJAB']
selected_states = india_states[india_states['STATE'].isin(target_states)]

# Step 1: Load all daily CSVs for the month
csv_files = glob(r"C:\Users\admin\rohan\DATA\OUTPUT\2023\NOV\*.csv")  
# Update this path
df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Step 2: Create geometry and convert DataFrame to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# Step 3: Spatial join to retain only points within the selected states
gdf_filtered = gpd.sjoin(gdf, selected_states, how='inner',  
predicate='within')

# Step 4: Extract AOD values and drop NaNs
aod_values = gdf_filtered['AOD'].dropna()

# Step 5: Estimate PDF using KDE
kde = gaussian_kde(aod_values)
x_vals = np.linspace(aod_values.min(), aod_values.max(), 1000)
pdf_vals = kde(x_vals)

# Step 6: Compute CDF using cumulative trapezoidal integration
cdf_vals = cumtrapz(pdf_vals, x_vals, initial=0)
cdf_vals /= cdf_vals[-1]  # Normalize to [0,1]

# Step 7: Find percentiles
p90 = x_vals[np.searchsorted(cdf_vals, 0.90)]
p95 = x_vals[np.searchsorted(cdf_vals, 0.95)]
p99 = x_vals[np.searchsorted(cdf_vals, 0.99)]

# Print results
print(f"From estimated PDF (only UP, Bihar, Haryana, Delhi, Punjab):")
print(f"90th percentile AOD: {p90:.4f}")
print(f"95th percentile AOD: {p95:.4f}")
print(f"99th percentile AOD: {p99:.4f}")

# Step 8: Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, pdf_vals, label='Estimated PDF', color='blue')
line_90 = plt.axvline(p90, color='orange', linestyle='--', label=f'90th Percentile: {p90:.4f}')
line_95 = plt.axvline(p95, color='red', linestyle='--', label=f'95th Percentile: {p95:.4f}')
line_99 = plt.axvline(p99, color='purple', linestyle='--', label=f'99th Percentile: {p99:.4f}')

# Adding text labels for the percentiles
#plt.text(p90, max(pdf_vals)*0.05, f'{p90:.4f}', color='orange', ha='center')
#plt.text(p95, max(pdf_vals)*0.05, f'{p95:.4f}', color='red', ha='center')
#plt.text(p99, max(pdf_vals)*0.05, f'{p99:.4f}', color='purple', ha='center')

plt.xlabel('AOD')
plt.ylabel('Density')
plt.title('NOV 2023 Estimated PDF of AOD (Filtered by States)')

# Update the legend to include percentiles
plt.legend(handles=[line_90, line_95, line_99], loc='best')

plt.grid(True)
plt.tight_layout()
plt.show()

