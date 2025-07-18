#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Define thresholds for hotspot categories
THRESHOLDS = {
    'moderate': 0.7000,  # 90th percentile
    'severe': 0.9040,    # 95th percentile
    'extreme': 1.6601    # 99th percentile
}

# Define custom colors for each category
COLORS = {
    'moderate': 'blue',
    'severe': 'yellow',
    'extreme': 'red'
}

def load_and_process_csv_files(data_dir, pattern="*.csv"):
    print("Loading CSV files...")
    all_files = glob.glob(os.path.join(data_dir, pattern))
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            required_cols = ['date', 'latitude', 'longitude', 'AOD']
            if not all(col in df.columns for col in required_cols):
                cols = df.columns
                if len(cols) >= 4:
                    df.columns = required_cols[:len(cols)]
                else:
                    print(f"Skipping {filename} - insufficient columns")
                    continue
            dfs.append(df)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df)} records from {len(all_files)} files")
        return combined_df
    else:
        raise ValueError("No valid data found in the specified files")

def categorize_aod_values(df):
    print("Categorizing AOD values...")
    conditions = [
        (df['AOD'] >= THRESHOLDS['extreme']),
        (df['AOD'] >= THRESHOLDS['severe']) & (df['AOD'] < THRESHOLDS['extreme']),
        (df['AOD'] >= THRESHOLDS['moderate']) & (df['AOD'] < THRESHOLDS['severe']),
        (df['AOD'] < THRESHOLDS['moderate'])
    ]
    choices = ['extreme', 'severe', 'moderate', 'normal']
    df['category'] = np.select(conditions, choices, default='normal')
    df = df[df['category'] != 'normal']
    return df

def handle_overlapping_points(df, grid_size=0.01):
    print("Handling overlapping points...")
    df['grid_lat'] = np.floor(df['latitude'] / grid_size) * grid_size
    df['grid_lon'] = np.floor(df['longitude'] / grid_size) * grid_size
    category_values = {'extreme': 3, 'severe': 2, 'moderate': 1, 'normal': 0}
    df['cat_value'] = df['category'].map(category_values)
    grid_stats = df.groupby(['grid_lat', 'grid_lon']).agg({
        'AOD': 'mean',
        'cat_value': 'max',
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    reverse_mapping = {v: k for k, v in category_values.items()}
    grid_stats['category'] = grid_stats['cat_value'].map(reverse_mapping)
    return grid_stats

def filter_by_states(df, shapefile_path, states):
    print(f"Filtering data for states: {', '.join(states)}...")
    india_states = gpd.read_file(shapefile_path)
    selected_states = india_states[india_states['STATE'].isin(states)]
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=selected_states.crs)
    within_states = gpd.sjoin(gdf, selected_states, how='inner', predicate='within')
    print(f"Filtered from {len(df)} to {len(within_states)} points")
    return within_states

def plot_hotspots(filtered_df, shapefile_path, states, output_file="15. APRIL_2025.png"):
    print("Creating hotspot visualization...")
    india_states = gpd.read_file(shapefile_path)
    selected_states = india_states[india_states['STATE'].isin(states)]

    fig, ax = plt.subplots(figsize=(20, 16))
    selected_states.boundary.plot(ax=ax, linewidth=2, color='black')

    for category, color in COLORS.items():
        category_points = filtered_df[filtered_df['category'] == category]
        if not category_points.empty:
            ax.scatter(
                category_points['longitude'],
                category_points['latitude'],
                c=color,
                s=0.5,
                alpha=0.5,
                label=category.capitalize()
            )

    # Axis labels and title
    ax.set_xlabel('Longitude', fontsize=28)
    ax.set_ylabel('Latitude', fontsize=28)
    ax.set_title('EOS-06 OCM AOD Hotspot Distribution of Indo-Gangetic Plain April 2025', fontsize=28)

    # Ticks font size
    ax.tick_params(axis='both', labelsize=24)

    # Legend inside plot
    legend_elements = [
        Patch(facecolor=COLORS['extreme'], edgecolor='black', alpha=0.7,
              label=f'Extreme (≥ {THRESHOLDS["extreme"]:.2f})'),
        Patch(facecolor=COLORS['severe'], edgecolor='black', alpha=0.7,
              label=f'Severe (≥ {THRESHOLDS["severe"]:.2f})'),
        Patch(facecolor=COLORS['moderate'], edgecolor='black', alpha=0.7,
              label=f'Moderate (≥ {THRESHOLDS["moderate"]:.2f})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=26, title='Hotspot Categories', title_fontsize=28)

    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    plt.close()
    return fig

# Dummy monthly stats plotting functions (replace with real ones if needed)
def monthly_analysis(df):
    return df

def plot_monthly_trends(df):
    pass

def plot_aod_distributions(df):
    pass

# ---------------- Main Function ----------------
def main():
    data_dir = r"C:\Users\admin\rohan\OUTPUT\2025\APRIL"
    shapefile_path = r"C:\Users\admin\rohan\IndiaStateDistShapeFile\STATE_BDY.shp"
    states = ['UTTARPRADESH', 'BIHAR', 'PUNJAB', 'HARYANA', 'DELHI']

    df = load_and_process_csv_files(data_dir)
    categorized_df = categorize_aod_values(df)
    grid_df = handle_overlapping_points(categorized_df)
    filtered_df = filter_by_states(grid_df, shapefile_path, states)
    monthly_stats = monthly_analysis(categorized_df)

    plot_hotspots(filtered_df, shapefile_path, states)
    plot_monthly_trends(monthly_stats)
    plot_aod_distributions(filtered_df)

if __name__ == "__main__":
    main()

