#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from glob import glob
import numpy as np
from matplotlib.lines import Line2D

# === Static Settings ===
color_map = {
    "Moderate": "blue",
    "Severe": "yellow",
    "Extreme": "red"
}

target_states = ["PUNJAB", "HARYANA", "DELHI", "UTTARPRADESH", "BIHAR"]

# === Paths ===
csv_folder = r'C:\Users\admin\rohan\DATA\OUTPUT\2025\APRIL'  # Input folder
output_folder = r'C:\Users\admin\rohan\DATA\OUTPUT\PLOTS'  # Output folder
state_shapefile = r'C:\Users\admin\rohan\IndiaStateDistShapeFile\STATE_BDY.shp'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === Load state shapefile ===
states = gpd.read_file(state_shapefile)
selected_states = states[states["STATE"].isin(target_states)]

# === Process all CSV files ===
csv_files = glob(os.path.join(csv_folder, '*.csv'))

for csv_path in csv_files:
    filename = os.path.splitext(os.path.basename(csv_path))[0]  # '20231130'

    try:
        # --- Load data ---
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["AOD", "latitude", "longitude"])

        # --- Compute PDF-based thresholds ---
        aod_values = df["AOD"].values
        aod_values = aod_values[~np.isnan(aod_values)]

        if len(aod_values) < 10:
            print(f"Skipping {filename} - not enough valid AOD values.")
            continue

        p90 = np.percentile(aod_values, 90)
        p95 = np.percentile(aod_values, 95)
        p99 = np.percentile(aod_values, 99)

        # --- Classify AOD levels ---
        def classify_aod(aod):
            if aod >= p99:
                return "Extreme"
            elif aod >= p95:
                return "Severe"
            elif aod >= p90:
                return "Moderate"
            else:
                return None

        df["HotspotLevel"] = df["AOD"].apply(classify_aod)
        df = df.dropna(subset=["HotspotLevel"])

        # If no points after classification, skip
        if df.empty:
            print(f"Skipping {filename} - no data above 90th percentile.")
            continue

        # --- Create GeoDataFrame ---
        geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # --- Clip to target states ---
        gdf_clipped = gpd.clip(gdf, selected_states)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(24, 26))

        # Only show selected state boundaries
        selected_states.boundary.plot(ax=ax, linewidth=1.5, edgecolor='black')

        for level, color in color_map.items():
            subset = gdf_clipped[gdf_clipped["HotspotLevel"] == level]
            subset.plot(ax=ax, markersize=2.5, color=color, label=level, alpha=0.4)

        plt.title(f"AOD Hotspot Map - {filename}", fontsize=34)

        # Custom legend with larger markers
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Moderate', markerfacecolor='blue', markersize=18),
            Line2D([0], [0], marker='o', color='w', label='Severe', markerfacecolor='yellow', markersize=18),
            Line2D([0], [0], marker='o', color='w', label='Extreme', markerfacecolor='red', markersize=18)
        ]
        ax.legend(handles=legend_elements, title="Hotspot Level", loc='upper right', fontsize=26, title_fontsize=28)

        # Annotate percentiles in bottom-left corner
        ax.text(0.02, 0.02,
                f"Percentile Values\n90th: {p90:.2f}\n95th: {p95:.2f}\n99th: {p99:.2f}",
                transform=ax.transAxes,
                fontsize=28,
                verticalalignment='bottom',
                bbox=dict(facecolor='white', edgecolor='black'))

        ax.set_xlabel('Longitude', fontsize=30)
        ax.set_ylabel('Latitude', fontsize=30)
        ax.tick_params(axis='both', labelsize=28)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plt.xlim([73.5, 88.5])
        plt.ylim([23.5, 32.5])
        plt.tight_layout()

        # --- Save figure to output folder ---
        output_path = os.path.join(output_folder, f"{filename}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

