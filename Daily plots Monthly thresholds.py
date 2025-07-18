#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from glob import glob

# === Static Settings ===
p90 = 0.8688
p95 = 1.2732
p99 = 2.3862

color_map = {
    "Moderate": "blue",
    "Severe": "yellow",
    "Extreme": "red"
}

target_states = ["PUNJAB", "HARYANA", "DELHI", "UTTARPRADESH", "BIHAR"]

# === Paths ===
csv_folder = r'C:\Users\admin\rohan\DATA\OUTPUT\2024\APRIL'
state_shapefile = r'C:\Users\admin\rohan\IndiaStateDistShapeFile\STATE_BDY.shp'

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
        states.boundary.plot(ax=ax, linewidth=1.5, edgecolor='grey')
        selected_states.boundary.plot(ax=ax, linewidth=1.5, edgecolor='black')

        for level, color in color_map.items():
            subset = gdf_clipped[gdf_clipped["HotspotLevel"] == level]
            subset.plot(ax=ax, markersize=1.5, color=color, label=level, alpha=0.4)

        plt.title(f"AOD Hotspot Map - {filename}", fontsize=20)
        plt.legend(title="Hotspot Level", loc='upper right')

        # Annotate percentiles
        ax.text(0.02, 0.98,
                f"Percentile Values\n90th: {p90:.2f}\n95th: {p95:.2f}\n99th: {p99:.2f}",
                transform=ax.transAxes,
                fontsize=15,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black'))

        ax.set_xlabel('Longitude', fontsize=20)
        ax.set_ylabel('Latitude', fontsize=20)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlim([73.5, 88.5])
        plt.ylim([23.5, 32.5])
        plt.tight_layout()

        # --- Save figure ---
        output_path = os.path.join(csv_folder, f"{filename}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

