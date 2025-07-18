#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from collections import Counter
import math

# --- Your data path ---
root_dir = r"C:\Users\admin\rohan\DATA\data"

# --- Bihar bounding box ---
min_lat, max_lat = 24.30, 27.50
min_lon, max_lon = 38.00, 88.10

# --- List of target coordinates (lat, lon) ---
target_coords_list = [
    (24.75746, 84.366208), 
    (24.762518, 84.982348), 
    (24.792403, 84.992416), 
    (24.7955, 84.9994),
    (25.03280, 85.41948), 
    (25.204762, 85.51496), 
    (25.42742023, 86.13886079), 
    (25.376776, 86.471523),
    (25.251013, 86.989001), 
    (25.265194, 87.012947), 
    (25.366336, 87.117468), 
    (25.560083, 87.553265),
    (25.892357, 86.590325), 
    (26.146529, 87.454184), 
    (26.0881305, 87.93840336), 
    (26.11442, 85.39813),
    (26.1403345, 85.3650192), 
    (26.1209, 85.3647), 
    (26.63086, 84.90051), 
    (26.80365, 84.51954),
    (27.308328, 84.531742), 
    (26.2271665, 84.3570427), 
    (25.56752, 83.966379), 
    (24.952822, 84.002396),
    (25.5626095, 84.663264), 
    (25.7808257, 84.7446768), 
    (25.697189, 85.2459), 
    (25.586562, 85.043586),
    (25.596727, 85.085624), 
    (25.599486, 85.113666), 
    (25.610369, 85.132568), 
    (25.619651, 85.147382),
    (25.592539, 85.227158), 
    (25.859655, 85.77944)
]

# --- Count occurrences of lat-lon ---
coord_counter = Counter()

# --- First pass to count coordinates within bounding box ---
for folder_name in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                df = pd.read_csv(file_path)
                df = df[
                    (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat) &
                    (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
                ]
                lat_lon_pairs = zip(df['latitude'].round(4), df['longitude'].round(4))
                coord_counter.update(lat_lon_pairs)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# --- Function to compute Euclidean distance between two lat-lon points ---
def distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

# --- For each target coordinate ---
for target_lat, target_lon in target_coords_list:
    target_lat = round(target_lat, 4)
    target_lon = round(target_lon, 4)

    # Find matching coordinates within ±0.01° range
    nearby_coords = {
        coord: count
        for coord, count in coord_counter.items()
        if abs(coord[0] - target_lat) <= 0.01 and abs(coord[1] - target_lon) <= 0.01
    }

    if not nearby_coords:
        print(f"❌ No coordinates found near target ({target_lat}, {target_lon})")
        continue

    # Sort by frequency first, then distance (closest most frequent first)
    sorted_coords = sorted(
        nearby_coords.items(),
        key=lambda x: (-x[1], distance(x[0][0], x[0][1], target_lat, target_lon))
    )[:10]

    print(f"\n✅ Top coordinates near target ({target_lat}, {target_lon}):")
    for i, (coord, count) in enumerate(sorted_coords, 1):
        print(f"{i}. Coordinate: ({coord[0]}, {coord[1]}) - Count: {count}")

    # --- Second pass to extract and save rows for top N coordinates ---
    for rank, (coord, count) in enumerate(sorted_coords, 1):
        lat, lon = coord
        print(f"\n--- Extracting rows for #{rank} coordinate: ({lat}, {lon}) ---")
        matched_rows = []

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for file_name in sorted(os.listdir(folder_path)):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        df = pd.read_csv(file_path)
                        df['latitude'] = df['latitude'].round(4)
                        df['longitude'] = df['longitude'].round(4)

                        mask = (df['latitude'] == lat) & (df['longitude'] == lon)
                        if mask.any():
                            matched = df.loc[mask, ['date', 'latitude', 'longitude', 'AOD']]
                            matched_rows.append(matched)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        if matched_rows:
            result_df = pd.concat(matched_rows, ignore_index=True)
            filename = f"{rank}_{target_lat}_{target_lon}.csv"
            result_df.to_csv(filename, index=False)
            print(f"Saved to '{filename}' with {len(result_df)} rows")
        else:
            print(f"No matching rows found for ({lat}, {lon})")

