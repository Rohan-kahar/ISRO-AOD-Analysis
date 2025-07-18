# ISRO-AOD-Analysis
# 🛰️ Characterization of aersol hotspots in Indo Gangetic Plains using EOS-06 OCM data
This repository contains all the datasets, preprocessing scripts, statistical and spatial visualizations, and machine learning models developed during my six-month internship at **Space Applications Centre, ISRO (SAC Ahmedabad)**.

The main objective was to **analyze Aerosol Optical Depth (AOD)** over the **Indo-Gangetic Plain** using **satellite-based L2 and L3 data**, identify pollution hotspots, and build machine learning models to predict **PM2.5 concentrations** from AOD and other spatio-temporal features.

---

## 📌 Project Objectives

- Subset and process L2 AOD data from MOSDAC (Oct 2024 – Feb 2025).
- Merge, clean, and analyze AOD values over specific Indian states (UP, Bihar, Delhi, Punjab, Haryana).
- Perform monthly statistical and spatial analysis.
- Identify high AOD hotspots via clustering methods (K-Means, Bayesian GMM).
- Predict PM2.5 using Stacking Regression with features such as AOD, latitude, longitude, and time (day/month/year).

---

## 🛰️ Data Sources

| Dataset | Description |
|--------|-------------|
| **L2 AOD** | Scenewise Aerosol Optical Depth over Land from MOSDAC (Oct 2024 – Feb 2025) |
| **L3 AOD** | Monthly gridded AOD from MOSDAC (Oct 2023 – Feb 2025, monsoon months excluded) |
| **PM2.5 Data** | Ground truth PM2.5 from CPCB (station-wise CSVs) |
| **Geo-Data** | Shapefiles for India state boundaries used for filtering and mapping |

---

## 📁 Folder Structure

---

## 📊 Key Analyses & Techniques

- **Spatial Filtering**: Subset AOD values only over the Indo-Gangetic Plain using shapely and geopandas.
- **Monthly Aggregation**: Compute mean, max, min, and standard deviation for each month.
- **Hotspot Identification**: Use K-Means and Bayesian Gaussian Mixture Models (GMM) for clustering AOD values into severity bins.
- **Classification Bins**: AOD values were grouped into:
  - ≤ 2.0 → Moderate
  - 2.0–3.0 → Severe
  - > 3.0  → Extreme

---

## 🤖 Stacking Regression Model for PM2.5 Prediction

We implemented a **Stacked Regressor Model** to estimate PM2.5 concentration at a location using AOD and other features (lat, lon, day, month, year). The aim was **spatial generalization** — predicting PM2.5 at a different location `B` using AOD at location `A`.

### 📌 Features Used

- **AOD**: Most dominant feature
- **Latitude / Longitude**
- **Year / Month / Day**

### ✅ Preprocessing Steps

- Removed rows with null PM2.5 or AOD
- Outlier handling using quantile-based clipping
- Scaled features using `RobustScaler` to avoid impact of outliers

### 🧠 Model Architecture

We used a **stacked ensemble** model combining three base regressors and one final meta-regressor.

#### Base Models:

- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `XGBRegressor`

#### Final Estimator:

- `Ridge Regression` or `SVR` (RBF kernel)

#### Stack Implementation:

```python
from sklearn.ensemble import StackingRegressor

base_models = [
    ('rf', RandomForestRegressor(n_estimators=100)),
    ('et', ExtraTreesRegressor(n_estimators=100)),
    ('xgb', xgb.XGBRegressor(n_estimators=100))
]

meta_model = Ridge()

stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=True
)






