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

#### Few Outputs
<img width="1000" height="600" alt="actual_vs_predicted_pm25" src="https://github.com/user-attachments/assets/79fe354b-69cf-4d77-bfeb-0a92d3f3112a" />
<img width="4033" height="2973" alt="Indo_Gangetic_Plain" src="https://github.com/user-attachments/assets/53ecbf73-7da5-463b-9ca7-c8316f9d8d88" />
<img width="4033" height="2973" alt="Indo_Gangetic_Plain" src="https://github.com/user-attachments/assets/16637fee-ffc4-456e-9048-3865b1423c44" />
<img width="3570" height="1770" alt="IIT_Delhi" src="https://github.com/user-attachments/assets/80eed628-bf64-4a4a-8451-ca38362fd43f" />
<img width="478" height="340" alt="image" src="https://github.com/user-attachments/assets/7fff8007-4aa1-4576-a6b5-e4a04b49ec20" />
<img width="478" height="340" alt="image" src="https://github.com/user-attachments/assets/7c2913f1-3f6b-4524-9039-b8bd2fc98f3d" />






