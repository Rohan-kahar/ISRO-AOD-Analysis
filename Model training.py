#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
import os

# === Load Data ===
df = pd.read_csv("/content/cleaned_file.csv")  # <-- Replace with your file path

# === Clean and Filter ===
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df[~((df['date'].dt.year == 2024) & (df['date'].dt.month.isin([6, 7, 8, 9])))]

# === Enhanced Feature Engineering ===
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['year'] = df['date'].dt.year
df['day_of_year'] = df['date'].dt.dayofyear

# Enhanced Season mapping including Pre-Monsoon
def map_season(month):
    if month in [10, 11]: return 'Post-Monsoon'
    elif month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Pre-Monsoon'
    elif month in [6, 7, 8, 9]: return 'Summer'
    else: return 'Other'

df['season'] = df['month'].apply(map_season)

# Enhanced AOD Transformations
df['aod_log'] = np.log1p(df['AOD'])
df['aod_sqrt'] = np.sqrt(df['AOD'])
df['aod_sq'] = df['AOD'] ** 2
df['aod_cube'] = df['AOD'] ** 3
df['aod_inv'] = 1 / (df['AOD'] + 1e-5)

# Interaction features
df['aod_month'] = df['AOD'] * df['month']
df['aod_lat'] = df['AOD'] * df['latitude']
df['aod_lon'] = df['AOD'] * df['longitude']

# Seasonal encoding
season_encoded = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, season_encoded], axis=1)

# Enhanced Features & Target
features = ['AOD', 'aod_log', 'aod_sqrt', 'aod_sq', 'aod_cube', 'aod_inv',
            'latitude', 'longitude', 'month', 'day', 'year', 'day_of_year',
            'aod_month', 'aod_lat', 'aod_lon'] + list(season_encoded.columns)
target = 'PM2.5'

X = df[features]
y = df[target]

# === Train-Test-Validation Split (70/20/10) ===
X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, test_size=2/9, random_state=42)

# === Enhanced Model Architecture ===
base_models = [
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                                min_samples_leaf=2, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=200, max_depth=15, min_samples_split=5,
                              min_samples_leaf=2, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=8,
                                   min_samples_split=5, random_state=42)),
    ('xgb', xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=8,
                            min_child_weight=3, random_state=42, verbosity=0)),
    ('lgb', lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, max_depth=8,
                             min_child_samples=5, random_state=42, verbosity=-1))
]

final_estimator = ElasticNet(alpha=0.1, l1_ratio=0.5)

scalers = {
    'StandardScaler': StandardScaler(),
    'RobustScaler': RobustScaler(),
    'QuantileTransformer': QuantileTransformer(output_distribution='normal', random_state=42)
}

results = []
os.makedirs("models", exist_ok=True)

# === Model Training & Evaluation ===
print("Training enhanced models...")
for scaler_name, scaler in scalers.items():
    print(f"Training with {scaler_name}...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stack_model = StackingRegressor(estimators=base_models, final_estimator=final_estimator, cv=5)
    stack_model.fit(X_train_scaled, y_train)
    y_pred = stack_model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        'Scaler': scaler_name,
        'RMSE (%)': rmse / y_test.mean() * 100,
        'MAE (%)': mae / y_test.mean() * 100,
        'R2': r2
    })

    # Save model
    joblib.dump(stack_model, f"models/stack_model_{scaler_name}.pkl")
    joblib.dump(scaler, f"models/scaler_{scaler_name}.pkl")

# === Results Table ===
results_df = pd.DataFrame(results)
print("\nModel Evaluation Summary:")
print(results_df.sort_values(by='RMSE (%)').round(4))

# === Plot: Actual vs Predicted (Best Model) ===
best_scaler = results_df.sort_values(by='RMSE (%)').iloc[0]['Scaler']
best_model = joblib.load(f"models/stack_model_{best_scaler}.pkl")
best_scaler_obj = joblib.load(f"models/scaler_{best_scaler}.pkl")
X_val_scaled = best_scaler_obj.transform(X_val)
y_val_pred = best_model.predict(X_val_scaled)

# Calculate R² and correlation
r2_val = r2_score(y_val, y_val_pred)
corr_val, _ = pearsonr(y_val, y_val_pred)

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_val_pred, alpha=0.6, label='Prediction', s=20)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Ideal', linewidth=2)
plt.xlabel('Actual PM 2.5')
plt.ylabel('Predicted PM 2.5')
plt.title(f'Actual vs Predicted PM 2.5 using {best_scaler}\nR² = {r2_val:.4f}, Correlation = {corr_val:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Seasonal Plots ===
df_val = df.iloc[X_val.index].copy()
df_val['y_val_pred'] = y_val_pred
df_val['y_val_actual'] = y_val.values

seasons = ['Winter', 'Pre-Monsoon', 'Summer', 'Post-Monsoon']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, season in enumerate(seasons):
    season_data = df_val[df_val['season'] == season]

    if len(season_data) > 0:
        r2_season = r2_score(season_data['y_val_actual'], season_data['y_val_pred'])
        corr_season, _ = pearsonr(season_data['y_val_actual'], season_data['y_val_pred'])

        axes[i].scatter(season_data['y_val_actual'], season_data['y_val_pred'],
                       alpha=0.6, s=20, label='Prediction')
        axes[i].plot([season_data['y_val_actual'].min(), season_data['y_val_actual'].max()],
                    [season_data['y_val_actual'].min(), season_data['y_val_actual'].max()],
                    'r--', label='Ideal', linewidth=2)
        axes[i].set_xlabel('Actual PM 2.5')
        axes[i].set_ylabel('Predicted PM 2.5')
        axes[i].set_title(f'{season}\nR² = {r2_season:.4f}, Correlation = {corr_season:.4f}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    else:
        axes[i].text(0.5, 0.5, f'No data for {season}', transform=axes[i].transAxes,
                    ha='center', va='center', fontsize=12)
        axes[i].set_title(f'{season} - No Data')

plt.tight_layout()
plt.show()

# === Seasonal AOD Boxplot ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='season', y='AOD', order=['Winter', 'Pre-Monsoon', 'Summer', 'Post-Monsoon'])
plt.title("AOD Distribution Across Seasons")
plt.xlabel("Season")
plt.ylabel("AOD")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === Feature Importance ===
feature_importance = best_model.named_estimators_['rf'].feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

print(f"\n✔ Enhanced model architecture implemented with {best_scaler} scaler.")
print(f"✔ RMSE: {results_df.sort_values(by='RMSE (%)').iloc[0]['RMSE (%)']:.4f}%")
print(f"✔ MAE: {results_df.sort_values(by='RMSE (%)').iloc[0]['MAE (%)']:.4f}%")
print(f"✔ R²: {results_df.sort_values(by='RMSE (%)').iloc[0]['R2']:.4f}")
print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# ============================= TESTING SECTION =============================
print("\n" + "="*60)
print("TESTING ON NEW DATA")
print("="*60)

# Function to prepare test data with same feature engineering
def prepare_test_data(test_df):
    """Apply same feature engineering as training data"""
    test_df = test_df.copy()
    
    # Convert date format
    test_df['date'] = pd.to_datetime(test_df['date'], format='%Y%m%d')
    
    # Create time-based features
    test_df['month'] = test_df['date'].dt.month
    test_df['day'] = test_df['date'].dt.day
    test_df['year'] = test_df['date'].dt.year
    test_df['day_of_year'] = test_df['date'].dt.dayofyear
    
    # Create season feature
    test_df['season'] = test_df['month'].apply(map_season)
    
    # AOD transformations
    test_df['aod_log'] = np.log1p(test_df['AOD'])
    test_df['aod_sqrt'] = np.sqrt(test_df['AOD'])
    test_df['aod_sq'] = test_df['AOD'] ** 2
    test_df['aod_cube'] = test_df['AOD'] ** 3
    test_df['aod_inv'] = 1 / (test_df['AOD'] + 1e-5)
    
    # Interaction features
    test_df['aod_month'] = test_df['AOD'] * test_df['month']
    test_df['aod_lat'] = test_df['AOD'] * test_df['latitude']
    test_df['aod_lon'] = test_df['AOD'] * test_df['longitude']
    
    # Seasonal encoding
    season_encoded_test = pd.get_dummies(test_df['season'], prefix='season')
    
    # Ensure all season columns are present (in case test data doesn't have all seasons)
    for col in season_encoded.columns:
        if col not in season_encoded_test.columns:
            season_encoded_test[col] = 0
    
    # Reorder columns to match training data
    season_encoded_test = season_encoded_test.reindex(columns=season_encoded.columns, fill_value=0)
    
    test_df = pd.concat([test_df, season_encoded_test], axis=1)
    
    return test_df

# Load and prepare test data
test_file_path = "/content/IITK_Kanpur-IITK˜˜_mean_aod_collocated.csv"  # <-- Replace with your test file path

try:
    # Load test data
    test_df = pd.read_csv(test_file_path)
    print(f"✔ Test data loaded successfully. Shape: {test_df.shape}")
    print(f"✔ Test data columns: {list(test_df.columns)}")
    
    # Prepare test data
    test_df_prepared = prepare_test_data(test_df)
    
    # Extract features and target
    X_test_new = test_df_prepared[features]
    y_test_new = test_df_prepared['PM2.5']
    
    # Scale test data using the best scaler
    X_test_new_scaled = best_scaler_obj.transform(X_test_new)
    
    # Make predictions
    y_pred_new = best_model.predict(X_test_new_scaled)
    
    # Calculate metrics
    rmse_test = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
    mae_test = mean_absolute_error(y_test_new, y_pred_new)
    r2_test = r2_score(y_test_new, y_pred_new)
    corr_test, _ = pearsonr(y_test_new, y_pred_new)
    
    print(f"\n✔ Test Data Evaluation Results:")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  RMSE (%): {rmse_test / y_test_new.mean() * 100:.4f}%")
    print(f"  MAE: {mae_test:.4f}")
    print(f"  MAE (%): {mae_test / y_test_new.mean() * 100:.4f}%")
    print(f"  R²: {r2_test:.4f}")
    print(f"  Correlation: {corr_test:.4f}")
    
    # Create results dataframe
    results_test_df = pd.DataFrame({
        'Date': test_df_prepared['date'],
        'Latitude': test_df_prepared['latitude'],
        'Longitude': test_df_prepared['longitude'],
        'AOD': test_df_prepared['AOD'],
        'Actual_PM2.5': y_test_new,
        'Predicted_PM2.5': y_pred_new,
        'Error': y_test_new - y_pred_new,
        'Absolute_Error': np.abs(y_test_new - y_pred_new),
        'Season': test_df_prepared['season']
    })
    
    print(f"\n✔ Sample predictions (first 10 rows):")
    print(results_test_df[['Date', 'Actual_PM2.5', 'Predicted_PM2.5', 'Error', 'Season']].head(10))
    
    # Save results
    results_test_df.to_csv("test_results.csv", index=False)
    print(f"\n✔ Test results saved to 'test_results.csv'")
    
    # Plot: Actual vs Predicted for test data
    plt.figure(figsize=(12, 8))
    
    # Main scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_test_new, y_pred_new, alpha=0.6, s=20, color='blue')
    plt.plot([y_test_new.min(), y_test_new.max()], [y_test_new.min(), y_test_new.max()], 'r--', linewidth=2)
    plt.xlabel('Actual PM 2.5')
    plt.ylabel('Predicted PM 2.5')
    plt.title(f'Test Data: Actual vs Predicted PM 2.5\nR² = {r2_test:.4f}, Correlation = {corr_test:.4f}')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 2, 2)
    plt.hist(results_test_df['Error'], bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    
    # Time series plot
    plt.subplot(2, 2, 3)
    plt.plot(results_test_df['Date'], results_test_df['Actual_PM2.5'], 'b-', label='Actual', alpha=0.7)
    plt.plot(results_test_df['Date'], results_test_df['Predicted_PM2.5'], 'r-', label='Predicted', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('PM 2.5')
    plt.title('Time Series: Actual vs Predicted')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Seasonal performance
    plt.subplot(2, 2, 4)
    seasonal_rmse = results_test_df.groupby('Season')['Absolute_Error'].mean()
    seasonal_rmse.plot(kind='bar', color='orange', alpha=0.7)
    plt.xlabel('Season')
    plt.ylabel('Mean Absolute Error')
    plt.title('Seasonal Performance (MAE)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics by season
    print(f"\n✔ Seasonal Performance Summary:")
    seasonal_stats = results_test_df.groupby('Season').agg({
        'Actual_PM2.5': ['mean', 'std'],
        'Predicted_PM2.5': ['mean', 'std'],
        'Absolute_Error': ['mean', 'std'],
        'Error': ['mean', 'std']
    }).round(4)
    print(seasonal_stats)
    
except FileNotFoundError:
    print(f"Test file not found at {test_file_path}")
    print("Please ensure the test CSV file is available with columns: date, latitude, longitude, AOD, PM2.5")
    print("Date format should be YYYYMMDD")
    
except Exception as e:
    print(f"Error in testing: {str(e)}")
    print("Please check your test data format and file path.")

print("\n" + "="*60)
print("TESTING COMPLETED")
print("="*60)

