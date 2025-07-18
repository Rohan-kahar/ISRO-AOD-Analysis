#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
test_file_path = "/content/KendriyaVidyalaya_Lucknow-CPCB˜˜_mean_aod_collocated.csv"  # <-- Replace with your test file path

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

