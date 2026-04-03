import pandas as pd
import numpy as np
import pickle
import sqlite3
import os

print("Loading models and features...")

rf      = pickle.load(open('data/processed/rf_model.pkl', 'rb'))
scaler  = pickle.load(open('data/processed/feature_scaler.pkl', 'rb'))
features = pd.read_csv('data/processed/features_complete.csv')

# Must match 06_ml_prep.py and 08_rf_xgb.py exactly
FEATURE_COLS = [
    'ndvi', 'rainfall_annual', 'temp_rabi',
    'ndvi_deficit', 'rain_deficit', 'temp_anomaly',
    'sspi_lag1', 'ndvi_lag1', 'ndvi_trend_slope'
]

features = features.dropna(subset=FEATURE_COLS + ['sspi']).reset_index(drop=True)

print(f"Districts: {features['district'].nunique()}")
print(f"Historical years: {sorted(features['year'].unique())}")
print("Generating 2026-2030 forecast for all Gujarat districts...")

predictions = []

for district in sorted(features['district'].unique()):
    dist_data = features[features['district'] == district].sort_values('year')

    future_rain  = dist_data['rainfall_annual'].mean()
    future_temp  = dist_data['temp_rabi'].mean()
    future_ndvi  = dist_data['ndvi'].iloc[-1]
    trend_slope  = dist_data['ndvi_trend_slope'].iloc[-1]

    ndvi_mean    = dist_data['ndvi'].mean()
    rain_mean    = dist_data['rainfall_annual'].mean()
    temp_mean    = dist_data['temp_rabi'].mean()

    lag1_sspi    = dist_data['sspi'].iloc[-1]
    lag1_ndvi    = dist_data['ndvi'].iloc[-1]

    for yr in range(2026, 2031):
        proj_ndvi    = max(0, future_ndvi + trend_slope)
        ndvi_deficit = max(0, ndvi_mean - proj_ndvi)
        rain_deficit = max(0, rain_mean - future_rain)
        temp_anomaly = max(0, future_temp - temp_mean)

        # 9 features: ndvi, rainfall_annual, temp_rabi, ndvi_deficit,
        #             rain_deficit, temp_anomaly, sspi_lag1, ndvi_lag1, ndvi_trend_slope
        row = [
            proj_ndvi, future_rain, future_temp,
            ndvi_deficit, rain_deficit, temp_anomaly,
            lag1_sspi, lag1_ndvi, trend_slope
        ]

        X = scaler.transform([row])
        pred_sspi = float(rf.predict(X)[0])
        pred_sspi = max(0, min(100, round(pred_sspi, 1)))

        predictions.append({
            'district':       district,
            'year':           yr,
            'predicted_sspi': pred_sspi
        })

        # Autoregressive: feed prediction back
        lag1_sspi   = pred_sspi
        lag1_ndvi   = proj_ndvi
        future_ndvi = proj_ndvi

pred_df = pd.DataFrame(predictions)

def classify_trend(x):
    if x.iloc[-1] > x.iloc[0] + 3:
        return 'Worsening'
    elif x.iloc[-1] < x.iloc[0] - 3:
        return 'Improving'
    else:
        return 'Stable'

pred_df['trend'] = pred_df.groupby('district')['predicted_sspi'].transform(classify_trend)

print("\n2026-2030 SALINITY FORECAST:")
print(pred_df.to_string(index=False))

print("\nTrend summary:")
print(pred_df.drop_duplicates('district')[['district','trend']]
      .groupby('trend')['district'].count().to_string())

os.makedirs('outputs/tables', exist_ok=True)
pred_df.to_csv('outputs/tables/salinity_forecast_2026_2030.csv', index=False)
print("\nForecast saved -> outputs/tables/salinity_forecast_2026_2030.csv")

db_path = 'data/processed/salinity_db.sqlite'
conn = sqlite3.connect(db_path)
conn.execute("DROP TABLE IF EXISTS salinity_forecast")
conn.commit()
pred_df.to_sql('salinity_forecast', conn, if_exists='replace', index=False)
conn.close()

print("\n✅ Step 15 complete!")
print("salinity_forecast_2026_2030.csv saved to outputs/tables/")