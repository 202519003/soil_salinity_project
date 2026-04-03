# 06_ml_prep.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

features = pd.read_csv('data/processed/features_complete.csv')
features = features.sort_values(['district', 'year']).reset_index(drop=True)

FEATURE_COLS = [
    'ndvi', 'rainfall_annual', 'temp_rabi',
    'ndvi_deficit', 'rain_deficit', 'temp_anomaly',
    'sspi_lag1', 'ndvi_lag1', 'ndvi_trend_slope'
]

TARGET = 'sspi'

features = features.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)

# Stricter temporal split
train = features[features['year'] <= 2021]
test  = features[features['year'] >= 2022]

X_train = train[FEATURE_COLS]
y_train = train[TARGET]
X_test  = test[FEATURE_COLS]
y_test  = test[TARGET]

print(f'Train size: {len(X_train)} rows  (2015-2021)')
print(f'Test size:  {len(X_test)} rows   (2022-2025)')
print(f'Train SSPI - mean:{y_train.mean():.1f} std:{y_train.std():.1f}')
print(f'Test  SSPI - mean:{y_test.mean():.1f} std:{y_test.std():.1f}')
print(f'Features: {FEATURE_COLS}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

with open('data/processed/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('data/processed/X_train.pkl', 'wb') as f:
    pickle.dump(X_train_scaled, f)
with open('data/processed/X_test.pkl', 'wb') as f:
    pickle.dump(X_test_scaled, f)
with open('data/processed/y_train.pkl', 'wb') as f:
    pickle.dump(y_train.values, f)
with open('data/processed/y_test.pkl', 'wb') as f:
    pickle.dump(y_test.values, f)

print('✅ Step 12 complete! Training data and scaler ready.')