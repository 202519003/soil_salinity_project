# 08_rf_xgb.py
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

features = pd.read_csv('data/processed/features_complete.csv')
features = features.sort_values(['district','year']).reset_index(drop=True)

FEATURE_COLS = [
    'ndvi', 'rainfall_annual', 'temp_rabi',
    'ndvi_deficit', 'rain_deficit', 'temp_anomaly',
    'sspi_lag1', 'ndvi_lag1', 'ndvi_trend_slope'
]

features = features.dropna(subset=FEATURE_COLS + ['sspi']).reset_index(drop=True)
train = features[features['year'] <= 2021]
test  = features[features['year'] >= 2022]

print(f'Train SSPI - mean:{train["sspi"].mean():.1f} std:{train["sspi"].std():.1f}')
print(f'Test  SSPI - mean:{test["sspi"].mean():.1f} std:{test["sspi"].std():.1f}')

sc   = StandardScaler()
X_tr = sc.fit_transform(train[FEATURE_COLS])
X_te = sc.transform(test[FEATURE_COLS])
y_tr = train['sspi']
y_te = test['sspi']

# ── RANDOM FOREST — one lag feature, moderate regularisation ─────────────────
print('\nTraining Random Forest...')
rf = RandomForestRegressor(
    n_estimators=300, max_depth=7,
    min_samples_split=4, min_samples_leaf=2,
    max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_tr, y_tr)
rf_pred = np.clip(rf.predict(X_te), 0, 100)

# ── XGBOOST — moderate regularisation ────────────────────────────────────────
print('Training XGBoost...')
xgb = XGBRegressor(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, subsample=0.75,
    colsample_bytree=0.75, reg_alpha=0.3,
    reg_lambda=1.0, random_state=42, verbosity=0
)
xgb.fit(X_tr, y_tr)
xgb_pred = np.clip(xgb.predict(X_te), 0, 100)

def eval_model(name, yt, yp):
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2   = r2_score(yt, yp)
    print(f'{name:25s}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f} ({r2*100:.1f}%)')
    return {'model': name, 'MAE': round(mae,2), 'RMSE': round(rmse,2), 'R2': round(r2,3)}

print('\n--- MODEL COMPARISON ---')
results = [
    eval_model('Random Forest', y_te, rf_pred),
    eval_model('XGBoost',       y_te, xgb_pred)
]

# ── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
fi.plot(kind='barh', ax=ax, color='#2E75B6')
ax.set_title('Random Forest Feature Importance\n(Attention Weight Analogue)', fontsize=13)
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('outputs/charts/01_feature_importance.png', dpi=150)
plt.close()
print('\nFeature importance chart saved!')

# ── SAVE ─────────────────────────────────────────────────────────────────────
pickle.dump(rf,  open('data/processed/rf_model.pkl',  'wb'))
pickle.dump(xgb, open('data/processed/xgb_model.pkl', 'wb'))
pickle.dump(sc,  open('data/processed/feature_scaler.pkl', 'wb'))

existing = pd.read_csv('outputs/tables/model_comparison.csv')
existing = existing[~existing['model'].isin(['Random Forest','XGBoost'])]
final = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
final.to_csv('outputs/tables/model_comparison.csv', index=False)
print(final.to_string())

print('\n✅ Step 14 (RF + XGBoost) complete!')