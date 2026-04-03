import pandas as pd
import numpy as np
import torch
import sqlite3
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Use lightning (not pytorch_lightning) — required for v1.6.1 ──────────────
import lightning.pytorch as pl

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Load & Prepare ───────────────────────────────────────────────────────────
print("Loading features...")
features = pd.read_csv('data/processed/features_complete.csv')
features = features.sort_values(['district', 'year']).reset_index(drop=True)
features = features.dropna(subset=['sspi_lag1', 'sspi_lag2'])

# Required columns check
required = ['district', 'year', 'sspi', 'zone_type',
            'ndvi', 'rainfall_annual', 'temp_rabi',
            'ndvi_deficit', 'rain_deficit', 'temp_anomaly',
            'sspi_lag1', 'sspi_lag2']
missing = [c for c in required if c not in features.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    print(f"   Available: {list(features.columns)}")
    exit(1)



# time_idx: sequential index per district (0,1,2,...,N)
features['time_idx'] = features.groupby('district').cumcount()

# Ensure categoricals are strings (required by pytorch-forecasting)
features['district']  = features['district'].astype(str)
features['zone_type'] = features['zone_type'].astype(str)

print(f"Data shape: {features.shape}")
print(f"Districts: {features['district'].nunique()}")
print(f"Years: {sorted(features['year'].unique())}")
print(f"time_idx range: 0 – {features['time_idx'].max()}")

# ─── TFT Dataset Parameters ───────────────────────────────────────────────────
max_encoder_length    = 6   # use 6 past years to predict
max_prediction_length = 3   # predict next 3 years

training_cutoff = features['time_idx'].max() - max_prediction_length

print(f"\nTraining cutoff time_idx: {training_cutoff}")
print(f"Train rows: {(features['time_idx'] <= training_cutoff).sum()}")
print(f"Val rows:   {(features['time_idx'] > training_cutoff).sum()}")

# ─── Build Datasets ───────────────────────────────────────────────────────────
training = TimeSeriesDataSet(
    features[features['time_idx'] <= training_cutoff],
    time_idx                   = 'time_idx',
    target                     = 'sspi',
    group_ids                  = ['district'],
    static_categoricals        = ['district', 'zone_type'],
    time_varying_known_reals   = [
        'ndvi', 'rainfall_annual', 'temp_rabi',
        'ndvi_deficit', 'rain_deficit', 'temp_anomaly',
    ],
    time_varying_unknown_reals = ['sspi'],
    max_encoder_length         = max_encoder_length,
    max_prediction_length      = max_prediction_length,
    min_encoder_length         = 3,
    add_relative_time_idx      = True,
    add_target_scales          = True,
    add_encoder_length         = True,
)

validation = TimeSeriesDataSet.from_dataset(
    training, features, predict=True, stop_randomization=True
)

train_loader = training.to_dataloader(train=True,  batch_size=16, num_workers=0)
val_loader   = validation.to_dataloader(train=False, batch_size=16, num_workers=0)

# ─── Build TFT Model ──────────────────────────────────────────────────────────
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate              = 0.03,
    hidden_size                = 32,
    attention_head_size        = 2,
    dropout                    = 0.1,
    hidden_continuous_size     = 16,
    loss                       = QuantileLoss(),
    log_interval               = 10,
    reduce_on_plateau_patience = 4,
)

print(f"\nTFT parameters: {sum(p.numel() for p in tft.parameters()):,}")

# ─── Train ────────────────────────────────────────────────────────────────────
# Use the version-safe Trainer — works with both pytorch-lightning <2 and >=2
trainer = pl.Trainer(
    max_epochs           = 30,
    enable_progress_bar  = True,
    gradient_clip_val    = 0.1,
    logger               = False,
    enable_checkpointing = False,
)

print("\nTraining TFT... (30 epochs)")
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

# ─── Evaluate ─────────────────────────────────────────────────────────────────
print("\nEvaluating...")
raw_predictions = tft.predict(val_loader, mode='prediction', return_x=False)
actuals         = torch.cat([y[0] for x, y in iter(val_loader)])

tft_pred = np.clip(raw_predictions.numpy().flatten(), 0, 100)
y_te     = actuals.numpy().flatten()

mae  = mean_absolute_error(y_te, tft_pred)
rmse = np.sqrt(mean_squared_error(y_te, tft_pred))
r2   = r2_score(y_te, tft_pred)

print(f"\n{'='*40}")
print(f"  TFT MODEL RESULTS")
print(f"{'='*40}")
print(f"  MAE  : {mae:.2f}  (avg prediction error in SSPI points)")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.3f}  ({r2*100:.1f}%)")
print(f"{'='*40}\n")

# ─── Save Model Checkpoint ────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
trainer.save_checkpoint('data/processed/tft_model.ckpt')
print("Model saved → data/processed/tft_model.ckpt")

# ─── Update Model Comparison Table ───────────────────────────────────────────
tft_row = pd.DataFrame([{
    'model': 'TFT Transformer',
    'mae':   round(mae,  2),
    'rmse':  round(rmse, 2),
    'r2':    round(r2,   3),
    'mape':  round(float(np.mean(np.abs((y_te - tft_pred) / (y_te + 1e-6)))) * 100, 1),
}])

csv_path = 'outputs/tables/model_comparison.csv'
os.makedirs('outputs/tables', exist_ok=True)

if os.path.exists(csv_path):
    existing = pd.read_csv(csv_path)
    existing = existing[existing['model'] != 'TFT Transformer']
    final = pd.concat([existing, tft_row], ignore_index=True)
else:
    final = tft_row

final.to_csv(csv_path, index=False)
print(f"Model comparison saved → {csv_path}")

# ─── Update SQLite database ───────────────────────────────────────────────────
db_path = 'data/processed/salinity_db.sqlite'
conn = sqlite3.connect(db_path)

try:
    existing_db = pd.read_sql("SELECT * FROM model_metrics", conn)
    existing_db = existing_db[existing_db['model'] != 'TFT Transformer']
    final_db = pd.concat([existing_db, tft_row], ignore_index=True)
except Exception:
    final_db = tft_row

# Drop and recreate to avoid duplicate column errors from schema drift
conn.execute("DROP TABLE IF EXISTS model_metrics")
conn.commit()
final_db.to_sql('model_metrics', conn, if_exists='replace', index=False)
conn.close()
print(f"Database updated → {db_path}")

print("\nFinal model comparison:")
print(final_db.to_string(index=False))
print("\n✅ TFT training complete!")