import pandas as pd
import sqlite3
import os

print("Connecting to SQLite...")
os.makedirs('data/processed', exist_ok=True)
conn = sqlite3.connect('data/processed/salinity_db.sqlite')

# Load files
master   = pd.read_csv('data/processed/features_complete.csv')
forecast = pd.read_csv('outputs/tables/salinity_forecast_2026_2030.csv')

# ─── Rebuild model_comparison cleanly from scratch ───────────────────────────
# Read raw, handle duplicate columns from mixed script runs
raw    = pd.read_csv('outputs/tables/model_comparison.csv', header=None)
header = [str(h).strip().lower() for h in raw.iloc[0]]
data   = raw.iloc[1:].reset_index(drop=True)

clean_rows = []
for _, row in data.iterrows():
    row = list(row)
    d = {}
    for i, col in enumerate(header):
        val = row[i] if i < len(row) else None
        if col not in d and str(val) not in ('nan', 'None', ''):
            d[col] = val
    if 'model' in d:
        clean_rows.append(d)

metrics = pd.DataFrame(clean_rows)
for c in ['mae', 'rmse', 'r2', 'mape']:
    if c in metrics.columns:
        metrics[c] = pd.to_numeric(metrics[c], errors='coerce')
metrics = metrics.drop_duplicates(subset='model', keep='last').reset_index(drop=True)
metrics.to_csv('outputs/tables/model_comparison.csv', index=False)

print("Final model comparison:")
print(metrics.to_string(index=False))

# Select columns from master
sspi_all = master[['district','year','zone_type',
                   'ndvi','rainfall_annual','temp_rabi',
                   'sspi','sspi_class']]

# Drop and recreate all tables
conn.execute("DROP TABLE IF EXISTS sspi_history")
conn.execute("DROP TABLE IF EXISTS sspi_forecast")
conn.execute("DROP TABLE IF EXISTS model_metrics")
conn.commit()

sspi_all.to_sql('sspi_history',  conn, if_exists='replace', index=False)
forecast.to_sql('sspi_forecast', conn, if_exists='replace', index=False)
metrics.to_sql('model_metrics',  conn, if_exists='replace', index=False)

print('\n✅ All tables written to SQLite!')

result = pd.read_sql('SELECT district, year, zone_type, sspi FROM sspi_history LIMIT 5', conn)
print(result)
conn.close()