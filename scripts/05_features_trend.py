# 05_features_trend.py
import pandas as pd
import pymannkendall as mk

master = pd.read_csv('data/processed/master_with_sspi.csv')
master = master.sort_values(['district', 'year']).reset_index(drop=True)

# ── Lag features (Positional Encoding analogue) ──
master['sspi_lag1'] = master.groupby('district')['sspi'].shift(1)
master['sspi_lag2'] = master.groupby('district')['sspi'].shift(2)
master['ndvi_lag1'] = master.groupby('district')['ndvi'].shift(1)

# ── Mann-Kendall trend test on NDVI per district ──
trend_results = []

for district in master['district'].unique():
    sub = master[master['district'] == district].sort_values('year')
    series = sub['ndvi'].values

    if len(series) >= 4:
        res = mk.original_test(series)
        trend_results.append({
            'district':         district,
            'ndvi_trend':       res.trend,
            'ndvi_trend_slope': round(res.slope, 4),
            'ndvi_trend_sig':   res.p < 0.05
        })

trend_df = pd.DataFrame(trend_results)

print('NDVI Trend Summary — Gujarat districts:')
print(trend_df.sort_values('ndvi_trend_slope').to_string())

master = master.merge(trend_df, on='district', how='left')

print(f'\nFeature table shape: {master.shape}')
print(f'Columns: {list(master.columns)}')

master.to_csv('data/processed/features_complete.csv', index=False)
print('\n✅ Step 11 complete! features_complete.csv saved.')
