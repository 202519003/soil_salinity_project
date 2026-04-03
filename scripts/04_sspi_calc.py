# 04_sspi_calc.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

master = pd.read_csv('data/processed/master_raw.csv')

# Compute deviation features
master['ndvi_mean'] = master.groupby('district')['ndvi'].transform('mean')
master['rain_mean'] = master.groupby('district')['rainfall_annual'].transform('mean')
master['temp_mean'] = master.groupby('district')['temp_rabi'].transform('mean')

master['ndvi_deficit'] = master['ndvi_mean'] - master['ndvi']
master['rain_deficit'] = master['rain_mean'] - master['rainfall_annual']
master['temp_anomaly'] = master['temp_rabi'] - master['temp_mean']

# Normalise each component to 0-1
scaler = MinMaxScaler()
master[['n_ndvi', 'n_rain', 'n_temp']] = scaler.fit_transform(
    master[['ndvi_deficit', 'rain_deficit', 'temp_anomaly']]
)

# SSPI formula
master['sspi'] = (
    0.40 * master['n_ndvi'] +
    0.35 * master['n_rain'] +
    0.25 * master['n_temp']
) * 100

# ── ZONE WEIGHT — KEY ADDITION ──
zone_weight = {'coastal': 1.5, 'canal': 1.2, 'inland': 0.8, 'hilly': 0.7}
master['zone_factor'] = master['zone_type'].map(zone_weight).fillna(1.0)
master['sspi'] = (master['sspi'] * master['zone_factor']).clip(0, 100).round(1)

def sspi_class(score):
    if score >= 75:   return 'Critical'
    elif score >= 50: return 'High'
    elif score >= 25: return 'Moderate'
    else:             return 'Low'

master['sspi_class'] = master['sspi'].apply(sspi_class)

print('SSPI distribution:')
print(master['sspi'].describe())
print('\nSSPI class distribution:')
print(master['sspi_class'].value_counts())
print('\nZone-wise SSPI mean:')
print(master.groupby('zone_type')['sspi'].mean())
print('\nCoastal districts SSPI:')
print(master[master['zone_type']=='coastal'].groupby('district')['sspi'].mean().sort_values(ascending=False))

master.to_csv('data/processed/master_with_sspi.csv', index=False)
print('\n✅ Step 10 complete! master_with_sspi.csv saved.')
