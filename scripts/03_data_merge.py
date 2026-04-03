# 03_data_merge.py
import pandas as pd, os, requests
import geopandas as gpd

# ── STEP 1: Auto-download missing weather files ──
missing = {
    'BanasKantha': (24.17, 72.43),
    'PanchMahals': (22.78, 73.52),
    'SabarKantha': (23.35, 73.02),
}

for district, (lat, lon) in missing.items():
    filepath = f'data/raw/weather/{district}_weather.csv'
    if os.path.exists(filepath):
        print(f'⏭️  Already exists: {district}')
        continue
    try:
        url = (
            f'https://power.larc.nasa.gov/api/temporal/daily/point'
            f'?parameters=PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M,WS2M'
            f'&community=RE&longitude={lon}&latitude={lat}'
            f'&start=20150101&end=20251231&format=CSV'
        )
        r = requests.get(url, timeout=60)
        with open(filepath, 'w') as f:
            f.write(r.text)
        print(f'✅ Downloaded: {district}')
    except Exception as e:
        print(f'❌ Download error: {district} — {e}')

# ── STEP 2: Merge NDVI + Weather ──
NAME_MAP = {
    'Ahmedabad':        'Ahmadabad',
    'Kutch':            'Kachchh',
    'Mehsana':          'Mahesana',
    'Gir_Somnath':      'GirSomnath',
    'Devbhumi_Dwarka':  'DevbhumiDwarka',
    'Banaskantha':      'BanasKantha',
    'Sabarkantha':      'SabarKantha',
    'Panchmahals':      'PanchMahals',
    'Chhota_Udaipur':   'ChhotaUdaipur',
    'Dang':             'TheDangs',
    'Jamnagar':         'Jamnagar',
    'Bhavnagar':        'Bhavnagar',
    'Surat':            'Surat',
    'Bharuch':          'Bharuch',
    'Porbandar':        'Porbandar',
    'Morbi':            'Morbi',
    'Anand':            'Anand',
    'Kheda':            'Kheda',
    'Surendranagar':    'Surendranagar',
    'Narmada':          'Narmada',
    'Navsari':          'Navsari',
    'Patan':            'Patan',
    'Aravalli':         'Aravalli',
    'Gandhinagar':      'Gandhinagar',
    'Vadodara':         'Vadodara',
    'Rajkot':           'Rajkot',
    'Amreli':           'Amreli',
    'Junagadh':         'Junagadh',
    'Botad':            'Botad',
    'Dahod':            'Dahod',
    'Mahisagar':        'Mahisagar',
    'Tapi':             'Tapi',
    'Valsad':           'Valsad',
    'Kachchh':          'Kachchh',
    'Mahesana':         'Mahesana',
    'GirSomnath':       'GirSomnath',
    'DevbhumiDwarka':   'DevbhumiDwarka',
    'BanasKantha':      'BanasKantha',
    'SabarKantha':      'SabarKantha',
    'PanchMahals':      'PanchMahals',
    'ChhotaUdaipur':    'ChhotaUdaipur',
    'TheDangs':         'TheDangs',
    'Ahmadabad':        'Ahmadabad',
}

SKIP = ['Middle_Gujarat', 'North_Gujarat', 'South_Gujarat',
        'South_Saurashtra', 'Saurashtra', 'East_Gujarat', 'Central_Gujarat']

# Load NDVI
ndvi = pd.read_csv('data/raw/satellite/Gujarat_NDVI_2015_2025.csv')
ndvi = ndvi[['NAME_2', 'year', 'mean']].copy()
ndvi.columns = ['district', 'year', 'ndvi']
ndvi['district'] = ndvi['district'].astype(str).str.strip()
ndvi = ndvi.dropna()
ndvi['year'] = ndvi['year'].astype(int)
print(f'NDVI rows: {len(ndvi)}')

# Load zone_type
gdf = gpd.read_file('data/processed/Gujarat_districts.gpkg')
zone_map = dict(zip(gdf['NAME_2'], gdf['zone_type']))

# Load weather
weather_records = []

for fname in os.listdir('data/raw/weather'):
    if not fname.endswith('_weather.csv'):
        continue

    raw_name = fname.replace('_weather.csv', '')

    if raw_name in SKIP:
        print(f'⏭️  Skipping: {raw_name}')
        continue

    district = NAME_MAP.get(raw_name, raw_name)

    try:
        df = pd.read_csv(
            f'data/raw/weather/{fname}',
            skiprows=26,
            on_bad_lines='skip'
        )
        df.columns = df.columns.str.strip()

        df['date'] = pd.to_datetime(
            df['YEAR'].astype(str) + '-' +
            df['MO'].astype(str) + '-' +
            df['DY'].astype(str),
            format='%Y-%m-%d',
            errors='coerce'
        )
        df['month'] = df['date'].dt.month
        df['year']  = df['date'].dt.year

        annual = df.groupby('year')['PRECTOTCORR'].sum().reset_index()
        annual.columns = ['year', 'rainfall_annual']

        rabi   = df[df['month'].isin([10,11,12,1,2,3])]
        rabi_t = rabi.groupby('year')['T2M_MAX'].mean().reset_index()
        rabi_t.columns = ['year', 'temp_rabi']

        merged = annual.merge(rabi_t, on='year')
        merged['district'] = district
        weather_records.append(merged)
        print(f'✅ {district}')

    except Exception as e:
        print(f'❌ Error {district}: {e}')

weather_df = pd.concat(weather_records, ignore_index=True)
print(f'\nWeather rows: {len(weather_df)}')
print(f'Weather districts: {weather_df["district"].nunique()}')

# Merge NDVI + weather + zone_type
master = ndvi.merge(weather_df, on=['district', 'year'], how='left')
master['zone_type'] = master['district'].map(zone_map).fillna('inland')

print(f'\nMaster table shape: {master.shape}')
print(f'Missing values:\n{master.isnull().sum()}')
print(master.head(10))

master.to_csv('data/processed/master_raw.csv', index=False)
print('\n✅ Step 9 complete! master_raw.csv saved.')

