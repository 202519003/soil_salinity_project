# 02_weather_download.py
import requests, os

districts = {
    # Coastal zone
    'Kachchh':          (23.73, 69.86),
    'Jamnagar':         (22.47, 70.06),
    'Bhavnagar':        (21.76, 72.15),
    'Surat':            (21.17, 72.83),
    'Bharuch':          (21.70, 72.98),
    'Porbandar':        (21.64, 69.61),
    'DevbhumiDwarka':   (22.23, 68.97),
    'GirSomnath':       (20.90, 70.37),
    'Morbi':            (22.82, 70.83),
    'Rajkot':           (22.30, 70.78),
    # Canal waterlogging zone
    'Anand':            (22.55, 72.95),
    'Kheda':            (22.75, 72.68),
    'Surendranagar':    (22.72, 71.65),
    'Narmada':          (21.87, 73.49),
    'Bharuch':          (21.70, 72.98),
    'Botad':            (22.17, 71.67),
    # Inland zone
    'Mahesana':         (23.59, 72.37),
    'Patan':            (23.84, 72.11),
    'BanasKantha':      (24.17, 72.43),
    'SabarKantha':      (23.35, 73.02),
    'Aravalli':         (23.70, 73.00),
    'Ahmadabad':        (23.02, 72.57),
    'Gandhinagar':      (23.22, 72.65),
    'Vadodara':         (22.30, 73.18),
    'Amreli':           (21.60, 71.22),
    'Junagadh':         (21.52, 70.45),
    # Tribal/hilly zones
    'Dahod':            (22.83, 74.25),
    'Mahisagar':        (23.08, 73.55),
    'PanchMahals':      (22.78, 73.52),
    'ChhotaUdaipur':    (22.30, 74.01),
    'Tapi':             (21.14, 73.41),
    'Valsad':           (20.59, 72.93),
    'TheDangs':         (20.75, 73.69),
    'Navsari':          (20.95, 72.92),
}

os.makedirs('data/raw/weather', exist_ok=True)

for district, (lat, lon) in districts.items():
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
        print(f'✅ Done: {district}')
    except Exception as e:
        print(f'❌ Error: {district} — {e}')

print('All 33 Gujarat districts weather data downloaded!')
