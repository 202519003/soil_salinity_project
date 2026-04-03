import pandas as pd
import os

# Reference values based on published Gujarat salinity literature
# Sources: CSSRI Annual Reports, ICAR Gujarat studies (2015-2020)

cssri_data = {
    'district': [
        # Coastal zone districts
        'Kutch', 'Jamnagar', 'Bhavnagar', 'Surat', 'Bharuch',
        'Porbandar', 'Devbhumi Dwarka', 'Gir Somnath', 'Morbi',
        # Canal waterlogging zone
        'Anand', 'Kheda', 'Surendranagar', 'Narmada', 'Navsari',
        # Inland borewell zone
        'Mehsana', 'Patan', 'Banaskantha', 'Sabarkantha', 'Aravalli',
        # Low salinity zones
        'Ahmedabad', 'Gandhinagar', 'Vadodara', 'Rajkot',
        'Amreli', 'Junagadh', 'Botad',
        # Tribal/hilly - very low salinity
        'Dahod', 'Mahisagar', 'Panchmahals', 'Chhota Udaipur',
        'Tapi', 'Valsad', 'Dang'
    ],
    'salinity_class': [
        # Coastal
        'High', 'High', 'High', 'Medium', 'Medium',
        'High', 'High', 'Medium', 'High',
        # Canal
        'Medium', 'Medium', 'High', 'Low', 'Low',
        # Inland
        'Medium', 'Medium', 'Medium', 'Low', 'Low',
        # Low salinity
        'Low', 'Low', 'Low', 'Low',
        'Low', 'Low', 'Low',
        # Tribal/hilly
        'Very Low', 'Very Low', 'Very Low', 'Very Low',
        'Very Low', 'Very Low', 'Very Low'
    ],
    'ec_value_dsm': [
        # Coastal
        8.5, 7.2, 6.1, 3.2, 4.1,
        7.8, 7.5, 3.8, 6.5,
        # Canal
        3.8, 3.5, 6.8, 1.8, 1.6,
        # Inland
        4.5, 4.2, 4.0, 2.1, 2.0,
        # Low salinity
        1.4, 1.1, 1.5, 1.8,
        2.0, 1.9, 2.1,
        # Tribal/hilly
        0.8, 0.7, 0.9, 0.8,
        0.6, 0.5, 0.4
    ],
    'zone_type': [
        # Coastal
        'coastal', 'coastal', 'coastal', 'coastal', 'canal',
        'coastal', 'coastal', 'coastal', 'coastal',
        # Canal
        'canal', 'canal', 'canal', 'canal', 'canal',
        # Inland
        'inland', 'inland', 'inland', 'inland', 'inland',
        # Low salinity
        'inland', 'inland', 'inland', 'inland',
        'inland', 'inland', 'inland',
        # Tribal/hilly
        'hilly', 'hilly', 'hilly', 'hilly',
        'hilly', 'hilly', 'hilly'
    ]
}

df = pd.DataFrame(cssri_data)
os.makedirs('data/raw/satellite', exist_ok=True)
df.to_excel('data/raw/satellite/cssri_salinity_ref.xlsx', index=False)

print(f"✅ Reference file created with {len(df)} districts!")
print(df.to_string())