import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load your existing Gujarat geopackage
gujarat = gpd.read_file('data/raw/spatial/GujaratGeo.gpkg')

# Dissolve from taluk level (213 rows) to district level (33 rows)
gujarat = gujarat.dissolve(by='NAME_2').reset_index()

print(f'Gujarat districts: {len(gujarat)}')

# Zone classification — matched to your exact district names
coastal = [
    'Kachchh',
    'Jamnagar',
    'DevbhumiDwarka',
    'Junagadh',
    'GirSomnath',
    'Bhavnagar',
    'Amreli',
    'Porbandar',
    'Morbi',
    'Rajkot',
]

canal = [
    'Anand',
    'Kheda',
    'Surendranagar',
    'Bharuch',
    'Narmada',
    'Botad',
]

hilly = [
    'Dahod',
    'Mahisagar',
    'PanchMahals',
    'ChhotaUdaipur',
    'Tapi',
    'Valsad',
    'TheDangs',
]

def get_zone(name):
    if name in coastal: return 'coastal'
    elif name in canal: return 'canal'
    elif name in hilly: return 'hilly'
    else:               return 'inland'

gujarat['zone_type'] = gujarat['NAME_2'].apply(get_zone)

print(gujarat[['NAME_2', 'zone_type']].to_string())

# Save to processed folder
gujarat.to_file('data/processed/Gujarat_districts.gpkg', driver='GPKG')

# Map coloured by zone_type with district name labels
color_map = {'coastal': '#2196F3', 'canal': '#9C27B0', 'inland': '#FF9800', 'hilly': '#4CAF50'}
gujarat['color'] = gujarat['zone_type'].map(color_map)

fig, ax = plt.subplots(figsize=(14, 12))
gujarat.plot(ax=ax, color=gujarat['color'], edgecolor='white', linewidth=0.8)

# Add district name labels
for _, row in gujarat.iterrows():
    centroid = row['geometry'].centroid
    plt.annotate(
        text=row['NAME_2'],
        xy=(centroid.x, centroid.y),
        ha='center',
        fontsize=6,
        color='black'
    )

# Add legend
coastal_patch = mpatches.Patch(color='#2196F3', label='Coastal (10 districts)\nSaltwater intrusion')
canal_patch   = mpatches.Patch(color='#9C27B0', label='Canal (6 districts)\nWaterlogging / Narmada')
inland_patch  = mpatches.Patch(color='#FF9800', label='Inland (10 districts)\nBorewell upconing')
hilly_patch   = mpatches.Patch(color='#4CAF50', label='Hilly (7 districts)\nNatural drainage')

ax.legend(
    handles=[coastal_patch, canal_patch, inland_patch, hilly_patch],
    loc='lower left',
    fontsize=10,
    title='Salinity Zone Type',
    title_fontsize=11,
    framealpha=0.9,
    edgecolor='grey'
)

plt.tight_layout()
plt.savefig('outputs/maps/00_gujarat_zones.png', dpi=150)
plt.show()
print('Step 8 complete!')