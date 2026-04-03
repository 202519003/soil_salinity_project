# 10b_map_2025_current.py
# Generates: outputs/maps/01_salinity_map_2025_current.png

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')
import os

os.makedirs('outputs/maps', exist_ok=True)

print('Loading data...')

# ─── LOAD SPATIAL DATA ───────────────────────────────────────────────────────
districts = gpd.read_file('data/processed/Gujarat_districts.gpkg')
districts = districts.rename(columns={'NAME_2': 'district'})

# ─── LOAD SSPI FEATURES ──────────────────────────────────────────────────────
features = pd.read_csv('data/processed/features_complete.csv')

# Debug: show what columns and years we have
print(f"Features columns: {list(features.columns)}")
print(f"Years available:  {sorted(features['year'].unique())}")

# Get 2025 (or latest available year)
sspi_2025 = features[features['year'] == 2025].copy()
if len(sspi_2025) == 0:
    latest_year = features['year'].max()
    print(f"⚠️  2025 not found. Using latest year: {latest_year}")
    sspi_2025 = features[features['year'] == latest_year].copy()
else:
    latest_year = 2025
    print(f"✅ Using year: {latest_year}")

# Keep only columns we need — zone_type comes from features CSV
cols_needed = ['district', 'sspi']

# Add zone_type if it exists in features
if 'zone_type' in sspi_2025.columns:
    cols_needed.append('zone_type')
    print("zone_type found in features CSV ✅")
else:
    print("⚠️  zone_type NOT in features CSV — will use zone_type from shapefile")

sspi_2025 = sspi_2025[cols_needed].copy()

# Recompute sspi_class cleanly
def compute_class(v):
    if v >= 75:   return 'Critical'
    elif v >= 50: return 'High'
    elif v >= 25: return 'Moderate'
    else:         return 'Low'

sspi_2025['sspi_class'] = sspi_2025['sspi'].apply(compute_class)
print(f"Districts in SSPI data: {len(sspi_2025)}")

# ─── MERGE WITH SHAPEFILE ────────────────────────────────────────────────────
# zone_type already in shapefile (from 01_setup_spatial.py)
gdf = districts.merge(sspi_2025, on='district', how='left')

# If zone_type still missing (e.g. not in shapefile either), build it manually
if 'zone_type' not in gdf.columns:
    print("Building zone_type from district names...")
    coastal = ['Kachchh','Jamnagar','DevbhumiDwarka','Junagadh','GirSomnath',
               'Bhavnagar','Amreli','Porbandar','Morbi','Rajkot']
    canal   = ['Anand','Kheda','Surendranagar','Bharuch','Narmada','Botad']
    hilly   = ['Dahod','Mahisagar','PanchMahals','ChhotaUdaipur','Tapi','Valsad','TheDangs']
    def get_zone(name):
        if name in coastal: return 'coastal'
        elif name in canal: return 'canal'
        elif name in hilly: return 'hilly'
        else:               return 'inland'
    gdf['zone_type'] = gdf['district'].apply(get_zone)

print(f"\nFinal GDF columns: {list(gdf.columns)}")
print(f"GDF rows: {len(gdf)}")

# ─── SUMMARY TABLE ───────────────────────────────────────────────────────────
print("\nSSPI 2025 Summary:")
print(gdf[['district', 'zone_type', 'sspi', 'sspi_class']].to_string(index=False))

# ─── MAP: CURRENT 2025 SSPI CHOROPLETH ───────────────────────────────────────
print('\nGenerating 2025 current salinity map...')

fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor('#f5f5f0')
ax.set_facecolor('#cde6f5')

gdf.plot(
    column='sspi',
    ax=ax,
    cmap='RdYlGn_r',
    legend=True,
    edgecolor='white',
    linewidth=0.8,
    vmin=0,
    vmax=100,
    missing_kwds={'color': '#cccccc', 'label': 'No data'},
    legend_kwds={
        'label':       f'Soil Salinity Proxy Index (SSPI) {latest_year} — 0=None, 100=Critical',
        'orientation': 'horizontal',
        'shrink':      0.7,
        'pad':         0.02,
    }
)

# ─── DISTRICT LABELS ─────────────────────────────────────────────────────────
for _, row in gdf.iterrows():
    if row['geometry'] is None:
        continue
    cx, cy   = row['geometry'].centroid.coords[0]
    sspi_val = row['sspi']
    cls      = row['sspi_class'] if pd.notna(row.get('sspi_class', None)) else ''
    label_color = 'white' if (pd.notna(sspi_val) and sspi_val > 55) else '#1C2833'

    ax.annotate(
        f"{row['district']}\n{sspi_val:.1f} ({cls})" if pd.notna(sspi_val) else row['district'],
        xy=(cx, cy), ha='center', va='center',
        fontsize=5.5, color=label_color, fontweight='bold'
    )

# ─── CLASS THRESHOLD BOX ─────────────────────────────────────────────────────
threshold_text = (
    "SSPI Classes:\n"
    "● Critical  ≥ 75\n"
    "● High      50–74\n"
    "● Moderate  25–49\n"
    "● Low        < 25"
)
ax.text(
    0.02, 0.22, threshold_text,
    transform=ax.transAxes, fontsize=8.5,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='grey'),
    color='#1C2833', family='monospace'
)

# ─── ZONE LEGEND ─────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor='none', edgecolor='#2196F3', linewidth=2, label='Coastal Zone'),
    mpatches.Patch(facecolor='none', edgecolor='#9C27B0', linewidth=2, label='Canal Zone'),
    mpatches.Patch(facecolor='none', edgecolor='#FF9800', linewidth=2, label='Inland Zone'),
    mpatches.Patch(facecolor='none', edgecolor='#4CAF50', linewidth=2, label='Hilly Zone'),
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
          title='Salinity Zone Type', title_fontsize=9,
          framealpha=0.9, edgecolor='grey')

# ─── TITLE ───────────────────────────────────────────────────────────────────
ax.set_title(
    f'Current Soil Salinity Status — Gujarat Districts ({latest_year})\n'
    'Soil Salinity Proxy Index (SSPI) based on NDVI · Rainfall · Temperature',
    fontsize=13, fontweight='bold', pad=15, color='#1C2833'
)

fig.text(
    0.5, 0.01,
    'Data: MODIS NDVI (NASA APPEEARS) · NASA POWER Weather · CSSRI Bharuch Reference\n'
    'SSPI = 0.40×NDVI_deficit + 0.35×Rain_deficit + 0.25×Temp_anomaly (zone-weighted)',
    ha='center', fontsize=7, color='#555555', style='italic'
)

ax.set_axis_off()
plt.tight_layout(rect=[0, 0.03, 1, 1])

output_path = 'outputs/maps/01_salinity_map_2025_current.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print(f'\n✅ Map saved → {output_path}')

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
print(f'\n=== ZONE SUMMARY {latest_year} ===')
print(gdf.groupby('zone_type')['sspi'].agg(['mean','min','max','count']).round(1))
print('\n=== CLASS DISTRIBUTION ===')
print(gdf['sspi_class'].value_counts())
print('\n=== TOP 5 HIGH SALINITY DISTRICTS ===')
print(gdf.nlargest(5, 'sspi')[['district','zone_type','sspi','sspi_class']].to_string(index=False))