import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

print('Loading data...')

# ─── LOAD DATA ───
districts = gpd.read_file('data/processed/Gujarat_districts.gpkg')
districts = districts.rename(columns={'NAME_2': 'district'})

forecast  = pd.read_csv('outputs/tables/salinity_forecast_2026_2030.csv')
features  = pd.read_csv('data/processed/features_complete.csv')

# ─── MAP 1: SSPI 2030 CHOROPLETH ───
print('Generating salinity map 2030...')

pred_2030 = forecast[forecast['year'] == 2030]
gdf = districts.merge(pred_2030[['district','predicted_sspi','trend']], on='district', how='left')

fig, ax = plt.subplots(figsize=(14, 12))
gdf.plot(
    column='predicted_sspi',
    ax=ax,
    cmap='RdYlGn_r',
    legend=True,
    edgecolor='white',
    linewidth=0.8,
    legend_kwds={
        'label': 'Predicted SSPI 2030 (0=Low, 100=Critical)',
        'orientation': 'horizontal'
    }
)

# Add district name labels
for _, row in gdf.iterrows():
    if row['geometry'] is not None:
        ax.annotate(
            f"{row['district']}\n({row['predicted_sspi']:.1f})",
            xy=row['geometry'].centroid.coords[0],
            ha='center',
            fontsize=6,
            color='#1C2833'
        )

ax.set_title(
    'Predicted Soil Salinity Proxy Index 2030\nGujarat Districts',
    fontsize=14,
    fontweight='bold'
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig('outputs/maps/02_salinity_map_2030.png', dpi=200, bbox_inches='tight')
plt.show()
print('Map 1 saved!')

# ─── MAP 2: TREND MAP ───
print('Generating trend map...')

fig2, ax2 = plt.subplots(figsize=(14, 12))

trend_colors = {
    'Worsening': '#E74C3C',
    'Stable':    '#2ECC71',
    'Improving': '#3498DB'
}

gdf['trend'] = gdf['trend'].fillna('Stable')
gdf['color'] = gdf['trend'].map(trend_colors)
gdf.plot(ax=ax2, color=gdf['color'], edgecolor='white', linewidth=0.8)

# Add district labels
for _, row in gdf.iterrows():
    if row['geometry'] is not None:
        ax2.annotate(
            row['district'],
            xy=row['geometry'].centroid.coords[0],
            ha='center',
            fontsize=6.5,
            color='#1C2833'
        )

# Legend
patches = [
    mpatches.Patch(color='#E74C3C', label='Worsening'),
    mpatches.Patch(color='#2ECC71', label='Stable'),
    mpatches.Patch(color='#3498DB', label='Improving'),
]
ax2.legend(handles=patches, loc='lower left', fontsize=12)
ax2.set_title(
    'Salinity Trend 2026–2030 by District\nGujarat',
    fontsize=14,
    fontweight='bold'
)
ax2.set_axis_off()
plt.tight_layout()
plt.savefig('outputs/maps/03_trend_map.png', dpi=200, bbox_inches='tight')
plt.show()
print('Map 2 saved!')

# ─── CHART 1: NDVI TREND — TOP WORSENING DISTRICTS ───
print('Generating NDVI trend chart...')

worsening = forecast[forecast['trend'] == 'Worsening']['district'].unique()[:5]

fig3, ax3 = plt.subplots(figsize=(12, 5))
colors = ['#E74C3C','#E67E22','#9B59B6','#1ABC9C','#3498DB']

for i, dist in enumerate(worsening):
    hist = features[features['district'] == dist].sort_values('year')
    if len(hist) > 0:
        ax3.plot(
            hist['year'],
            hist['ndvi'],
            'o-',
            color=colors[i],
            label=dist,
            linewidth=2
        )

ax3.axvline(x=2025.5, color='gray', linestyle='--', alpha=0.6, label='Forecast boundary')
ax3.set_title('NDVI Trend — Top Worsening Gujarat Districts (2015–2025)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Year')
ax3.set_ylabel('Mean Annual NDVI')
ax3.legend()
plt.tight_layout()
plt.savefig('outputs/charts/03_ndvi_trend_chart.png', dpi=150)
plt.show()
print('Chart 1 saved!')

# ─── CHART 2: SSPI FORECAST BAR CHART ───
print('Generating forecast summary chart...')

top_districts = forecast[forecast['year'] == 2030].sort_values(
    'predicted_sspi', ascending=False).head(10)

fig4, ax4 = plt.subplots(figsize=(12, 6))
colors_bar = ['#E74C3C' if t == 'Worsening' else '#2ECC71'
              for t in top_districts['trend']]

ax4.barh(top_districts['district'], top_districts['predicted_sspi'],
         color=colors_bar)
ax4.set_xlabel('Predicted SSPI 2030')
ax4.set_title('Top 10 Gujarat Districts by Predicted Salinity 2030', fontsize=13, fontweight='bold')

patches = [
    mpatches.Patch(color='#E74C3C', label='Worsening'),
    mpatches.Patch(color='#2ECC71', label='Stable'),
]
ax4.legend(handles=patches)
plt.tight_layout()
plt.savefig('outputs/charts/04_forecast_bar_chart.png', dpi=150)
plt.show()
print('Chart 2 saved!')

print('\n✅ Step 16 complete!')
print('Maps saved to outputs/maps/')
print('Charts saved to outputs/charts/')
