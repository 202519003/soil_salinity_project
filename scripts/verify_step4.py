import geopandas as gpd

gdf = gpd.read_file('data/raw/spatial/GujaratGeo.gpkg')

print("Shape:", gdf.shape)
print("Columns:", gdf.columns.tolist())
print("CRS:", gdf.crs)
print(gdf.head())