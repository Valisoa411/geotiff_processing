import rasterio

FILENAME    = "Belobaka3_Boeny"
TIF_PATH     = f"../sample_drone/{FILENAME}.tif"
with rasterio.open(TIF_PATH) as src:
    print(src.tags())          # general metadata
    print(src.profile)