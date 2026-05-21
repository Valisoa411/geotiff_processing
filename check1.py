import rasterio

FILENAME    = "Belobaka3_Boeny"
TIF_PATH     = f"../sample_drone/{FILENAME}.tif"
with rasterio.open(TIF_PATH) as src:
    print("Bands:", src.count)
    print("Size:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("Dtype:", src.dtypes)