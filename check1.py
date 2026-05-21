import rasterio
with rasterio.open("your_map.tif") as src:
    print("Bands:", src.count)
    print("Size:", src.width, "x", src.height)
    print("CRS:", src.crs)
    print("Dtype:", src.dtypes)