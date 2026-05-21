from deepforest import main
from deepforest.utilities import read_file
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import os
import pandas as pd

# ── 1. Load pre-trained model ──────────────────────────────────────────────
model = main.deepforest()
model.use_release()  # downloads pre-trained weights automatically

# ── 2. Config ──────────────────────────────────────────────────────────────
FILENAME    = "Belobaka3_Boeny"
TIF_PATH     = f"{FILENAME}.tif"
OUTPUT_DIR   = f"tiles/{FILENAME}_tiles"
TILE_SIZE    = 512        # px — window size fed into DeepForest
OVERLAP      = 0.2        # 20% overlap avoids missing crowns at tile edges
CROP_PADDING = 32         # extra px added around each detected crown
MIN_SCORE    = 0.4        # confidence threshold — lower = more detections
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 3. Predict on large tif (handles tiling internally) ───────────────────
print("Running crown detection... (this will take a while)")
predictions = model.predict_tile(
    raster_path=TIF_PATH,
    patch_size=TILE_SIZE,
    patch_overlap=OVERLAP,
    return_plot=False
)

# predictions is a GeoDataFrame with columns:
# xmin, ymin, xmax, ymax, score, label, geometry

print(f"Detected {len(predictions)} tree crowns")

# ── 4. Filter by confidence ────────────────────────────────────────────────
predictions = predictions[predictions["score"] >= MIN_SCORE]
print(f"After filtering: {len(predictions)} crowns")

# ── 5. Crop a tile around each detected crown ──────────────────────────────
with rasterio.open(TIF_PATH) as src:
    for idx, row in predictions.iterrows():
        # Add padding around the bounding box
        xmin = max(0, int(row["xmin"]) - CROP_PADDING)
        ymin = max(0, int(row["ymin"]) - CROP_PADDING)
        xmax = min(src.width,  int(row["xmax"]) + CROP_PADDING)
        ymax = min(src.height, int(row["ymax"]) + CROP_PADDING)

        width  = xmax - xmin
        height = ymax - ymin

        # Skip if crop is too small (likely a false positive)
        if width < 32 or height < 32:
            continue

        window = Window(xmin, ymin, width, height)
        patch  = src.read(window=window)          # shape: (bands, H, W)
        patch  = np.moveaxis(patch[:3], 0, -1)   # → (H, W, 3) RGB only

        # Save as PNG
        out_path = os.path.join(OUTPUT_DIR, f"crown_{idx:05d}_score{row['score']:.2f}.png")
        Image.fromarray(patch.astype(np.uint8)).save(out_path)

print(f"Saved tiles to: {OUTPUT_DIR}/")

# ── 6. Save the detections as CSV for reference ────────────────────────────
predictions.drop(columns="geometry").to_csv(f"{OUTPUT_DIR}/{FILENAME}_detections.csv", index=False)
print(f"Detection coordinates saved to {OUTPUT_DIR}/{FILENAME}_detections.csv")