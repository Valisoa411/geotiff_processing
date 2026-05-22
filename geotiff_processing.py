from deepforest import main
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image
import os
import shutil

# ── Config ─────────────────────────────────────────────────────────────────
FILENAME      = "19_08_2025_AMBONIO_Melaky"
TIF_PATH      = f"../sample_drone/{FILENAME}.tif"
TEMP_DIR      = "temp_tiles"                # intermediate tiles fed into DeepForest
OUTPUT_DIR   = f"tiles/{FILENAME}_tiles"    # final cropped crowns
MODEL_DIR     = "models/deepforest_tree"
TILE_SIZE     = 1024                        # px — size of chunks to split the tif into
OVERLAP_PX    = 100                         # px overlap between chunks to avoid missing crowns at edges
MIN_SCORE     = 0.4
CROP_PADDING  = 32

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────
model = main.deepforest()
if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
    print("Loading model from local cache...")
    model.load_model(MODEL_DIR)
else:
    print("Downloading model...")
    model.load_model("weecology/deepforest-tree", revision="main")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.model.save_pretrained(MODEL_DIR)

# ── Increase PIL limit just for our controlled crops ──────────────────────
Image.MAX_IMAGE_PIXELS = None   # we control tile size so this is safe

# ── Step 1: split tif into manageable tiles, run DeepForest on each ───────
all_predictions = []

with rasterio.open(TIF_PATH) as src:
    W, H = src.width, src.height
    print(f"Image size: {W} x {H} px")

    step = TILE_SIZE - OVERLAP_PX   # step with overlap

    tile_idx = 0
    for row_off in range(0, H, step):
        for col_off in range(0, W, step):

            # Clamp to image bounds
            actual_w = min(TILE_SIZE, W - col_off)
            actual_h = min(TILE_SIZE, H - row_off)

            if actual_w < 64 or actual_h < 64:
                continue

            # Read tile from tif
            window = Window(col_off, row_off, actual_w, actual_h)
            data   = src.read(window=window)         # (bands, H, W)
            rgb    = np.moveaxis(data[:3], 0, -1)    # (H, W, 3)

            # Skip nearly empty/black tiles (water, no data)
            if rgb.mean() < 5:
                continue

            # Save tile as temp PNG for DeepForest
            temp_path = os.path.join(TEMP_DIR, f"tile_{tile_idx:05d}.png")
            Image.fromarray(rgb.astype(np.uint8)).save(temp_path)

            # Run DeepForest on this tile
            try:
                preds = model.predict_image(path=temp_path)
            except Exception as e:
                print(f"  Skipping tile {tile_idx} — {e}")
                tile_idx += 1
                continue

            if preds is None or len(preds) == 0:
                tile_idx += 1
                continue

            # Filter by confidence
            preds = preds[preds["score"] >= MIN_SCORE]

            # Convert local tile coords → global image coords
            preds["xmin"] += col_off
            preds["xmax"] += col_off
            preds["ymin"] += row_off
            preds["ymax"] += row_off
            preds["tile"]  = tile_idx

            all_predictions.append(preds)
            print(f"  Tile {tile_idx}: {len(preds)} crowns found")

            tile_idx += 1

print(f"\nTotal tiles processed: {tile_idx}")

# ── Step 2: crop a patch around each detected crown ───────────────────────
import pandas as pd
if not all_predictions:
    print("No crowns detected. Try lowering MIN_SCORE.")
else:
    all_preds_df = pd.concat(all_predictions, ignore_index=True)
    print(f"Total crowns detected: {len(all_preds_df)}")

    with rasterio.open(TIF_PATH) as src:
        for idx, row in all_preds_df.iterrows():
            xmin = max(0, int(row["xmin"]) - CROP_PADDING)
            ymin = max(0, int(row["ymin"]) - CROP_PADDING)
            xmax = min(src.width,  int(row["xmax"]) + CROP_PADDING)
            ymax = min(src.height, int(row["ymax"]) + CROP_PADDING)

            w = xmax - xmin
            h = ymax - ymin
            if w < 32 or h < 32:
                continue

            window = Window(xmin, ymin, w, h)
            patch  = src.read(window=window)
            rgb    = np.moveaxis(patch[:3], 0, -1)

            out_path = os.path.join(OUTPUT_DIR, f"crown_{idx:05d}_score{row['score']:.2f}.png")
            Image.fromarray(rgb.astype(np.uint8)).save(out_path)

    all_preds_df.to_csv(f"{OUTPUT_DIR}/{FILENAME}_detections.csv", index=False)
    print(f"Detection coordinates saved to {OUTPUT_DIR}/{FILENAME}_detections.csv")
    print(f"Saved {len(all_preds_df)} crown tiles to {OUTPUT_DIR}/")

shutil.rmtree(TEMP_DIR)
print("Temp tiles cleaned up.")