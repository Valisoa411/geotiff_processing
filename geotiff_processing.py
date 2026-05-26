from deepforest import main
import rasterio
from rasterio.windows import Window
import rasterio.transform
import numpy as np
from PIL import Image
import os
import shutil
import pandas as pd
from datetime import datetime
import shutil

# ── Config ─────────────────────────────────────────────────────────────────
FILENAME     = "Belobaka3_Boeny"
TIF_PATH     = f"../sample_drone/{FILENAME}.tif"
TEMP_DIR     = "temp_tiles"
OUTPUT_DIR   = f"tiles/{FILENAME}_tiles"
MODEL_DIR    = "models/deepforest_tree"
TILE_SIZE    = 1024
OVERLAP_PX   = 100
MIN_SCORE    = 0.4
CROP_PADDING = 32

try:
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

    Image.MAX_IMAGE_PIXELS = None

    # ── Step 1: split tif into manageable tiles, run DeepForest on each ───────
    all_predictions = []

    with rasterio.open(TIF_PATH) as src:
        W, H       = src.width, src.height
        transform  = src.transform   # affine transform: pixel → geographic coords
        crs        = src.crs         # coordinate reference system
        print(f"Image size : {W} x {H} px")
        print(f"CRS        : {crs}")
        print(f"Transform  : {transform}")

        step = TILE_SIZE - OVERLAP_PX

        tile_idx = 0
        for row_off in range(0, H, step):
            for col_off in range(0, W, step):

                actual_w = min(TILE_SIZE, W - col_off)
                actual_h = min(TILE_SIZE, H - row_off)

                if actual_w < 64 or actual_h < 64:
                    continue

                window = Window(col_off, row_off, actual_w, actual_h)
                data   = src.read(window=window)
                rgb    = np.moveaxis(data[:3], 0, -1)

                if rgb.mean() < 5:
                    continue

                temp_path = os.path.join(TEMP_DIR, f"tile_{tile_idx:05d}.png")
                Image.fromarray(rgb.astype(np.uint8)).save(temp_path)

                try:
                    preds = model.predict_image(path=temp_path)
                except Exception as e:
                    print(f"  Skipping tile {tile_idx} — {e}")
                    tile_idx += 1
                    continue

                if preds is None or len(preds) == 0:
                    tile_idx += 1
                    continue

                preds = preds[preds["score"] >= MIN_SCORE].copy()

                # Convert local tile coords → global pixel coords
                preds["xmin"] += col_off
                preds["xmax"] += col_off
                preds["ymin"] += row_off
                preds["ymax"] += row_off
                preds["tile"]  = tile_idx

                all_predictions.append(preds)
                print(f"  Tile {tile_idx}: {len(preds)} crowns found")

                tile_idx += 1

    print(f"\nTotal tiles processed: {tile_idx}")

    # ── Step 2: crop patches + extract geographic coordinates ─────────────────
    if not all_predictions:
        print("No crowns detected. Try lowering MIN_SCORE.")
    else:
        all_preds_df = pd.concat(all_predictions, ignore_index=True)
        print(f"Total crowns detected: {len(all_preds_df)}")

        # Columns we will fill for the final CSV
        records = []

        with rasterio.open(TIF_PATH) as src:
            for idx, row in all_preds_df.iterrows():

                # ── Pixel bounding box with padding ───────────────────────────
                xmin = max(0, int(row["xmin"]) - CROP_PADDING)
                ymin = max(0, int(row["ymin"]) - CROP_PADDING)
                xmax = min(src.width,  int(row["xmax"]) + CROP_PADDING)
                ymax = min(src.height, int(row["ymax"]) + CROP_PADDING)

                w = xmax - xmin
                h = ymax - ymin
                if w < 32 or h < 32:
                    continue

                # ── Crop and save tile image ───────────────────────────────────
                tile_filename = f"crown_{idx:05d}_score{row['score']:.2f}.png"
                window  = Window(xmin, ymin, w, h)
                patch   = src.read(window=window)
                rgb     = np.moveaxis(patch[:3], 0, -1)
                Image.fromarray(rgb.astype(np.uint8)).save(
                    os.path.join(OUTPUT_DIR, tile_filename)
                )

                # ── Geographic coordinates of the crown center ────────────────
                center_col = (int(row["xmin"]) + int(row["xmax"])) / 2   # no padding — true crown center
                center_row = (int(row["ymin"]) + int(row["ymax"])) / 2

                geo_lon, geo_lat = rasterio.transform.xy(
                    src.transform,
                    center_row,   # row = y axis
                    center_col,   # col = x axis
                    offset="center"
                )

                # ── Geographic bounding box corners ───────────────────────────
                geo_xmin, geo_ymax = rasterio.transform.xy(src.transform, ymin, xmin, offset="ul")
                geo_xmax, geo_ymin = rasterio.transform.xy(src.transform, ymax, xmax, offset="ul")

                records.append({
                    # identification
                    "sample"         : FILENAME,
                    "tile_filename"  : tile_filename,
                    "crown_id"       : f"{FILENAME}_{idx:05d}",
                    # pixel coordinates (for reconstructing position on the tif)
                    "pixel_xmin"     : xmin,
                    "pixel_ymin"     : ymin,
                    "pixel_xmax"     : xmax,
                    "pixel_ymax"     : ymax,
                    "pixel_center_x" : center_col,
                    "pixel_center_y" : center_row,
                    # geographic coordinates (for the map)
                    "geo_lat"        : geo_lat,
                    "geo_lon"        : geo_lon,
                    "geo_bbox_xmin"  : geo_xmin,
                    "geo_bbox_ymin"  : geo_ymin,
                    "geo_bbox_xmax"  : geo_xmax,
                    "geo_bbox_ymax"  : geo_ymax,
                    # detection metadata
                    "score"          : round(row["score"], 4),
                    "label"          : row.get("label", "Tree"),
                    "processed_at"   : datetime.now().isoformat(),
                })

        # ── Save CSV ───────────────────────────────────────────────────────────
        csv_path = f"{OUTPUT_DIR}/{FILENAME}_detections.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"\nSaved {len(records)} crown tiles to    : {OUTPUT_DIR}/")
        print(f"Detection CSV saved to                 : {csv_path}")
        print(f"\nCSV columns: {list(pd.DataFrame(records).columns)}")

finally:
    # ── Cleanup ────────────────────────────────────────────────────────────────
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print("Temp tiles cleaned up.")