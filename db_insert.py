import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

import rasterio

# ── Config ─────────────────────────────────────────────────────────────────
FILENAME    = "Belobaka3_Boeny"
CSV_PATH    = f"tiles/{FILENAME}_tiles/{FILENAME}_detections.csv"
TIF_PATH    = f"../sample_drone/{FILENAME}.tif"
TILES_DIR   = f"tiles/{FILENAME}_tiles"
SITE_NAME   = "Belobaka"
REGION      = "Boeny"
CAPTURED_AT = "2025-08-04"    # YYYY-MM-DD, or None if unknown

DB_CONFIG  = {
    "host"    : "localhost",
    "port"    : 5432,
    "dbname"  : "mangrove_db",
    "user"    : "postgres",
    "password": "root",
}

# ── Connect ────────────────────────────────────────────────────────────────
print("Connecting to database...")
conn   = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()
print("Connected.")

try:
    # ── Load CSV ───────────────────────────────────────────────────────────
    print(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded")

    # ── Get tif metadata ───────────────────────────────────────────────────
    with rasterio.open(TIF_PATH) as src:
        tif_width  = src.width
        tif_height = src.height
        crs        = str(src.crs)

    print(f"  TIF size : {tif_width} x {tif_height} px")
    print(f"  CRS      : {crs}")

    # ── Insert sample ──────────────────────────────────────────────────────

    cursor.execute("""
        INSERT INTO samples (filename, site_name, region, captured_at, total_crowns, tif_width, tif_height, crs)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE
            SET total_crowns = EXCLUDED.total_crowns,
                processed_at = now()
        RETURNING id
    """, (FILENAME, SITE_NAME, REGION, CAPTURED_AT, len(df), tif_width, tif_height, crs))

    sample_id = cursor.fetchone()[0]
    print(f"  Sample inserted — id: {sample_id}")

    # ── Insert crowns ──────────────────────────────────────────────────────
    print("Inserting crowns...")

    rows = []
    for _, row in df.iterrows():
        tile_path = os.path.join(TILES_DIR, row["tile_filename"])

        # PostGIS point: ST_SetSRID(ST_MakePoint(lon, lat), 4326)
        # PostGIS bbox polygon from the 4 corners
        geom_point = f"SRID=4326;POINT({row['geo_lon']} {row['geo_lat']})"
        geom_bbox  = (
            f"SRID=4326;POLYGON(("
            f"{row['geo_bbox_xmin']} {row['geo_bbox_ymin']}, "
            f"{row['geo_bbox_xmax']} {row['geo_bbox_ymin']}, "
            f"{row['geo_bbox_xmax']} {row['geo_bbox_ymax']}, "
            f"{row['geo_bbox_xmin']} {row['geo_bbox_ymax']}, "
            f"{row['geo_bbox_xmin']} {row['geo_bbox_ymin']}"
            f"))"
        )

        rows.append((
            row["crown_id"],
            sample_id,
            row["tile_filename"],
            tile_path,
            int(row["pixel_xmin"]),
            int(row["pixel_ymin"]),
            int(row["pixel_xmax"]),
            int(row["pixel_ymax"]),
            float(row["pixel_center_x"]),
            float(row["pixel_center_y"]),
            float(row["geo_lat"]),
            float(row["geo_lon"]),
            float(row["geo_bbox_xmin"]),
            float(row["geo_bbox_ymin"]),
            float(row["geo_bbox_xmax"]),
            float(row["geo_bbox_ymax"]),
            geom_point,
            geom_bbox,
            float(row["score"]),
            row.get("label", "Tree"),
        ))

    for row_data in rows:
        cursor.execute("""
            INSERT INTO crowns (
                crown_id, sample_id,
                tile_filename, tile_path,
                pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax,
                pixel_center_x, pixel_center_y,
                geo_lat, geo_lon,
                geo_bbox_xmin, geo_bbox_ymin, geo_bbox_xmax, geo_bbox_ymax,
                geom_point, geom_bbox,
                score, label
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                ST_GeomFromEWKT(%s), ST_GeomFromEWKT(%s),
                %s, %s
            )
            ON CONFLICT (crown_id) DO NOTHING
        """, row_data)

    conn.commit()
    print(f"  {len(rows)} crowns inserted into database.")
    print("Done.")

except Exception as e:
    conn.rollback()
    print(f"Error: {e}")
    raise

finally:
    cursor.close()
    conn.close()
    print("Connection closed.")
