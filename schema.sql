-- ── Enable PostGIS ─────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS postgis;

-- ── samples ────────────────────────────────────────────────────────────────
-- One row per drone .tif file processed
CREATE TABLE IF NOT EXISTS samples (
    id              SERIAL PRIMARY KEY,
    filename        TEXT        NOT NULL UNIQUE,    -- e.g. 19_08_2025_AMBONIO_Melaky
    site_name       TEXT,                           -- e.g. AMBONIO
    region          TEXT,                           -- e.g. Melaky
    captured_at     DATE,                           -- date from filename
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    tif_width       INT,                            -- full image width in px
    tif_height      INT,                            -- full image height in px
    crs             TEXT,                           -- coordinate reference system e.g. EPSG:4326
    total_crowns    INT                             -- total crowns detected in this sample
);

-- ── crowns ─────────────────────────────────────────────────────────────────
-- One row per detected tree crown (output of DeepForest)
CREATE TABLE IF NOT EXISTS crowns (
    id              SERIAL PRIMARY KEY,
    crown_id        TEXT        NOT NULL UNIQUE,    -- e.g. 19_08_2025_AMBONIO_Melaky_00001
    sample_id       INT         NOT NULL REFERENCES samples(id) ON DELETE CASCADE,

    -- tile image
    tile_filename   TEXT        NOT NULL,           -- crown_00001_score0.87.png
    tile_path       TEXT,                           -- full path to the PNG

    -- pixel coordinates on the original tif
    pixel_xmin      INT,
    pixel_ymin      INT,
    pixel_xmax      INT,
    pixel_ymax      INT,
    pixel_center_x  FLOAT,
    pixel_center_y  FLOAT,

    -- geographic coordinates
    geo_lat         FLOAT,                          -- center latitude
    geo_lon         FLOAT,                          -- center longitude
    geo_bbox_xmin   FLOAT,                          -- bounding box
    geo_bbox_ymin   FLOAT,
    geo_bbox_xmax   FLOAT,
    geo_bbox_ymax   FLOAT,

    -- PostGIS geometry columns (for spatial queries)
    geom_point      GEOMETRY(Point, 4326),          -- center point
    geom_bbox       GEOMETRY(Polygon, 4326),        -- bounding box polygon

    -- detection metadata
    score           FLOAT,                          -- DeepForest confidence 0-1
    label           TEXT        DEFAULT 'Tree',     -- will be updated by Stage 3 model
    species         TEXT,                           -- filled after species classification
    health_status   TEXT,                           -- filled after health classification
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── indexes ────────────────────────────────────────────────────────────────
-- Speed up spatial queries on the map
CREATE INDEX IF NOT EXISTS idx_crowns_geom_point ON crowns USING GIST(geom_point);
CREATE INDEX IF NOT EXISTS idx_crowns_geom_bbox  ON crowns USING GIST(geom_bbox);
CREATE INDEX IF NOT EXISTS idx_crowns_sample_id  ON crowns(sample_id);
CREATE INDEX IF NOT EXISTS idx_crowns_score      ON crowns(score);
