"""Dashboard overlays from the fix6 Prithvi-EO 2.0 (seed 42) maps, the same maps every Prithvi number in the paper uses.

Writes, per city, into results/pipeline_fix6/web/{city}/:
  web_{city}_{2018,2021,2026}.png      RGBA class overlay in EPSG:4326 (NaN input pixels transparent)
  web_deforestation_{city}.png         2018 vegetation pixels coloured by their 2026 class (others transparent)
and results/pipeline_fix6/web/vegetation_change_stats.json (pixel counts on pixels valid in both years).
The reprojection matches fix_projection.py (default transform to EPSG:4326, nearest neighbour), so the image
bounds equal the ones already hard-coded in the dashboard.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pipeline_fix6" / "web"
COLORS = [(34, 197, 94), (239, 68, 68), (234, 179, 8), (59, 130, 246), (210, 180, 140)]  # dashboard palette
NODATA = 255
INPUTS = {
    "lahore": {y: ROOT / "data" / "inference" / "lahore" / f"sentinel2_{y}_10bands.tif" for y in (2018, 2021, 2026)},
    "gujranwala": {y: ROOT / "data" / "inference" / "gujranwala" / f"gujranwala_sentinel2_{y}_10bands.tif" for y in (2018, 2021, 2026)},
}


def load(city, year):
    with rasterio.open(ROOT / "results" / "pipeline_fix6" / city / f"lulc_{city}_prithvi_{year}.tif") as src:
        pred = src.read(1).astype(np.uint8)
        profile = (src.crs, src.transform, src.width, src.height, src.bounds)
    with rasterio.open(INPUTS[city][year]) as src:
        nan = np.zeros(pred.shape, bool)
        for b in range(1, src.count + 1):
            nan |= np.isnan(src.read(b))
    pred[nan] = NODATA
    return pred, profile


def to_wgs84(arr, profile):
    crs, transform, w, h, bounds = profile
    dst_t, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, *bounds)
    dst = np.full((dh, dw), NODATA, np.uint8)
    reproject(arr, dst, src_transform=transform, src_crs=crs, dst_transform=dst_t, dst_crs="EPSG:4326",
              resampling=Resampling.nearest, src_nodata=NODATA, dst_nodata=NODATA)
    b = {"south": dst_t.f + dh * dst_t.e, "west": dst_t.c, "north": dst_t.f, "east": dst_t.c + dw * dst_t.a}
    return dst, b


def colorize(arr):
    rgba = np.zeros(arr.shape + (4,), np.uint8)
    for c, col in enumerate(COLORS):
        m = arr == c
        rgba[m, :3] = col
        rgba[m, 3] = 255
    return rgba


stats, bounds_all = {}, {}
for city in ("lahore", "gujranwala"):
    (OUT / city).mkdir(parents=True, exist_ok=True)
    preds = {}
    for year in (2018, 2021, 2026):
        pred, prof = load(city, year)
        preds[year] = pred
        w84, b = to_wgs84(pred, prof)
        Image.fromarray(colorize(w84)).save(OUT / city / f"web_{city}_{year}.png", optimize=True)
        bounds_all[f"{city}_{year}"] = b
        print(f"{city} {year}: {w84.shape[1]}x{w84.shape[0]}, NaN px {int((pred == NODATA).sum()):,}, bounds {b}")
    p18, p26 = preds[2018], preds[2026]
    valid = (p18 != NODATA) & (p26 != NODATA)
    veg18 = valid & (p18 == 0)
    change = np.full(p18.shape, NODATA, np.uint8)
    change[veg18] = p26[veg18]
    w84, _ = to_wgs84(change, prof)
    Image.fromarray(colorize(w84)).save(OUT / city / f"web_deforestation_{city}.png", optimize=True)
    to = {k: int((veg18 & (p26 == c)).sum()) for k, c in [("still_veg_px", 0), ("to_buildup", 1), ("to_cropland", 2), ("to_water", 3), ("to_barren", 4)]}
    lost = int(veg18.sum()) - to["still_veg_px"]
    stats[city] = {"veg_2018_px": int(veg18.sum()), **to, "lost_px": lost, "pct_lost": round(100 * lost / int(veg18.sum()), 1)}
    print(city, stats[city])

(OUT / "vegetation_change_stats.json").write_text(json.dumps({"stats": stats, "bounds": bounds_all}, indent=2))
print("saved", OUT)
