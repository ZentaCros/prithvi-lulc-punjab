"""
Generate deforestation highlight maps.
Shows pixels that were Vegetation in 2018 but changed to another class by 2026.
Color-codes by what they became (Built-up = red, Cropland = gold, etc.)
Also generates a "vegetation persistence" map showing what remained green.
"""

import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

CLASS_NAMES = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]

# Deforestation map colors: what vegetation turned INTO
DEFORESTATION_COLORS = {
    # Still vegetation = bright green (persistence)
    "persisted":  (34, 197, 94),
    # Converted TO:
    1: (239,  68,  68),   # Built-up   = red
    2: (234, 179,   8),   # Cropland   = gold
    3: ( 59, 130, 246),   # Water      = blue
    4: (210, 180, 140),   # Barren     = tan
    # Not vegetation in 2018 = transparent (we'll use alpha)
}


def load_prediction_tif(path):
    """Load a prediction GeoTIFF (single-band uint8 class map)."""
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8)


def generate_deforestation_map(pred_2018, pred_2026, out_path):
    """
    Generate a deforestation overlay PNG.
    - Transparent: pixels that were NOT vegetation in 2018
    - Green: pixels that stayed vegetation
    - Red/Gold/Blue/Tan: pixels that changed from vegetation to another class
    """
    H, W = pred_2018.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    # Vegetation class = 0
    was_veg = pred_2018 == 0
    still_veg = was_veg & (pred_2026 == 0)
    lost_veg = was_veg & (pred_2026 != 0)

    # Persisted vegetation (semi-transparent green)
    rgba[still_veg] = (*DEFORESTATION_COLORS["persisted"], 160)

    # Lost vegetation — color by what it became
    for target_cls in [1, 2, 3, 4]:
        mask = was_veg & (pred_2026 == target_cls)
        if mask.any():
            rgba[mask] = (*DEFORESTATION_COLORS[target_cls], 220)

    Image.fromarray(rgba).save(str(out_path))
    
    # Stats
    total_veg_2018 = was_veg.sum()
    total_lost = lost_veg.sum()
    pct_lost = (total_lost / max(total_veg_2018, 1)) * 100
    
    print(f"  Vegetation 2018: {total_veg_2018:,} px")
    print(f"  Still Vegetation 2026: {still_veg.sum():,} px")
    print(f"  Lost: {total_lost:,} px ({pct_lost:.1f}%)")
    for c in [1, 2, 3, 4]:
        cnt = (was_veg & (pred_2026 == c)).sum()
        if cnt > 0:
            print(f"    -> {CLASS_NAMES[c]}: {cnt:,} px")
    
    return {
        "veg_2018_px": int(total_veg_2018),
        "still_veg_px": int(still_veg.sum()),
        "lost_px": int(total_lost),
        "pct_lost": round(pct_lost, 1),
        "to_buildup": int((was_veg & (pred_2026 == 1)).sum()),
        "to_cropland": int((was_veg & (pred_2026 == 2)).sum()),
        "to_water": int((was_veg & (pred_2026 == 3)).sum()),
        "to_barren": int((was_veg & (pred_2026 == 4)).sum()),
    }


def main():
    import json
    
    for city in ["lahore", "gujranwala"]:
        print(f"\n=== {city.upper()} ===")
        
        pipe_dir = PROJECT_ROOT / "results" / "pipeline" / city
        
        # Use ensemble predictions (best overall)
        tif_2018 = pipe_dir / f"lulc_{city}_ensemble_2018.tif"
        tif_2026 = pipe_dir / f"lulc_{city}_ensemble_2026.tif"
        
        if not tif_2018.exists() or not tif_2026.exists():
            print(f"  [SKIP] Missing TIF files for {city}")
            continue
        
        pred_2018 = load_prediction_tif(tif_2018)
        pred_2026 = load_prediction_tif(tif_2026)
        
        # Crop to same size
        h = min(pred_2018.shape[0], pred_2026.shape[0])
        w = min(pred_2018.shape[1], pred_2026.shape[1])
        pred_2018 = pred_2018[:h, :w]
        pred_2026 = pred_2026[:h, :w]
        
        out_path = pipe_dir / f"deforestation_{city}_2018_2026.png"
        stats = generate_deforestation_map(pred_2018, pred_2026, out_path)
        print(f"  Saved: {out_path.name}")
        
        # Save stats JSON
        stats_path = pipe_dir / f"deforestation_stats_{city}.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: {stats_path.name}")


if __name__ == "__main__":
    main()
