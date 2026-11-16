"""
Table 5 areas, percentage changes and 2018->2026 transition matrices from
the final Prithvi maps (results/pipeline_fix6_fullcov). Nodata (255)
pixels, and pixels nodata in either year, are excluded.
Writes change_stats.json next to the maps.
"""
import json
import numpy as np
import rasterio
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
MAPS = Path(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else PROJECT / "results" / "pipeline_fix6_fullcov"
NAMES = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]

out = {}
for city in ("lahore", "gujranwala"):
    with rasterio.open(MAPS / city / f"lulc_{city}_prithvi_2018.tif") as s:
        a = s.read(1); px_km2 = abs(s.res[0] * s.res[1]) / 1e6
    with rasterio.open(MAPS / city / f"lulc_{city}_prithvi_2026.tif") as s:
        b = s.read(1)
    H, W = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:H, :W], b[:H, :W]
    valid = (a < 5) & (b < 5)
    trans = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            trans[i, j] = ((a == i) & (b == j) & valid).sum() * px_km2
    area18, area26 = trans.sum(1), trans.sum(0)
    rec = {"pixel_km2": px_km2, "valid_km2": float(valid.sum() * px_km2), "classes": {}}
    print(f"\n{city}: valid {valid.sum()*px_km2:.1f} km2 (pixel {px_km2*1e6:.0f} m2)")
    for c, n in enumerate(NAMES):
        d = area26[c] - area18[c]
        rec["classes"][n] = {"2018": round(area18[c], 2), "2026": round(area26[c], 2),
                             "delta": round(d, 2), "pct": round(100 * d / area18[c], 2)}
        print(f"  {n:13s} {area18[c]:8.1f} {area26[c]:8.1f} {d:+7.1f} ({100*d/area18[c]:+.1f}%)")
    rec["transition_km2"] = [[round(x, 2) for x in r] for r in trans]
    gain = area26[1] - trans[1, 1]
    print(f"  new built-up (gross gain) {gain:.1f} km2")
    for i in (4, 0, 2, 3):
        print(f"    {NAMES[i]} -> Built-up {trans[i,1]:.1f} ({100*trans[i,1]/gain:.1f}% of gain; {100*trans[i,1]/area18[i]:.1f}% of 2018 {NAMES[i]})")
    print(f"  Barren -> Vegetation {trans[4,0]:.1f}, Barren -> Cropland {trans[4,2]:.1f}")
    print("  matrix rows=2018 cols=2026:")
    for i in range(5):
        print("   ", NAMES[i][:6].ljust(6), " ".join(f"{x:8.1f}" for x in trans[i]))
    out[city] = rec
(MAPS / "change_stats.json").write_text(json.dumps(out, indent=2))
