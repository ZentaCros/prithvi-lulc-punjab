"""
Re-run Prithvi inference with fix6 checkpoint on all cities/years.
Updates: full-city eval (Table 3), per-class zero-shot (Table 4b),
change detection (Table 5), and dry-season eval (Table 6).

Usage:
    python -I evaluation/rerun_prithvi_inference.py
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PATCH_SIZE = 256
OVERLAP = 224
NUM_CLASSES = 5
CLASS_NAMES = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]
NDVI_NIR_IDX = 6
NDVI_RED_IDX = 2

NEW_CKPT = PROJECT_ROOT / "checkpoints" / "prithvi_v2_fix6_seed42" / "best_model.pt"
OUT_DIR = PROJECT_ROOT / "results" / "pipeline_fix6"


def load_raster(path):
    import rasterio
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return data.astype(np.float32), profile, transform, crs


def compute_ndvi(image_10band):
    nir = image_10band[NDVI_NIR_IDX].astype(np.float32)
    red = image_10band[NDVI_RED_IDX].astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    ndvi = np.clip(ndvi, -1.0, 1.0) * 10000.0
    return ndvi[np.newaxis]


def load_and_prepare_image(path, metadata):
    raw, profile, transform, crs = load_raster(path)
    ndvi = compute_ndvi(raw)
    image_11 = np.concatenate([raw, ndvi], axis=0)
    means = np.array(metadata["band_means"]).reshape(-1, 1, 1)
    stds = np.array(metadata["band_stds"]).reshape(-1, 1, 1)
    image_norm = (image_11 - means) / (stds + 1e-8)
    return image_norm, profile, transform, crs


def load_prithvi_model(checkpoint_path, device, num_classes=5):
    from training.train_prithvi import (
        load_prithvi_backbone, PrithviSegmentation,
        adapt_input_to_11bands, setup_training_params
    )
    backbone, feat_dim, backbone_type = load_prithvi_backbone()
    adapt_input_to_11bands(backbone, backbone_type)
    model = PrithviSegmentation(backbone, feat_dim, backbone_type, num_classes)
    setup_training_params(model, lora_rank=64, unfreeze_last_n=8)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_full_image(model, image, device, batch_size=12):
    C, H, W = image.shape
    stride = PATCH_SIZE - OVERLAP

    pad_h = (stride - (H - PATCH_SIZE) % stride) % stride
    pad_w = (stride - (W - PATCH_SIZE) % stride) % stride
    img_p = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, Hp, Wp = img_p.shape

    prob_sum = np.zeros((NUM_CLASSES, Hp, Wp), dtype=np.float32)
    count = np.zeros((Hp, Wp), dtype=np.float32)

    coords = [
        (y, x)
        for y in range(0, Hp - PATCH_SIZE + 1, stride)
        for x in range(0, Wp - PATCH_SIZE + 1, stride)
    ]

    print(f"    Patches: {len(coords)} (stride={stride})")

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        patches = [img_p[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                   for y, x in batch_coords]
        batch = torch.from_numpy(np.stack(patches)).float().to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for j, (y, x) in enumerate(batch_coords):
            prob_sum[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += probs[j]
            count[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1

        done = min(i + batch_size, len(coords))
        if done % 500 < batch_size or done == len(coords):
            print(f"    {100 * done / len(coords):.0f}% ({done}/{len(coords)})")

    prob_sum /= np.maximum(count[np.newaxis], 1)
    pred_map = prob_sum[:, :H, :W].argmax(axis=0).astype(np.uint8)
    return pred_map


def compute_metrics(pred, gt):
    valid = gt != 255
    y_true = gt[valid].astype(int)
    y_pred = pred[valid].astype(int)
    if len(y_true) == 0:
        return {}

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    inter = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - inter
    iou_per = np.divide(inter, union,
                        out=np.zeros_like(inter, dtype=float),
                        where=union != 0)
    miou = float(np.nanmean(iou_per)) * 100

    return {
        "OA": round(acc, 2),
        "F1": round(f1, 2),
        "mIoU": round(miou, 2),
        "per_class_IoU": {CLASS_NAMES[i]: round(float(v * 100), 2) for i, v in enumerate(iou_per)},
    }


def compute_change_detection(map_2018, map_2026, pixel_area_m2=100):
    result = {}
    for i, name in enumerate(CLASS_NAMES):
        a2018 = int((map_2018 == i).sum()) * pixel_area_m2 / 1e6
        a2026 = int((map_2026 == i).sum()) * pixel_area_m2 / 1e6
        result[name] = {
            "area_2018_km2": round(a2018, 2),
            "area_2026_km2": round(a2026, 2),
            "net_change_km2": round(a2026 - a2018, 2),
        }
    return result


def save_geotiff(pred_map, out_path, profile, transform, crs):
    import rasterio
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(pred_map[np.newaxis].astype(np.uint8))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {NEW_CKPT}")

    with open(PROJECT_ROOT / "data" / "patches" / "metadata.json") as f:
        metadata = json.load(f)

    print("\nLoading Prithvi model (fix6 checkpoint)...")
    model = load_prithvi_model(str(NEW_CKPT), device)
    print("  Model loaded.\n")

    inf_dir = PROJECT_ROOT / "data" / "inference"
    all_results = {}

    # ── Task 1: Lahore full-city 2021 (Table 3 Prithvi Lahore row) ──
    print("=" * 65)
    print("  TASK 1: Lahore 2021 full-city eval")
    print("=" * 65)
    lahore_out = OUT_DIR / "lahore"
    lahore_out.mkdir(parents=True, exist_ok=True)

    img, profile, transform, crs = load_and_prepare_image(
        inf_dir / "lahore" / "sentinel2_2021_10bands.tif", metadata)
    gt_raw, _, _, _ = load_raster(
        PROJECT_ROOT / "data" / "labels" / "worldcover_2021_5class.tif")
    gt = gt_raw[0, :img.shape[1], :img.shape[2]].astype(np.uint8)

    pred_lahore_2021 = predict_full_image(model, img, device)
    save_geotiff(pred_lahore_2021, lahore_out / "lulc_lahore_prithvi_2021.tif",
                 profile, transform, crs)
    metrics_lahore = compute_metrics(pred_lahore_2021, gt)
    all_results["lahore_2021"] = metrics_lahore
    print(f"  Lahore 2021: OA={metrics_lahore['OA']}%, mIoU={metrics_lahore['mIoU']}%")
    print(f"  Per-class IoU: {metrics_lahore['per_class_IoU']}")

    # ── Task 2: Gujranwala full-city 2021 (Table 3 Prithvi Gujranwala row) ──
    print(f"\n{'=' * 65}")
    print("  TASK 2: Gujranwala 2021 full-city eval (zero-shot)")
    print("=" * 65)
    guj_out = OUT_DIR / "gujranwala"
    guj_out.mkdir(parents=True, exist_ok=True)

    img, profile, transform, crs = load_and_prepare_image(
        inf_dir / "gujranwala" / "gujranwala_sentinel2_2021_10bands.tif", metadata)
    gt_raw, _, _, _ = load_raster(
        inf_dir / "gujranwala" / "worldcover_2021_5class_gujranwala.tif")
    gt = gt_raw[0, :img.shape[1], :img.shape[2]].astype(np.uint8)

    pred_guj_2021 = predict_full_image(model, img, device)
    save_geotiff(pred_guj_2021, guj_out / "lulc_gujranwala_prithvi_2021.tif",
                 profile, transform, crs)
    metrics_guj = compute_metrics(pred_guj_2021, gt)
    all_results["gujranwala_2021"] = metrics_guj
    print(f"  Gujranwala 2021: OA={metrics_guj['OA']}%, mIoU={metrics_guj['mIoU']}%")
    print(f"  Per-class IoU: {metrics_guj['per_class_IoU']}")

    # ── Task 3: Lahore change detection (2018 & 2026) ──
    print(f"\n{'=' * 65}")
    print("  TASK 3: Lahore change detection (2018 -> 2026)")
    print("=" * 65)
    for year in ["2018", "2026"]:
        print(f"  Processing {year}...")
        img, profile, transform, crs = load_and_prepare_image(
            inf_dir / "lahore" / f"sentinel2_{year}_10bands.tif", metadata)
        pred = predict_full_image(model, img, device)
        save_geotiff(pred, lahore_out / f"lulc_lahore_prithvi_{year}.tif",
                     profile, transform, crs)
        if year == "2018":
            pred_lahore_2018 = pred
        else:
            pred_lahore_2026 = pred

    cd_lahore = compute_change_detection(pred_lahore_2018, pred_lahore_2026)
    all_results["lahore_change"] = cd_lahore
    print("  Lahore change detection:")
    for name, vals in cd_lahore.items():
        chg = vals["net_change_km2"]
        print(f"    {name:<15}: {vals['area_2018_km2']:>8.2f} -> {vals['area_2026_km2']:>8.2f} km2 ({'+' if chg >= 0 else ''}{chg:.2f})")

    # ── Task 4: Gujranwala change detection (2018 & 2026) ──
    print(f"\n{'=' * 65}")
    print("  TASK 4: Gujranwala change detection (2018 -> 2026)")
    print("=" * 65)
    for year in ["2018", "2026"]:
        print(f"  Processing {year}...")
        img, profile, transform, crs = load_and_prepare_image(
            inf_dir / "gujranwala" / f"gujranwala_sentinel2_{year}_10bands.tif", metadata)
        pred = predict_full_image(model, img, device)
        save_geotiff(pred, guj_out / f"lulc_gujranwala_prithvi_{year}.tif",
                     profile, transform, crs)
        if year == "2018":
            pred_guj_2018 = pred
        else:
            pred_guj_2026 = pred

    cd_guj = compute_change_detection(pred_guj_2018, pred_guj_2026)
    all_results["gujranwala_change"] = cd_guj
    print("  Gujranwala change detection:")
    for name, vals in cd_guj.items():
        chg = vals["net_change_km2"]
        print(f"    {name:<15}: {vals['area_2018_km2']:>8.2f} -> {vals['area_2026_km2']:>8.2f} km2 ({'+' if chg >= 0 else ''}{chg:.2f})")

    # ── Task 5: Dry-season Gujranwala eval ──
    print(f"\n{'=' * 65}")
    print("  TASK 5: Gujranwala dry-season eval (seasonal robustness)")
    print("=" * 65)
    dryseason_path = inf_dir / "gujranwala" / "gujranwala_sentinel2_2021_dryseason_10bands.tif"
    img, profile, transform, crs = load_and_prepare_image(dryseason_path, metadata)
    gt_raw, _, _, _ = load_raster(
        inf_dir / "gujranwala" / "worldcover_2021_5class_gujranwala.tif")
    gt = gt_raw[0, :img.shape[1], :img.shape[2]].astype(np.uint8)

    pred_dry = predict_full_image(model, img, device)
    metrics_dry = compute_metrics(pred_dry, gt)
    all_results["gujranwala_dryseason"] = metrics_dry
    print(f"  Dry-season: OA={metrics_dry['OA']}%, mIoU={metrics_dry['mIoU']}%")
    print(f"  Per-class IoU: {metrics_dry['per_class_IoU']}")

    # ── Save all results ──
    summary_path = OUT_DIR / "fix6_inference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 65}")
    print("  ALL INFERENCE COMPLETE")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'=' * 65}")

    # ── Print comparison table ──
    print(f"\n  PAPER UPDATES NEEDED:")
    print(f"  Table 3 (Lahore): OA={metrics_lahore['OA']}, mIoU={metrics_lahore['mIoU']}")
    print(f"  Table 3 (Gujranwala): OA={metrics_guj['OA']}, mIoU={metrics_guj['mIoU']}")
    print(f"  Table 6 (dry-season): OA={metrics_dry['OA']}, mIoU={metrics_dry['mIoU']}")
    print(f"  Lahore built-up change: {cd_lahore['Built-up']['net_change_km2']:+.2f} km2")
    print(f"  Lahore veg change: {cd_lahore['Vegetation']['net_change_km2']:+.2f} km2")


if __name__ == "__main__":
    main()
