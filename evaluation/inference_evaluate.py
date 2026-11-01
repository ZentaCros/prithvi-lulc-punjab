"""
LULC Unified Inference & Evaluation Pipeline (v2 - Fresh Start)
================================================================
- Runs 4 models: U-Net, SegFormer, DeepLabV3+, Prithvi v2
- Computes NDVI on-the-fly from 10-band TIFs (no new downloads needed)
- Max-Confidence Ensemble: picks highest softmax probability per pixel
- Validates 2021 predictions against ESA 2021 ground truth
- Computes change detection: 2018 vs 2026

Usage:
    python evaluation/inference_evaluate.py --city lahore
    python evaluation/inference_evaluate.py --city gujranwala
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Constants ──────────────────────────────────────────────────
PATCH_SIZE    = 256
OVERLAP       = 224   # stride=32 -> ~36 overlapping patches per pixel
NUM_CLASSES   = 5
CLASS_NAMES   = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]
CLASS_COLORS  = [
    (34,  139,  34),   # Vegetation  - green
    (220,  20,  60),   # Built-up    - red
    (255, 215,   0),   # Cropland    - gold
    ( 30, 144, 255),   # Water       - blue
    (210, 180, 140),   # Barren/Other- tan
]
NDVI_NIR_IDX = 6   # Band 8 NIR  (0-indexed in 10-band array)
NDVI_RED_IDX = 2   # Band 4 Red


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_raster(path):
    import rasterio
    with rasterio.open(path) as src:
        data    = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs       = src.crs
    return data.astype(np.float32), profile, transform, crs


def compute_ndvi(image_10band):
    """Compute NDVI from 10-band image and return as (1, H, W) scaled array."""
    nir   = image_10band[NDVI_NIR_IDX].astype(np.float32)
    red   = image_10band[NDVI_RED_IDX].astype(np.float32)
    denom = nir + red
    ndvi  = np.where(denom > 0, (nir - red) / denom, 0.0)
    ndvi  = np.clip(ndvi, -1.0, 1.0) * 10000.0   # match band scale
    return ndvi[np.newaxis]                         # (1, H, W)


def load_and_prepare_image(path, metadata):
    """Load 10-band TIF, compute NDVI, normalize to 11 bands."""
    raw, profile, transform, crs = load_raster(path)  # (10, H, W)
    ndvi = compute_ndvi(raw)
    image_11 = np.concatenate([raw, ndvi], axis=0)   # (11, H, W)

    means = np.array(metadata["band_means"]).reshape(-1, 1, 1)
    stds  = np.array(metadata["band_stds"]).reshape(-1, 1, 1)
    image_norm = (image_11 - means) / (stds + 1e-8)

    return image_norm, profile, transform, crs


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════

def load_model(model_name, checkpoint_path, device, num_classes=5, in_channels=11):
    import segmentation_models_pytorch as smp

    if model_name == "unet":
        model = smp.Unet(
            encoder_name="efficientnet-b4", encoder_weights=None,
            in_channels=in_channels, classes=num_classes,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    elif model_name == "segformer":
        model = smp.FPN(
            encoder_name="mit_b2", encoder_weights=None,
            in_channels=in_channels, classes=num_classes,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    elif model_name == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101", encoder_weights=None,
            in_channels=in_channels, classes=num_classes,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    elif model_name == "prithvi":
        from training.train_prithvi import (
            load_prithvi_backbone, PrithviSegmentation,
            adapt_input_to_11bands, setup_training_params
        )
        backbone, feat_dim, backbone_type = load_prithvi_backbone()
        adapt_input_to_11bands(backbone, backbone_type)
        model = PrithviSegmentation(backbone, feat_dim, backbone_type, num_classes)
        setup_training_params(model, lora_rank=64, unfreeze_last_n=8)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    model.eval()
    print(f"  [OK] Loaded {model_name} from {Path(checkpoint_path).parent.name}")
    return model


# ═══════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_full_image(model, image, model_name, device, batch_size=48):
    """Dense sliding-window inference with heavy overlap.
    Returns (NUM_CLASSES, H, W) softmax probabilities."""
    C, H, W = image.shape
    stride   = PATCH_SIZE - OVERLAP   # 32 pixels

    pad_h = (stride - (H - PATCH_SIZE) % stride) % stride
    pad_w = (stride - (W - PATCH_SIZE) % stride) % stride
    img_p = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, Hp, Wp = img_p.shape

    prob_sum = np.zeros((NUM_CLASSES, Hp, Wp), dtype=np.float32)
    count    = np.zeros((Hp, Wp),              dtype=np.float32)

    coords = [
        (y, x)
        for y in range(0, Hp - PATCH_SIZE + 1, stride)
        for x in range(0, Wp - PATCH_SIZE + 1, stride)
    ]

    print(f"      Total patches: {len(coords)} (stride={stride}, overlap={OVERLAP})")

    bs = 12 if model_name == "prithvi" else batch_size
    for i in range(0, len(coords), bs):
        batch_coords = coords[i:i + bs]
        patches = [img_p[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                   for y, x in batch_coords]
        batch  = torch.from_numpy(np.stack(patches)).float().to(device)
        logits = model(batch)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

        for j, (y, x) in enumerate(batch_coords):
            prob_sum[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE] += probs[j]
            count[y:y + PATCH_SIZE, x:x + PATCH_SIZE]       += 1

        done = min(i + bs, len(coords))
        if done % 500 < bs or done == len(coords):
            pct = 100 * done / len(coords)
            print(f"      {pct:.0f}% ({done}/{len(coords)} patches)")

    prob_sum /= np.maximum(count[np.newaxis], 1)
    return prob_sum[:, :H, :W]   # (NUM_CLASSES, H, W) — raw probabilities


# ═══════════════════════════════════════════════════════════════
# POST-PROCESSING
# ═══════════════════════════════════════════════════════════════

def postprocess_prediction(pred_map, median_size=5, min_area=25):
    """Clean up raw predictions with:
    1. Median filter to remove salt-and-pepper noise
    2. Minimum mapping unit: remove tiny isolated patches
    """
    from scipy.ndimage import median_filter, label

    # 1. Median filter
    cleaned = median_filter(pred_map.astype(np.int16), size=median_size).astype(np.uint8)

    # 2. Remove small isolated patches (morphological sieve)
    for cls in range(NUM_CLASSES):
        mask = cleaned == cls
        labeled, n_features = label(mask)
        for region_id in range(1, n_features + 1):
            region = labeled == region_id
            if region.sum() < min_area:
                # Replace with most common neighbor class
                from scipy.ndimage import binary_dilation
                dilated = binary_dilation(region, iterations=2)
                border = dilated & ~region
                if border.any():
                    neighbor_vals = cleaned[border]
                    replacement = np.bincount(neighbor_vals, minlength=NUM_CLASSES).argmax()
                    cleaned[region] = replacement

    return cleaned


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE & SAVING
# ═══════════════════════════════════════════════════════════════

def max_confidence_ensemble(prob_maps):
    """
    Max-Confidence Ensemble: for every pixel, pick the class that has the
    highest softmax probability across ALL models combined.
    prob_maps: list of (NUM_CLASSES, H, W) arrays
    Returns: (H, W) uint8 prediction
    """
    stacked  = np.stack(prob_maps, axis=0)  # (M, C, H, W)
    max_prob = stacked.max(axis=0)           # (C, H, W) — highest confidence per class
    return max_prob.argmax(axis=0).astype(np.uint8)


def save_colored_png(pred_map, out_path):
    from PIL import Image
    H, W  = pred_map.shape
    rgb   = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, color in enumerate(CLASS_COLORS):
        rgb[pred_map == cls] = color
    Image.fromarray(rgb).save(str(out_path), quality=95)


def save_geotiff(pred_map, out_path, profile, transform, crs):
    import rasterio
    profile.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(pred_map[np.newaxis].astype(np.uint8))


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def compute_metrics(pred, gt):
    """Overall accuracy, macro F1, mIoU vs ESA ground truth."""
    valid = gt != 255
    y_true = gt[valid].astype(int)
    y_pred = pred[valid].astype(int)

    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, [0.0] * NUM_CLASSES

    acc  = accuracy_score(y_true, y_pred) * 100
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100

    cm   = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    inter = np.diag(cm)
    union = cm.sum(1) + cm.sum(0) - inter
    iou_per = np.divide(inter, union,
                        out=np.zeros_like(inter, dtype=float),
                        where=union != 0)
    miou = float(np.nanmean(iou_per)) * 100

    return acc, f1, miou, [float(v * 100) for v in iou_per]


# ═══════════════════════════════════════════════════════════════
# CHANGE DETECTION
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LULC Inference & Evaluation Pipeline"
    )
    parser.add_argument("--city", required=True, choices=["lahore", "gujranwala"])
    parser.add_argument("--skip-years", nargs="*", default=[],
                        help="Years to skip (e.g. --skip-years 2018). "
                             "Ensemble TIFs for skipped years are loaded from disk.")
    args = parser.parse_args()
    skip_years = set(args.skip_years)

    city   = args.city
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  INFERENCE PIPELINE — {city.upper()}")
    print(f"  Device: {device}")
    print(f"{'='*65}")

    # ── Metadata ──
    with open(PROJECT_ROOT / "data" / "patches" / "metadata.json") as f:
        metadata = json.load(f)

    # ── Paths ──
    inf_dir  = PROJECT_ROOT / "data" / "inference"
    out_dir  = PROJECT_ROOT / "results" / "pipeline" / city
    out_dir.mkdir(parents=True, exist_ok=True)

    if city == "lahore":
        img_paths = {
            "2018": inf_dir / "lahore" / "sentinel2_2018_10bands.tif",
            "2021": inf_dir / "lahore" / "sentinel2_2021_10bands.tif",
            "2026": inf_dir / "lahore" / "sentinel2_2026_10bands.tif",
        }
        gt_path = PROJECT_ROOT / "data" / "labels" / "worldcover_2021_5class.tif"
    else:
        img_paths = {
            "2018": inf_dir / "gujranwala" / "gujranwala_sentinel2_2018_10bands.tif",
            "2021": inf_dir / "gujranwala" / "gujranwala_sentinel2_2021_10bands.tif",
            "2026": inf_dir / "gujranwala" / "gujranwala_sentinel2_2026_10bands.tif",
        }
        gt_path = inf_dir / "gujranwala" / "worldcover_2021_5class_gujranwala.tif"

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    models_info = {
        "unet":      ckpt_dir / "unet_seed42"      / "best_model.pt",
        "segformer": ckpt_dir / "segformer_seed42"  / "best_model.pt",
        "deeplabv3": ckpt_dir / "deeplabv3_seed42"  / "best_model.pt",
        "prithvi":   ckpt_dir / "prithvi_v2_seed42" / "best_model.pt",
    }

    # ── Load Models ──
    print("\nLoading models...")
    loaded = {}
    for mname, mpath in models_info.items():
        if mpath.exists():
            loaded[mname] = load_model(mname, str(mpath), device,
                                       num_classes=metadata["num_classes"])
        else:
            print(f"  [SKIP] {mname}: checkpoint not found at {mpath}")

    # ── Load Ground Truth ──
    gt_raw, _, _, _ = load_raster(gt_path)
    gt_map = gt_raw[0].astype(np.uint8)

    # ── Per-year inference ──
    pred_argmax = {}   # year -> model_name -> (H, W) uint8
    pred_probs  = {}   # year -> model_name -> (C, H, W) float32
    evaluations = {}   # model_name -> metrics dict (only 2021)

    for year, img_path in img_paths.items():
        # Skip years that are already completed (resume support)
        if year in skip_years:
            ens_tif = out_dir / f"lulc_{city}_ensemble_{year}.tif"
            if ens_tif.exists():
                print(f"\n--- {year} [SKIPPED — loading from disk] ---")
                ens_data, _, _, _ = load_raster(ens_tif)
                pred_argmax[year] = {"ensemble": ens_data[0].astype(np.uint8)}
                pred_probs[year] = {}
                continue
            else:
                print(f"\n--- {year} [skip requested but no TIF found, running] ---")
        print(f"\n--- {year} ---")
        image, profile, transform, crs = load_and_prepare_image(img_path, metadata)
        H, W = image.shape[1], image.shape[2]

        # Crop GT to match image size if needed
        if year == "2021":
            gt = gt_map[:H, :W]

        pred_argmax[year] = {}
        pred_probs[year]  = {}

        for mname, model in loaded.items():
            print(f"  Predicting with {mname}...")
            probs = predict_full_image(model, image, mname, device)
            pred_raw = probs.argmax(axis=0).astype(np.uint8)

            # Evaluate 2021 vs ESA on RAW predictions (before post-processing)
            if year == "2021":
                acc, f1, miou, iou_pc = compute_metrics(pred_raw, gt)
                evaluations[mname] = {
                    "Accuracy": round(acc, 2),
                    "F1":       round(f1, 2),
                    "mIoU":     round(miou, 2),
                    "per_class_IoU": {CLASS_NAMES[i]: round(v, 2)
                                      for i, v in enumerate(iou_pc)}
                }

            # Post-process AFTER evaluation for clean visualization
            print(f"    Post-processing...")
            pred = postprocess_prediction(pred_raw, median_size=5, min_area=25)

            pred_probs[year][mname]  = probs
            pred_argmax[year][mname] = pred

            # Save post-processed maps for dashboard
            save_colored_png(pred, out_dir / f"lulc_{city}_{mname}_{year}.png")

        # Max-Confidence Ensemble
        print(f"  Building Max-Confidence Ensemble...")
        prob_list     = [pred_probs[year][m] for m in loaded]
        ens_raw       = max_confidence_ensemble(prob_list)

        # Evaluate ensemble on RAW predictions
        if year == "2021":
            acc, f1, miou, iou_pc = compute_metrics(ens_raw, gt)
            evaluations["ensemble"] = {
                "Accuracy": round(acc, 2),
                "F1":       round(f1, 2),
                "mIoU":     round(miou, 2),
                "per_class_IoU": {CLASS_NAMES[i]: round(v, 2)
                                  for i, v in enumerate(iou_pc)}
            }

        # Post-process ensemble for visualization
        print(f"    Post-processing ensemble...")
        ens_pred      = postprocess_prediction(ens_raw, median_size=5, min_area=25)
        pred_argmax[year]["ensemble"] = ens_pred

        save_colored_png(ens_pred, out_dir / f"lulc_{city}_ensemble_{year}.png")
        save_geotiff(ens_pred, out_dir / f"lulc_{city}_ensemble_{year}.tif",
                     profile, transform, crs)


    # ── Evaluation Table ──
    print(f"\n{'='*65}")
    print(f"  EVALUATION vs ESA 2021 — {city.upper()}")
    print(f"{'='*65}")
    header = f"| {'Model':<15} | {'Accuracy':>10} | {'F1':>8} | {'mIoU':>8} |"
    sep    = f"|{'-'*17}|{'-'*12}|{'-'*10}|{'-'*10}|"
    print(header); print(sep)

    md_lines = [
        f"# Evaluation Metrics — {city.capitalize()} (vs ESA 2021)\n",
        header, sep
    ]

    model_order = ["unet", "segformer", "deeplabv3", "prithvi", "ensemble"]
    model_labels = {
        "unet":      "U-Net (B4)",
        "segformer": "SegFormer",
        "deeplabv3": "DeepLabV3+",
        "prithvi":   "Prithvi v2",
        "ensemble":  "**Ensemble**",
    }
    for m in model_order:
        if m in evaluations:
            e = evaluations[m]
            row = (f"| {model_labels[m]:<15} | "
                   f"{e['Accuracy']:>9.2f}% | "
                   f"{e['F1']:>7.2f}% | "
                   f"{e['mIoU']:>7.2f}% |")
            print(row)
            md_lines.append(row)

    with open(out_dir / f"evaluation_table_{city}.md", "w") as f:
        f.write("\n".join(md_lines))

    # Save full evaluation JSON
    with open(out_dir / f"evaluation_{city}.json", "w") as f:
        json.dump(evaluations, f, indent=2)

    # ── Change Detection 2018 → 2026 (using Ensemble) ──
    print(f"\n--- Change Detection (2018 -> 2026, Ensemble) ---")
    # Load from disk if not in memory (skipped years)
    for yr in ["2018", "2026"]:
        if yr not in pred_argmax or "ensemble" not in pred_argmax.get(yr, {}):
            ens_tif = out_dir / f"lulc_{city}_ensemble_{yr}.tif"
            if ens_tif.exists():
                print(f"  Loading {yr} ensemble from disk...")
                ens_data, _, _, _ = load_raster(ens_tif)
                if yr not in pred_argmax:
                    pred_argmax[yr] = {}
                pred_argmax[yr]["ensemble"] = ens_data[0].astype(np.uint8)
            else:
                print(f"  [ERROR] Cannot find ensemble TIF for {yr}")
                return
    cd = compute_change_detection(
        pred_argmax["2018"]["ensemble"],
        pred_argmax["2026"]["ensemble"]
    )
    for name, vals in cd.items():
        chg = vals["net_change_km2"]
        sign = "+" if chg >= 0 else ""
        print(f"  {name:<15}: {vals['area_2018_km2']:>8.2f} km2 -> "
              f"{vals['area_2026_km2']:>8.2f} km2  ({sign}{chg:.2f} km2)")

    with open(out_dir / f"change_detection_{city}_2018_2026.json", "w") as f:
        json.dump(cd, f, indent=2)

    print(f"\n  All outputs saved to: {out_dir}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
