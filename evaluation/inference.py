"""
LULC Lahore - Full-Image Inference & Change Detection
Runs trained models on full Sentinel-2 mosaics (2018 & 2024),
produces LULC maps, and computes land-use change statistics.

Usage:
    python evaluation/inference.py --model segformer --checkpoint checkpoints/segformer_seed123/best_model.pt
    python evaluation/inference.py --model deeplabv3 --checkpoint checkpoints/deeplabv3_seed456/best_model.pt
    python evaluation/inference.py --model prithvi --checkpoint checkpoints/prithvi_seed42/best_model.pt
    python evaluation/inference.py --all   # Run all 3 models
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Constants ──────────────────────────────────────────────────
PATCH_SIZE = 256
OVERLAP = 64  # Overlap between patches to reduce edge artifacts
NUM_CLASSES = 5
CLASS_NAMES = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]
CLASS_COLORS = [
    (34, 139, 34),    # Vegetation - green
    (220, 20, 60),    # Built-up - red
    (255, 215, 0),    # Cropland - gold
    (30, 144, 255),   # Water - blue
    (210, 180, 140),  # Barren/Other - tan
]
PRITHVI_BAND_INDICES = [0, 1, 2, 3, 4, 5]


def load_model(model_name, checkpoint_path, device, metadata):
    """Load a trained model from checkpoint."""
    num_classes = metadata["num_classes"]
    in_channels = metadata["num_bands"]

    if model_name in ["segformer", "deeplabv3"]:
        import segmentation_models_pytorch as smp
        if model_name == "segformer":
            model = smp.FPN(
                encoder_name="mit_b2", encoder_weights=None,
                in_channels=in_channels, classes=num_classes,
            )
        else:
            model = smp.DeepLabV3Plus(
                encoder_name="resnet101", encoder_weights=None,
                in_channels=in_channels, classes=num_classes,
            )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded {model_name} from {checkpoint_path}")

    elif model_name == "prithvi":
        # Import Prithvi model class
        from training.train_prithvi import (
            load_prithvi_backbone, PrithviSegmentation, adapt_input_to_11bands, setup_training_params
        )
        backbone, feat_dim, backbone_type = load_prithvi_backbone()
        adapt_input_to_11bands(backbone, backbone_type)
        model = PrithviSegmentation(backbone, feat_dim, backbone_type, num_classes)
        setup_training_params(model, lora_rank=64, unfreeze_last_n=8)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"  Loaded Prithvi from {checkpoint_path}")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    model.eval()
    return model


def load_raster(path):
    """Load a GeoTIFF as numpy array + geospatial metadata."""
    import rasterio
    with rasterio.open(path) as src:
        data = src.read()  # (C, H, W)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return data.astype(np.float32), profile, transform, crs


def normalize_image(image, metadata):
    """Normalize image using training-set band statistics."""
    means = np.array(metadata["band_means"]).reshape(-1, 1, 1)
    stds = np.array(metadata["band_stds"]).reshape(-1, 1, 1)
    return (image - means) / (stds + 1e-10)


@torch.no_grad()
def predict_full_image(model, image, model_name, device, batch_size=4):
    """
    Predict LULC for a full Sentinel-2 image using sliding window with overlap.
    Returns predicted class map (H, W) as uint8.
    """
    C, H, W = image.shape
    stride = PATCH_SIZE - OVERLAP

    # For Prithvi, we now use all 10 bands in v2
    if model_name == "prithvi":
        pass

    # Pad image so patches tile evenly
    pad_h = (stride - (H - PATCH_SIZE) % stride) % stride
    pad_w = (stride - (W - PATCH_SIZE) % stride) % stride
    image_padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    _, H_pad, W_pad = image_padded.shape

    # Accumulate predictions and count overlaps for averaging
    pred_sum = np.zeros((NUM_CLASSES, H_pad, W_pad), dtype=np.float32)
    count = np.zeros((H_pad, W_pad), dtype=np.float32)

    # Collect all patch coordinates
    coords = []
    for y in range(0, H_pad - PATCH_SIZE + 1, stride):
        for x in range(0, W_pad - PATCH_SIZE + 1, stride):
            coords.append((y, x))

    print(f"    Patches: {len(coords)} ({PATCH_SIZE}x{PATCH_SIZE}, stride={stride})")

    # Process in batches
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i+batch_size]
        patches = []
        for y, x in batch_coords:
            patch = image_padded[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patches.append(patch)

        batch = torch.from_numpy(np.stack(patches)).float().to(device)
        logits = model(batch)  # (B, C, H, W)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for j, (y, x) in enumerate(batch_coords):
            pred_sum[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += probs[j]
            count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

        if (i // batch_size) % 50 == 0:
            print(f"    Progress: {i}/{len(coords)} patches ({100*i/len(coords):.0f}%)")

    # Average overlapping predictions and take argmax
    pred_sum /= np.maximum(count[np.newaxis], 1)
    pred_map = pred_sum.argmax(axis=0)[:H, :W].astype(np.uint8)

    return pred_map


def save_lulc_map(pred_map, output_path, profile, transform, crs):
    """Save LULC prediction as a GeoTIFF."""
    import rasterio
    profile.update(
        count=1,
        dtype='uint8',
        nodata=255,
        compress='lzw',
    )
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(pred_map[np.newaxis].astype(np.uint8))
    print(f"    Saved: {output_path} ({pred_map.shape})")


def save_colored_map(pred_map, output_path):
    """Save a colored RGB visualization of the LULC map."""
    from PIL import Image
    H, W = pred_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(CLASS_COLORS):
        rgb[pred_map == cls_idx] = color
    Image.fromarray(rgb).save(output_path, quality=95)
    print(f"    Saved colored map: {output_path}")


def compute_class_stats(pred_map, pixel_area_m2=100):
    """Compute area statistics for each class."""
    stats = {}
    total_pixels = pred_map.size
    for i, name in enumerate(CLASS_NAMES):
        count = int((pred_map == i).sum())
        pct = 100 * count / total_pixels
        area_km2 = count * pixel_area_m2 / 1e6
        stats[name] = {"pixels": count, "percent": round(pct, 2), "area_km2": round(area_km2, 2)}
    return stats


def compute_change_detection(map_2018, map_2024):
    """Compute change matrix and transition statistics between two LULC maps."""
    H, W = map_2018.shape
    assert map_2018.shape == map_2024.shape, "Maps must have the same shape"

    # Transition matrix: from_class (rows) -> to_class (cols)
    transition_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for from_cls in range(NUM_CLASSES):
        for to_cls in range(NUM_CLASSES):
            transition_matrix[from_cls, to_cls] = int(
                ((map_2018 == from_cls) & (map_2024 == to_cls)).sum()
            )

    # Overall change
    changed_pixels = int((map_2018 != map_2024).sum())
    total_pixels = map_2018.size
    change_pct = 100 * changed_pixels / total_pixels

    # Per-class gain/loss
    changes = {}
    for i, name in enumerate(CLASS_NAMES):
        pixels_2018 = int((map_2018 == i).sum())
        pixels_2024 = int((map_2024 == i).sum())
        gain = int(transition_matrix[:, i].sum() - transition_matrix[i, i])
        loss = int(transition_matrix[i, :].sum() - transition_matrix[i, i])
        net = gain - loss
        changes[name] = {
            "pixels_2018": pixels_2018,
            "pixels_2024": pixels_2024,
            "gain": gain,
            "loss": loss,
            "net_change": net,
            "pct_change": round(100 * net / max(pixels_2018, 1), 2),
        }

    return {
        "transition_matrix": transition_matrix.tolist(),
        "class_names": CLASS_NAMES,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "change_percent": round(change_pct, 2),
        "per_class_changes": changes,
    }


def save_change_map(map_2018, map_2024, output_path):
    """Save a change detection RGB map highlighting areas of change."""
    from PIL import Image
    H, W = map_2018.shape
    rgb = np.full((H, W, 3), 200, dtype=np.uint8)  # Gray = no change

    changed = map_2018 != map_2024
    # Color changed areas by their 2024 class
    for cls_idx, color in enumerate(CLASS_COLORS):
        mask = changed & (map_2024 == cls_idx)
        rgb[mask] = color

    # No-change areas: darker version of their class
    for cls_idx, color in enumerate(CLASS_COLORS):
        mask = (~changed) & (map_2024 == cls_idx)
        rgb[mask] = [max(0, c - 100) for c in color]

    Image.fromarray(rgb).save(output_path, quality=95)
    print(f"    Saved change map: {output_path}")


def run_inference_for_model(model_name, checkpoint_path, metadata, device, output_dir):
    """Run full inference pipeline for one model."""
    print(f"\n{'='*60}")
    print(f"  INFERENCE: {model_name.upper()}")
    print(f"{'='*60}")

    model = load_model(model_name, checkpoint_path, device, metadata)
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Images to process
    images = {
        "2018": PROJECT_ROOT / "data" / "sentinel2" / "sentinel2_2018_10bands.tif",
        "2024": PROJECT_ROOT / "data" / "sentinel2" / "sentinel2_2024_10bands.tif",
    }

    pred_maps = {}
    for year, img_path in images.items():
        print(f"\n  Processing {year}...")
        print(f"    Loading: {img_path.name}")
        raw, profile, transform, crs = load_raster(img_path)
        print(f"    Shape: {raw.shape} | CRS: {crs}")

        # Normalize
        normalized = normalize_image(raw, metadata)

        # Predict
        print(f"    Running inference...")
        pred_map = predict_full_image(model, normalized, model_name, device,
                                       batch_size=2 if model_name == "prithvi" else 4)

        # Save outputs
        save_lulc_map(pred_map, model_dir / f"lulc_{year}.tif", profile, transform, crs)
        save_colored_map(pred_map, model_dir / f"lulc_{year}_colored.png")

        # Class stats
        stats = compute_class_stats(pred_map)
        with open(model_dir / f"stats_{year}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"    Class distribution ({year}):")
        for name, s in stats.items():
            print(f"      {name:<15}: {s['percent']:>6.2f}% ({s['area_km2']:.1f} km2)")

        pred_maps[year] = pred_map

    # Change detection
    print(f"\n  Computing change detection (2018 -> 2024)...")
    changes = compute_change_detection(pred_maps["2018"], pred_maps["2024"])

    with open(model_dir / "change_detection.json", 'w') as f:
        json.dump(changes, f, indent=2)

    save_change_map(pred_maps["2018"], pred_maps["2024"],
                    model_dir / "change_map.png")

    # Print change summary
    print(f"\n  CHANGE DETECTION SUMMARY ({model_name})")
    print(f"  {'='*50}")
    print(f"  Overall change: {changes['change_percent']:.1f}%")
    print(f"  {'Class':<15} {'2018':>10} {'2024':>10} {'Net':>10} {'%Change':>8}")
    print(f"  {'-'*55}")
    for name, c in changes["per_class_changes"].items():
        print(f"  {name:<15} {c['pixels_2018']:>10,} {c['pixels_2024']:>10,} "
              f"{c['net_change']:>+10,} {c['pct_change']:>+7.1f}%")

    # Transition matrix
    print(f"\n  Transition Matrix (rows=2018, cols=2024, millions of pixels):")
    print(f"  {'':>15}", end="")
    for name in CLASS_NAMES:
        print(f" {name[:8]:>10}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15}", end="")
        for j in range(NUM_CLASSES):
            val = changes["transition_matrix"][i][j] / 1e6
            print(f" {val:>10.2f}", end="")
        print()

    return pred_maps, changes


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load metadata
    with open(PROJECT_ROOT / "data" / "patches" / "metadata.json") as f:
        metadata = json.load(f)

    output_dir = PROJECT_ROOT / "results" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define model configs
    model_configs = {
        "segformer": PROJECT_ROOT / "checkpoints" / "segformer_seed123" / "best_model.pt",
        "deeplabv3": PROJECT_ROOT / "checkpoints" / "deeplabv3_seed456" / "best_model.pt",
        "prithvi": PROJECT_ROOT / "checkpoints" / "prithvi_v2_seed42" / "best_model.pt",
    }

    if args.all:
        models_to_run = list(model_configs.keys())
    else:
        models_to_run = [args.model]

    all_results = {}
    for model_name in models_to_run:
        ckpt = args.checkpoint if args.checkpoint else str(model_configs[model_name])
        if not Path(ckpt).exists():
            print(f"\n  SKIP {model_name}: checkpoint not found at {ckpt}")
            continue
        pred_maps, changes = run_inference_for_model(
            model_name, ckpt, metadata, device, output_dir
        )
        all_results[model_name] = changes

    # If multiple models, save comparison
    if len(all_results) > 1:
        comparison_path = output_dir / "model_comparison.json"
        # Save change percentages side-by-side
        comparison = {}
        for model_name, changes in all_results.items():
            comparison[model_name] = {
                "overall_change_pct": changes["change_percent"],
                "per_class": {
                    name: c["pct_change"]
                    for name, c in changes["per_class_changes"].items()
                }
            }
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  Model comparison saved: {comparison_path}")

    print(f"\n{'='*60}")
    print(f"  ALL INFERENCE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LULC Inference & Change Detection")
    parser.add_argument("--model", type=str, default="segformer",
                        choices=["segformer", "deeplabv3", "prithvi"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Run inference for all 3 models")
    args = parser.parse_args()
    main(args)
