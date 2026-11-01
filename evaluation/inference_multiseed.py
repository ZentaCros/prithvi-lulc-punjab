"""
Full-City Inference for All Models × All Seeds

Runs full-city inference on Lahore and Gujranwala for every trained checkpoint,
then aggregates results into mean±std tables.

Usage:
    python evaluation/inference_multiseed.py --city lahore
    python evaluation/inference_multiseed.py --city gujranwala
    python evaluation/inference_multiseed.py --city both
    python evaluation/inference_multiseed.py --dry_run
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.inference import (
    load_raster, normalize_image, predict_full_image,
    save_lulc_map, save_colored_map, compute_class_stats,
    compute_change_detection, NUM_CLASSES, CLASS_NAMES,
)

MODELS_SMP = ["unet", "segformer", "deeplabv3"]
SEEDS = [42, 123, 7]

CITY_CONFIGS = {
    "lahore": {
        "images": {
            "2018": "data/inference/lahore/sentinel2_2018_10bands.tif",
            "2021": "data/inference/lahore/sentinel2_2021_10bands.tif",
            "2024": "data/inference/lahore/sentinel2_2024_10bands.tif",
            "2026": "data/inference/lahore/sentinel2_2026_10bands.tif",
        },
        "label": "data/labels/worldcover_2021_5class.tif",
    },
    "gujranwala": {
        "images": {
            "2018": "data/inference/gujranwala/gujranwala_sentinel2_2018_10bands.tif",
            "2021": "data/inference/gujranwala/gujranwala_sentinel2_2021_10bands.tif",
            "2024": "data/inference/gujranwala/gujranwala_sentinel2_2024_10bands.tif",
            "2026": "data/inference/gujranwala/gujranwala_sentinel2_2026_10bands.tif",
        },
        "label": "data/inference/gujranwala/worldcover_2021_5class_gujranwala.tif",
    },
}


def load_smp_model(model_name, checkpoint_path, device, num_bands, num_classes):
    """Load an SMP model."""
    import segmentation_models_pytorch as smp
    if model_name == "unet":
        model = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None,
                         in_channels=num_bands, classes=num_classes)
    elif model_name == "segformer":
        model = smp.FPN(encoder_name="mit_b2", encoder_weights=None,
                        in_channels=num_bands, classes=num_classes)
    elif model_name == "deeplabv3":
        model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None,
                                   in_channels=num_bands, classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_prithvi_model(checkpoint_path, device, num_classes):
    """Load Prithvi model with LoRA."""
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


def evaluate_against_worldcover(pred_map, label_path):
    """Evaluate prediction against WorldCover ground truth."""
    from sklearn.metrics import f1_score, jaccard_score
    label, _, _, _ = load_raster(label_path)
    label = label[0]  # (H, W)

    # Crop to matching size
    h = min(pred_map.shape[0], label.shape[0])
    w = min(pred_map.shape[1], label.shape[1])
    pred = pred_map[:h, :w].flatten()
    gt = label[:h, :w].astype(np.int64).flatten()

    # Filter valid pixels
    valid = gt < NUM_CLASSES
    pred = pred[valid]
    gt = gt[valid]

    oa = float((pred == gt).sum() / len(gt))
    miou = float(np.mean(jaccard_score(gt, pred, average=None,
                                        labels=range(NUM_CLASSES), zero_division=0)))
    mf1 = float(np.mean(f1_score(gt, pred, average=None,
                                   labels=range(NUM_CLASSES), zero_division=0)))
    iou_per_class = jaccard_score(gt, pred, average=None,
                                   labels=range(NUM_CLASSES), zero_division=0)

    return {
        "oa": oa, "miou": miou, "mf1": mf1,
        "iou_per_class": [float(x) for x in iou_per_class],
    }


def find_checkpoint(model, seed):
    """Find the best_model.pt checkpoint for a model+seed."""
    if model == "prithvi":
        path = PROJECT_ROOT / "checkpoints" / f"prithvi_v2_seed{seed}" / "best_model.pt"
    else:
        path = PROJECT_ROOT / "checkpoints" / f"{model}_seed{seed}" / "best_model.pt"
    return path if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Full-City Inference")
    parser.add_argument("--city", default="both", choices=["lahore", "gujranwala", "both"])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--year", default="2021", help="Year for evaluation against WorldCover")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load metadata
    with open(PROJECT_ROOT / "data" / "patches" / "metadata.json") as f:
        metadata = json.load(f)

    models = args.models or (MODELS_SMP + ["prithvi"])
    cities = ["lahore", "gujranwala"] if args.city == "both" else [args.city]

    # Check what checkpoints exist
    print("\nCheckpoints found:")
    available_runs = []
    for model in models:
        for seed in SEEDS:
            ckpt = find_checkpoint(model, seed)
            status = "OK" if ckpt else "MISSING"
            print(f"  {model:<12} seed {seed}: {status}")
            if ckpt:
                available_runs.append((model, seed, ckpt))

    if args.dry_run:
        print(f"\n  Would run {len(available_runs)} × {len(cities)} city = "
              f"{len(available_runs) * len(cities)} inference jobs")
        return

    # Run inference
    all_city_results = {}
    for city in cities:
        print(f"\n{'=' * 60}")
        print(f"  CITY: {city.upper()}")
        print(f"{'=' * 60}")

        config = CITY_CONFIGS[city]
        year = args.year
        img_path = PROJECT_ROOT / config["images"].get(year, config["images"]["2021"])
        label_path = PROJECT_ROOT / config["label"]

        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            continue

        # Load and normalize image once
        print(f"  Loading {img_path.name}...")
        raw, profile, transform, crs = load_raster(img_path)
        normalized = normalize_image(raw, metadata)

        city_results = {}
        for model, seed, ckpt in available_runs:
            print(f"\n  Running {model} seed {seed}...")

            # Load model
            if model == "prithvi":
                net = load_prithvi_model(ckpt, device, metadata["num_classes"])
                batch_size = 2
            else:
                net = load_smp_model(model, ckpt, device,
                                     metadata["num_bands"], metadata["num_classes"])
                batch_size = 4

            # Predict
            pred_map = predict_full_image(net, normalized, model, device, batch_size=batch_size)

            # Save prediction
            out_dir = PROJECT_ROOT / "results" / "inference_multiseed" / city / f"{model}_seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_lulc_map(pred_map, out_dir / f"lulc_{year}.tif", profile, transform, crs)
            save_colored_map(pred_map, out_dir / f"lulc_{year}_colored.png")

            # Evaluate against WorldCover
            if label_path.exists():
                metrics = evaluate_against_worldcover(pred_map, str(label_path))
                with open(out_dir / f"metrics_{year}.json", 'w') as f:
                    json.dump(metrics, f, indent=2)

                print(f"    OA: {metrics['oa']*100:.2f}%  mIoU: {metrics['miou']*100:.2f}%  "
                      f"mF1: {metrics['mf1']*100:.2f}%")

                key = f"{model}_seed{seed}"
                city_results[key] = metrics

            # Free GPU memory
            del net
            torch.cuda.empty_cache()

        all_city_results[city] = city_results

        # Aggregate per-model stats
        print(f"\n  AGGREGATE RESULTS — {city.upper()} ({year})")
        print(f"  {'Model':<25} {'OA':>12} {'mIoU':>12} {'mF1':>12}")
        print(f"  {'-' * 63}")
        for model in models:
            model_metrics = [v for k, v in city_results.items() if k.startswith(model)]
            if not model_metrics:
                continue
            oa_vals = [m["oa"] for m in model_metrics]
            miou_vals = [m["miou"] for m in model_metrics]
            mf1_vals = [m["mf1"] for m in model_metrics]
            name = model.replace("_", " ").title()
            print(f"  {name:<25} "
                  f"{np.mean(oa_vals)*100:.2f}±{np.std(oa_vals, ddof=1)*100:.2f}  "
                  f"{np.mean(miou_vals)*100:.2f}±{np.std(miou_vals, ddof=1)*100:.2f}  "
                  f"{np.mean(mf1_vals)*100:.2f}±{np.std(mf1_vals, ddof=1)*100:.2f}")

    # Save all results
    output_path = PROJECT_ROOT / "results" / "inference_multiseed" / "all_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_city_results, f, indent=2)
    print(f"\n  All results saved to: {output_path}")


if __name__ == "__main__":
    main()
