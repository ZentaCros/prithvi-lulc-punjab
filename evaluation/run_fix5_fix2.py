"""
Fix #5: Dry-season Gujranwala inference & evaluation
Fix #2: Per-model TIFs + Prithvi-only change detection

Usage:
    python -I evaluation/run_fix5_fix2.py --fix 5     # dry-season eval only
    python -I evaluation/run_fix5_fix2.py --fix 2     # Prithvi change detection only
    python -I evaluation/run_fix5_fix2.py --fix all    # both (default)
"""

import os, sys, json, argparse, time
import numpy as np
import torch
from pathlib import Path
from scipy.ndimage import median_filter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.inference_evaluate import (
    load_raster, load_and_prepare_image, load_model,
    predict_full_image, max_confidence_ensemble, save_geotiff,
    compute_metrics, compute_change_detection,
    CLASS_NAMES, NUM_CLASSES,
)


def light_postprocess(pred_map):
    """Median filter only - skips the slow connected-component sieve."""
    return median_filter(pred_map.astype(np.int16), size=5).astype(np.uint8)


def run_fix5(device, metadata):
    """Run all 4 models on dry-season Gujranwala 2021, evaluate vs WorldCover."""
    print("\n" + "=" * 65)
    print("  FIX #5: DRY-SEASON GUJRANWALA INFERENCE")
    print("=" * 65, flush=True)

    inf_dir = PROJECT_ROOT / "data" / "inference" / "gujranwala"
    dry_tif = inf_dir / "gujranwala_sentinel2_2021_dryseason_10bands.tif"
    gt_path = inf_dir / "worldcover_2021_5class_gujranwala.tif"
    out_dir = PROJECT_ROOT / "results" / "pipeline" / "gujranwala_dryseason"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not dry_tif.exists():
        print(f"  [ERROR] Dry-season TIF not found: {dry_tif}")
        return None
    if not gt_path.exists():
        print(f"  [ERROR] Ground truth not found: {gt_path}")
        return None

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    models_info = {
        "unet":      ckpt_dir / "unet_seed42" / "best_model.pt",
        "segformer": ckpt_dir / "segformer_seed42" / "best_model.pt",
        "deeplabv3": ckpt_dir / "deeplabv3_seed42" / "best_model.pt",
        "prithvi":   ckpt_dir / "prithvi_v2_seed42" / "best_model.pt",
    }

    print("\nLoading models...", flush=True)
    loaded = {}
    for mname, mpath in models_info.items():
        if mpath.exists():
            loaded[mname] = load_model(mname, str(mpath), device,
                                       num_classes=metadata["num_classes"])
        else:
            print(f"  [SKIP] {mname}: checkpoint not found")

    print(f"\nLoading dry-season image: {dry_tif.name}", flush=True)
    image, profile, transform, crs = load_and_prepare_image(dry_tif, metadata)
    H, W = image.shape[1], image.shape[2]

    gt_raw, _, _, _ = load_raster(gt_path)
    gt = gt_raw[0].astype(np.uint8)[:H, :W]

    evaluations = {}
    prob_maps = []

    for mname, model in loaded.items():
        print(f"\n  Predicting with {mname}...", flush=True)
        t0 = time.time()
        probs = predict_full_image(model, image, mname, device)
        pred_raw = probs.argmax(axis=0).astype(np.uint8)
        elapsed = time.time() - t0
        print(f"    Inference done in {elapsed:.0f}s", flush=True)

        print(f"    Computing metrics vs WorldCover...", flush=True)
        acc, f1, miou, iou_pc = compute_metrics(pred_raw, gt)
        evaluations[mname] = {
            "Accuracy": round(acc, 2),
            "F1": round(f1, 2),
            "mIoU": round(miou, 2),
            "per_class_IoU": {CLASS_NAMES[i]: round(v, 2) for i, v in enumerate(iou_pc)},
        }
        print(f"    {mname}: OA={acc:.2f}% mIoU={miou:.2f}%", flush=True)

        prob_maps.append(probs)
        del probs
        torch.cuda.empty_cache()

    # Ensemble
    print("\n  Building ensemble...", flush=True)
    ens_raw = max_confidence_ensemble(prob_maps)
    acc, f1, miou, iou_pc = compute_metrics(ens_raw, gt)
    evaluations["ensemble"] = {
        "Accuracy": round(acc, 2),
        "F1": round(f1, 2),
        "mIoU": round(miou, 2),
        "per_class_IoU": {CLASS_NAMES[i]: round(v, 2) for i, v in enumerate(iou_pc)},
    }
    print(f"    ensemble: OA={acc:.2f}% mIoU={miou:.2f}%", flush=True)

    # Save ensemble TIF with light postprocess only
    ens_clean = light_postprocess(ens_raw)
    save_geotiff(ens_clean, out_dir / "lulc_gujranwala_dryseason_ensemble.tif",
                 profile, transform, crs)

    del prob_maps, ens_raw, ens_clean
    torch.cuda.empty_cache()

    # Print table
    print(f"\n{'=' * 65}")
    print(f"  DRY-SEASON GUJRANWALA vs ESA WorldCover 2021")
    print(f"{'=' * 65}")
    header = f"| {'Model':<15} | {'Accuracy':>10} | {'F1':>8} | {'mIoU':>8} |"
    sep = f"|{'-' * 17}|{'-' * 12}|{'-' * 10}|{'-' * 10}|"
    print(header)
    print(sep)
    for m in ["unet", "segformer", "deeplabv3", "prithvi", "ensemble"]:
        if m in evaluations:
            e = evaluations[m]
            label = {"unet": "U-Net (B4)", "segformer": "SegFormer",
                     "deeplabv3": "DeepLabV3+", "prithvi": "Prithvi v2",
                     "ensemble": "**Ensemble**"}.get(m, m)
            print(f"| {label:<15} | {e['Accuracy']:>9.2f}% | {e['F1']:>7.2f}% | {e['mIoU']:>7.2f}% |")

    with open(out_dir / "evaluation_gujranwala_dryseason.json", "w") as f:
        json.dump(evaluations, f, indent=2)

    print(f"\n  Results saved to: {out_dir}", flush=True)
    return evaluations


def run_fix2(device, metadata):
    """Save per-model TIFs for 2018/2026, compute Prithvi-only change detection."""
    print("\n" + "=" * 65)
    print("  FIX #2: PER-MODEL TIFs + PRITHVI CHANGE DETECTION")
    print("=" * 65, flush=True)

    inf_dir = PROJECT_ROOT / "data" / "inference"
    ckpt_dir = PROJECT_ROOT / "checkpoints"

    prithvi_path = ckpt_dir / "prithvi_v2_seed42" / "best_model.pt"
    if not prithvi_path.exists():
        print(f"  [ERROR] Prithvi checkpoint not found: {prithvi_path}")
        return None

    print("\nLoading Prithvi model...", flush=True)
    prithvi = load_model("prithvi", str(prithvi_path), device,
                         num_classes=metadata["num_classes"])

    cities = {
        "lahore": {
            "2018": inf_dir / "lahore" / "sentinel2_2018_10bands.tif",
            "2026": inf_dir / "lahore" / "sentinel2_2026_10bands.tif",
        },
        "gujranwala": {
            "2018": inf_dir / "gujranwala" / "gujranwala_sentinel2_2018_10bands.tif",
            "2026": inf_dir / "gujranwala" / "gujranwala_sentinel2_2026_10bands.tif",
        },
    }

    all_cd = {}

    for city, year_paths in cities.items():
        out_dir = PROJECT_ROOT / "results" / "pipeline" / city
        out_dir.mkdir(parents=True, exist_ok=True)
        preds = {}

        for year, tif_path in year_paths.items():
            out_tif = out_dir / f"lulc_{city}_prithvi_{year}.tif"

            if out_tif.exists():
                print(f"\n  {city}/{year}: Loading existing Prithvi TIF", flush=True)
                data, _, _, _ = load_raster(out_tif)
                preds[year] = data[0].astype(np.uint8)
                continue

            print(f"\n  {city}/{year}: Running Prithvi inference...", flush=True)
            t0 = time.time()
            image, profile, transform, crs = load_and_prepare_image(tif_path, metadata)
            probs = predict_full_image(prithvi, image, "prithvi", device)
            pred_raw = probs.argmax(axis=0).astype(np.uint8)
            elapsed = time.time() - t0
            print(f"    Inference done in {elapsed:.0f}s", flush=True)

            print(f"    Applying median filter...", flush=True)
            pred = light_postprocess(pred_raw)

            save_geotiff(pred, out_tif, profile, transform, crs)
            print(f"    Saved: {out_tif.name}", flush=True)

            preds[year] = pred
            del probs, image, pred_raw
            torch.cuda.empty_cache()

        print(f"\n  {city}: Computing Prithvi-only change detection (2018 -> 2026)...",
              flush=True)
        cd = compute_change_detection(preds["2018"], preds["2026"])

        print(f"\n  {'Class':<15} | {'2018 km2':>10} | {'2026 km2':>10} | {'Change':>10}")
        print(f"  {'-' * 15}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
        for name, vals in cd.items():
            chg = vals["net_change_km2"]
            sign = "+" if chg >= 0 else ""
            print(f"  {name:<15} | {vals['area_2018_km2']:>9.2f} | "
                  f"{vals['area_2026_km2']:>9.2f} | {sign}{chg:>8.2f}")

        cd_out = out_dir / f"change_detection_{city}_prithvi_2018_2026.json"
        with open(cd_out, "w") as f:
            json.dump(cd, f, indent=2)
        print(f"    Saved: {cd_out.name}", flush=True)

        all_cd[city] = cd

    return all_cd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", default="all", choices=["5", "2", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    with open(PROJECT_ROOT / "data" / "patches" / "metadata.json") as f:
        metadata = json.load(f)

    if args.fix in ("5", "all"):
        run_fix5(device, metadata)
        torch.cuda.empty_cache()

    if args.fix in ("2", "all"):
        run_fix2(device, metadata)

    print("\n" + "=" * 65)
    print("  ALL DONE")
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
