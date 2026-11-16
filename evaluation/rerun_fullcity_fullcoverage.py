"""
Full-city inference for Tables 3, 4, 5 and 6 with complete window coverage.

Earlier scripts (rerun_all_fullcity.py for Tables 3/4/6, and
rerun_prithvi_fullcity_nanfix.py for the Table 5 maps) placed 256 px windows
at stride 32 starting from 0 and stopped before the image end, so the last
1-31 rows/columns were never predicted and defaulted to class 0 (Vegetation).
Here a final window is added flush with the bottom and right edges, so every
pixel is predicted. Preprocessing is unchanged: NaN -> 0, NDVI, z-score,
softmax averaging over overlapping windows.

Outputs (results/pipeline_fix6_fullcov/):
  fullcov_results.json               metrics for all models (Tables 3, 4, 6)
  {city}/lulc_{city}_prithvi_{year}.tif   Prithvi maps, NaN input -> 255

Resumable: finished jobs in the JSON are skipped.
Usage: python -I evaluation/rerun_fullcity_fullcoverage.py
"""
import sys
import json
import time
import numpy as np
import torch
import rasterio
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
import segmentation_models_pytorch as smp
from training.train_prithvi import (load_prithvi_backbone, PrithviSegmentation,
                                    adapt_input_to_11bands, setup_training_params)

NUM_CLASSES = 5
CLASS_NAMES = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]
PATCH, STRIDE = 256, 32
BATCH = {"prithvi": 2, "unet": 4, "deeplabv3": 4, "segformer": 4}
OUT = PROJECT / "results" / "pipeline_fix6_fullcov"
INF = PROJECT / "data" / "inference"
CK = PROJECT / "checkpoints"
GT_L = PROJECT / "data" / "labels" / "worldcover_2021_5class.tif"
GT_G = INF / "gujranwala" / "worldcover_2021_5class_gujranwala.tif"
ALL = ["unet", "deeplabv3", "segformer", "prithvi"]
# (job, image, ground truth, models, trim-to job for dry season)
JOBS = [
    ("lahore_2021", INF / "lahore" / "sentinel2_2021_10bands.tif", GT_L, ALL),
    ("gujranwala_2021", INF / "gujranwala" / "gujranwala_sentinel2_2021_10bands.tif", GT_G, ALL),
    ("dryseason_2021", INF / "gujranwala" / "gujranwala_sentinel2_2021_dryseason_10bands.tif", GT_G, ALL),
    ("lahore_2018", INF / "lahore" / "sentinel2_2018_10bands.tif", None, ["prithvi"]),
    ("lahore_2026", INF / "lahore" / "sentinel2_2026_10bands.tif", None, ["prithvi"]),
    ("gujranwala_2018", INF / "gujranwala" / "gujranwala_sentinel2_2018_10bands.tif", None, ["prithvi"]),
    ("gujranwala_2026", INF / "gujranwala" / "gujranwala_sentinel2_2026_10bands.tif", None, ["prithvi"]),
]


def load_image(path, metadata):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
        profile = dict(src.profile)
    nan_any = np.isnan(img).any(axis=0)
    nan_b0 = np.isnan(img[0])  # mask used by the published Table 3 script
    img = np.nan_to_num(img, nan=0.0)
    nir, red = img[6], img[2]
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    ndvi = np.clip(ndvi, -1.0, 1.0) * 10000.0
    img = np.concatenate([img, ndvi[np.newaxis]], axis=0)
    means = np.array(metadata["band_means"]).reshape(-1, 1, 1)
    stds = np.array(metadata["band_stds"]).reshape(-1, 1, 1)
    img = (img - means) / (stds + 1e-8)
    return torch.from_numpy(img).float(), nan_any, nan_b0, profile


def starts(n):
    s = list(range(0, n - PATCH + 1, STRIDE))
    if s[-1] != n - PATCH:
        s.append(n - PATCH)
    return s


@torch.no_grad()
def sliding_window(model, img, device, batch, tag):
    C, H, W = img.shape
    pred_sum = torch.zeros(NUM_CLASSES, H, W, dtype=torch.float32)
    count = torch.zeros(H, W, dtype=torch.float32)
    pos = [(y, x) for y in starts(H) for x in starts(W)]
    t0 = time.time()
    for i in range(0, len(pos), batch):
        chunk = pos[i:i + batch]
        b = torch.stack([img[:, y:y + PATCH, x:x + PATCH] for y, x in chunk]).to(device)
        probs = torch.softmax(model(b), dim=1).cpu()
        for k, (y, x) in enumerate(chunk):
            pred_sum[:, y:y + PATCH, x:x + PATCH] += probs[k]
            count[y:y + PATCH, x:x + PATCH] += 1
        done = i + len(chunk)
        if done % 1000 < batch or done == len(pos):
            el = time.time() - t0
            print(f"  {tag}: {done}/{len(pos)} windows, {el/60:.1f} min, eta {el/done*(len(pos)-done)/60:.1f} min", flush=True)
    if (count == 0).any():
        raise RuntimeError(f"{tag}: {(count == 0).sum()} pixels not covered")
    if torch.isnan(pred_sum).any():
        raise RuntimeError(f"{tag}: NaN in probabilities")
    return (pred_sum / count.unsqueeze(0)).argmax(dim=0).numpy().astype(np.uint8)


def metrics(pred, gt, nan_mask):
    valid = (~nan_mask) & (gt != 255)
    yt, yp = gt[valid].astype(int), pred[valid].astype(int)
    cm = confusion_matrix(yt, yp, labels=np.arange(NUM_CLASSES))
    inter = np.diag(cm); union = cm.sum(1) + cm.sum(0) - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
    return {"OA": round(accuracy_score(yt, yp) * 100, 2), "mIoU": round(float(np.nanmean(iou)) * 100, 2),
            "per_class_IoU": {CLASS_NAMES[i]: round(float(iou[i] * 100), 2) for i in range(NUM_CLASSES)},
            "n_valid_px": int(valid.sum())}


def build_models(device, needed):
    models = {}
    if "unet" in needed:
        m = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=11, classes=5)
        m.load_state_dict(torch.load(CK / "unet_seed42" / "best_model.pt", map_location=device)); models["unet"] = m
    if "deeplabv3" in needed:
        m = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=11, classes=5)
        m.load_state_dict(torch.load(CK / "deeplabv3_seed42" / "best_model.pt", map_location=device)); models["deeplabv3"] = m
    if "segformer" in needed:
        m = smp.FPN(encoder_name="mit_b2", encoder_weights=None, in_channels=11, classes=5)
        m.load_state_dict(torch.load(CK / "segformer_seed42" / "best_model.pt", map_location=device)); models["segformer"] = m
    if "prithvi" in needed:
        backbone, feat_dim, b_type = load_prithvi_backbone()
        adapt_input_to_11bands(backbone, b_type)
        m = PrithviSegmentation(backbone, feat_dim, b_type, NUM_CLASSES)
        setup_training_params(m, lora_rank=64, unfreeze_last_n=8)
        m.load_state_dict(torch.load(CK / "prithvi_v2_fix6_seed42" / "best_model.pt", map_location=device), strict=True)
        models["prithvi"] = m
    for m in models.values():
        m.to(device).eval()
    return models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "fullcov_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    todo = [(j, p, g, m) for j, p, g, ms in JOBS for m in ms if f"{j}/{m}" not in results]
    print("PLAN: " + ",".join(f"{j}/{m}" for j, _, _, ms in JOBS for m in ms), flush=True)
    print("DONE: " + ",".join(results), flush=True)
    if not todo:
        print("Nothing to do."); return
    metadata = json.loads((PROJECT / "data" / "patches" / "metadata.json").read_text())
    models = build_models(device, {m for *_, m in todo})
    print(f"Device {device}; models {list(models)}", flush=True)
    shape_g = rasterio.open(INF / "gujranwala" / "gujranwala_sentinel2_2021_10bands.tif").shape
    for job, path, gt_path, ms in JOBS:
        pending = [m for m in ms if f"{job}/{m}" not in results]
        if not pending:
            continue
        print(f"\n== {job} ==", flush=True)
        img, nan_any, nan_b0, profile = load_image(path, metadata)
        if job == "dryseason_2021":  # published script trims the dry composite to the full-year grid
            H, W = shape_g
            img, nan_any, nan_b0 = img[:, :H, :W], nan_any[:H, :W], nan_b0[:H, :W]
        gt = rasterio.open(gt_path).read(1)[:img.shape[1], :img.shape[2]] if gt_path else None
        for m in pending:
            tag = f"{job}/{m}"
            pred = sliding_window(models[m], img, device, BATCH[m], tag)
            rec = {"shape": list(pred.shape)}
            if gt is not None:
                rec["metrics_table3_mask"] = metrics(pred, gt, nan_b0)
                rec["metrics_anyband_mask"] = metrics(pred, gt, nan_any)
                r = rec["metrics_table3_mask"]
                print(f"  RESULT {tag}: OA {r['OA']} mIoU {r['mIoU']} {r['per_class_IoU']}", flush=True)
            if m == "prithvi" and job != "dryseason_2021":
                city, year = job.split("_")
                out_map = pred.copy(); out_map[nan_any] = 255
                (OUT / city).mkdir(exist_ok=True)
                prof = dict(profile); prof.update(count=1, dtype="uint8", nodata=255, compress="lzw")
                with rasterio.open(OUT / city / f"lulc_{city}_prithvi_{year}.tif", "w", **prof) as d:
                    d.write(out_map[np.newaxis])
                rec["class_px"] = {CLASS_NAMES[c]: int((out_map == c).sum()) for c in range(NUM_CLASSES)}
            results[tag] = rec
            res_path.write_text(json.dumps(results, indent=2))
            print(f"  SAVED {tag}", flush=True)
            torch.cuda.empty_cache()
        del img
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
