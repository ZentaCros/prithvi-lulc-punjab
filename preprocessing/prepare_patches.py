"""
LULC Lahore  Patch Extraction & Dataset Preparation
=====================================================
Reads full-scene Sentinel-2 GeoTIFFs + WorldCover labels,
extracts aligned 256256 patches, computes normalization stats,
splits into train/val/test, and saves as .npy files.

Usage:
    python preprocessing/prepare_patches.py
"""

import os
import sys
import json
import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import Counter

#  CONFIG 
PATCH_SIZE = 256
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
MIN_VALID_PIXELS = 0.80  # Skip patches with >20% nodata

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
S2_2021_PATH = DATA_DIR / "inference" / "lahore" / "sentinel2_2021_10bands.tif"
S2_2018_PATH = DATA_DIR / "inference" / "lahore" / "sentinel2_2018_10bands.tif"
LABEL_PATH = DATA_DIR / "labels" / "worldcover_2021_5class.tif"
PATCHES_DIR = DATA_DIR / "patches"

# Band names for reference (10 Sentinel-2 bands + computed NDVI)
BAND_NAMES = ['B2_Blue', 'B3_Green', 'B4_Red', 'B5_RE1', 'B6_RE2',
              'B7_RE3', 'B8_NIR', 'B8A_NIR2', 'B11_SWIR1', 'B12_SWIR2', 'NDVI']

# Band indices for NDVI: (NIR - Red) / (NIR + Red)
# Band 6 = B8 NIR (index 6), Band 2 = B4 Red (index 2) in our 0-indexed 10-band array
NDVI_NIR_IDX = 6  # B8
NDVI_RED_IDX = 2  # B4

# Class info
CLASS_NAMES = ['Vegetation', 'Built-up', 'Cropland', 'Water', 'Barren/Other']
NUM_CLASSES = 5


def load_geotiff(path, description=""):
    """Load a GeoTIFF and return (data, profile)."""
    print(f"  Loading {description}: {path.name}")
    with rasterio.open(path) as src:
        data = src.read()  # [C, H, W] for multi-band, [1, H, W] for single
        profile = src.profile.copy()
        print(f"    Shape: {data.shape}, Dtype: {data.dtype}")
        print(f"    CRS: {src.crs}, Resolution: {src.res}")
        print(f"    Bounds: {src.bounds}")
    return data, profile


def align_rasters(image, labels, img_profile, lbl_profile):
    """
    Ensure image and labels have the same spatial extent and resolution.
    Since we exported both from GEE with same CRS and region, they should
    already be aligned. This function verifies and trims if needed.
    """
    _, img_h, img_w = image.shape
    _, lbl_h, lbl_w = labels.shape

    # Use the minimum overlap
    h = min(img_h, lbl_h)
    w = min(img_w, lbl_w)

    if img_h != lbl_h or img_w != lbl_w:
        print(f"   Spatial mismatch! Image: {img_h}{img_w}, Labels: {lbl_h}{lbl_w}")
        print(f"    Trimming both to {h}{w}")

    image = image[:, :h, :w]
    labels = labels[:, :h, :w]

    return image, labels


def compute_ndvi(image):
    """
    Compute NDVI from a 10-band Sentinel-2 image and append as band 11.
    NDVI = (NIR - Red) / (NIR + Red)
    Values are clipped to [-1, 1] and scaled to match reflectance range.
    """
    nir = image[NDVI_NIR_IDX].astype(np.float32)
    red = image[NDVI_RED_IDX].astype(np.float32)
    denom = nir + red
    # Avoid division by zero
    ndvi = np.where(denom > 0, (nir - red) / denom, 0.0)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    # Scale to match typical reflectance range of our bands (roughly 0-10000)
    ndvi_scaled = ndvi * 10000.0
    return ndvi_scaled[np.newaxis]  # shape (1, H, W)


def extract_patches(image, labels, patch_size=256, min_valid=0.80):
    """
    Extract non-overlapping patches from image and label arrays.
    Skips patches with too much nodata (NaN/0 in all bands).
    
    Returns: list of (image_patch, label_patch) tuples
    """
    C, H, W = image.shape
    patches = []
    skipped_nodata = 0
    skipped_edge = 0

    n_rows = H // patch_size
    n_cols = W // patch_size

    print(f"  Grid: {n_rows} rows  {n_cols} cols = {n_rows * n_cols} potential patches")

    for i in tqdm(range(n_rows), desc="  Extracting patches"):
        for j in range(n_cols):
            r_start = i * patch_size
            c_start = j * patch_size

            img_patch = image[:, r_start:r_start+patch_size, c_start:c_start+patch_size]
            lbl_patch = labels[0, r_start:r_start+patch_size, c_start:c_start+patch_size]

            # Check for nodata: if ANY band is all-zero or NaN for too many pixels
            valid_mask = np.all(np.isfinite(img_patch), axis=0) & np.any(img_patch != 0, axis=0)
            valid_ratio = valid_mask.sum() / (patch_size * patch_size)

            if valid_ratio < min_valid:
                skipped_nodata += 1
                continue

            # Replace remaining NaN with 0
            img_patch = np.nan_to_num(img_patch, nan=0.0)

            patches.append((img_patch.astype(np.float32), lbl_patch.astype(np.int64)))

    print(f"   Extracted: {len(patches)} valid patches")
    print(f"   Skipped: {skipped_nodata} (nodata), {skipped_edge} (edge)")
    return patches


def compute_band_statistics(patches):
    """Compute per-band mean and std from training patches for normalization."""
    print("  Computing per-band normalization statistics...")
    
    # Stack all image patches: [N, C, H, W]
    all_images = np.stack([p[0] for p in patches], axis=0)
    N, C, H, W = all_images.shape
    
    # Compute mean and std per band (across all pixels in all patches)
    band_means = []
    band_stds = []
    
    for b in range(C):
        band_data = all_images[:, b, :, :].flatten()
        # Exclude zero/nodata pixels from statistics
        valid = band_data[band_data != 0]
        if len(valid) > 0:
            band_means.append(float(np.mean(valid)))
            band_stds.append(float(np.std(valid)))
        else:
            band_means.append(0.0)
            band_stds.append(1.0)
    
    print(f"  Band means: {[f'{m:.1f}' for m in band_means]}")
    print(f"  Band stds:  {[f'{s:.1f}' for s in band_stds]}")
    
    return band_means, band_stds


def compute_class_distribution(patches):
    """Count pixels per class across all patches."""
    all_labels = np.concatenate([p[1].flatten() for p in patches])
    counts = Counter(all_labels.tolist())
    total = len(all_labels)
    
    print(f"\n  Class Distribution ({total:,} total pixels):")
    print(f"  {'Class':<20} {'Pixels':>12} {'Percent':>8}")
    print(f"  {'-'*42}")
    
    class_weights = []
    for cls_id in range(NUM_CLASSES):
        count = counts.get(cls_id, 0)
        pct = count / total * 100
        print(f"  {CLASS_NAMES[cls_id]:<20} {count:>12,} {pct:>7.1f}%")
        # Inverse frequency weighting
        class_weights.append(total / (NUM_CLASSES * max(count, 1)))
    
    # Normalize weights so they sum to NUM_CLASSES
    w_sum = sum(class_weights)
    class_weights = [w * NUM_CLASSES / w_sum for w in class_weights]
    print(f"\n  Class weights (inv. freq): {[f'{w:.3f}' for w in class_weights]}")
    
    return dict(counts), class_weights


def save_patches(patches, output_dir, split_name):
    """Save patches as individual .npy files."""
    split_dir = output_dir / split_name
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img, lbl) in enumerate(patches):
        np.save(img_dir / f"patch_{idx:05d}.npy", img)
        np.save(lbl_dir / f"patch_{idx:05d}.npy", lbl)

    print(f"   Saved {len(patches)} patches to {split_dir}/")


def main():
    print("=" * 65)
    print("  LULC LAHORE  PATCH EXTRACTION PIPELINE")
    print("=" * 65)

    #  1. LOAD DATA 
    print("\n Step 1: Loading GeoTIFFs...")
    
    if not S2_2021_PATH.exists():
        print(f"   File not found: {S2_2021_PATH}")
        sys.exit(1)
    if not LABEL_PATH.exists():
        print(f"   File not found: {LABEL_PATH}")
        sys.exit(1)

    # We train on 2021 imagery (Perfect temporal alignment with 2021 WorldCover labels)
    image_2021, img_profile = load_geotiff(S2_2021_PATH, "Sentinel-2 2021")
    labels, lbl_profile = load_geotiff(LABEL_PATH, "WorldCover 2021 (5-class)")

    #  2. ALIGN 
    print("\n Step 2: Aligning rasters...")
    image_2021, labels = align_rasters(image_2021, labels, img_profile, lbl_profile)
    print(f"  Final shapes - Image: {image_2021.shape}, Labels: {labels.shape}")

    #  2b. COMPUTE & APPEND NDVI 
    print("\n Step 2b: Computing NDVI and appending as band 11...")
    ndvi = compute_ndvi(image_2021)
    image_2021 = np.concatenate([image_2021, ndvi], axis=0)  # (11, H, W)
    print(f"  Image shape after NDVI: {image_2021.shape}")

    #  3. EXTRACT PATCHES 
    print(f"\n  Step 3: Extracting {PATCH_SIZE}{PATCH_SIZE} patches...")
    patches = extract_patches(image_2021, labels, PATCH_SIZE, MIN_VALID_PIXELS)

    if len(patches) == 0:
        print("   No valid patches extracted! Check your data.")
        sys.exit(1)

    #  4. CLASS DISTRIBUTION 
    print("\n Step 4: Analyzing class distribution...")
    class_counts, class_weights = compute_class_distribution(patches)

    #  5. TRAIN/VAL/TEST SPLIT 
    print(f"\n Step 5: Splitting into train/val/test ({TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO})...")

    # Get dominant class per patch for stratified splitting
    patch_classes = [np.argmax(np.bincount(p[1].flatten(), minlength=NUM_CLASSES)) for p in patches]

    # First split: train+val vs test
    train_val_patches, test_patches, train_val_classes, _ = train_test_split(
        patches, patch_classes,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=patch_classes
    )

    # Second split: train vs val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_patches, val_patches = train_test_split(
        train_val_patches,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=train_val_classes
    )

    print(f"  Train: {len(train_patches)} patches")
    print(f"  Val:   {len(val_patches)} patches")
    print(f"  Test:  {len(test_patches)} patches")

    #  6. COMPUTE NORMALIZATION STATS (from train set only) 
    print(f"\n Step 6: Computing normalization statistics (train set only)...")
    band_means, band_stds = compute_band_statistics(train_patches)

    #  7. SAVE PATCHES 
    print(f"\n Step 7: Saving patches to {PATCHES_DIR}/...")
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    save_patches(train_patches, PATCHES_DIR, "train")
    save_patches(val_patches, PATCHES_DIR, "val")
    save_patches(test_patches, PATCHES_DIR, "test")

    #  8. SAVE METADATA 
    print(f"\n Step 8: Saving metadata...")
    metadata = {
        "patch_size": PATCH_SIZE,
        "num_bands": image_2021.shape[0],  # 11 (10 Sentinel-2 + NDVI)
        "band_names": BAND_NAMES,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_weights": class_weights,
        "class_counts": {CLASS_NAMES[int(k)]: v for k, v in class_counts.items() if int(k) < NUM_CLASSES},
        "band_means": band_means,
        "band_stds": band_stds,
        "splits": {
            "train": len(train_patches),
            "val": len(val_patches),
            "test": len(test_patches),
            "total": len(patches),
        },
        "source_image": S2_2021_PATH.name,
        "label_source": LABEL_PATH.name,
        "image_shape": list(image_2021.shape),
        "crs": str(img_profile.get("crs", "unknown")),
        "random_seed": RANDOM_SEED,
    }

    meta_path = PATCHES_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata to {meta_path}")

    #  SUMMARY 
    print(f"\n{'=' * 65}")
    print(f"   PREPROCESSING COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Image size:    {image_2021.shape[1]}{image_2021.shape[2]} pixels")
    print(f"  Bands:         {image_2021.shape[0]}")
    print(f"  Patch size:    {PATCH_SIZE}{PATCH_SIZE}")
    print(f"  Total patches: {len(patches)}")
    print(f"  Train/Val/Test: {len(train_patches)}/{len(val_patches)}/{len(test_patches)}")
    print(f"  Output:        {PATCHES_DIR}/")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
