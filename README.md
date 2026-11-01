# Benchmarking a LoRA-Adapted EO Foundation Model for Land Cover Mapping and Zero-Shot Cross-City Generalization in Punjab, Pakistan

This repository contains the code for benchmarking NASA/IBM Prithvi-EO 2.0 (a 300M-parameter earth observation foundation model) against U-Net, DeepLabV3+, and MiT-B2+FPN for land use/land cover (LULC) segmentation in Punjab, Pakistan.

## Highlights

- **Prithvi-EO 2.0 + LoRA** outperforms all baselines by +2 pp mIoU (61.19% vs 59.16%) on Lahore test patches, while fine-tuning only 37.6% of parameters
- **Zero-shot cross-city generalization**: models trained on Lahore are applied to Gujranwala without retraining; Prithvi achieves the highest accuracy (mIoU 46.60%)
- **LoRA rank ablation** (r = 8, 16, 32, 64) shows diminishing returns above r = 32
- **Multi-temporal change detection** (2018-2026) reveals +108 km2 built-up expansion in Lahore
- **Consumer-grade hardware**: all training done on a single NVIDIA RTX 4050 (6 GB VRAM) with FP16 mixed precision

## Results

### Multi-Seed Patch-Level Evaluation (Lahore, mean +/- std over seeds 42, 123, 7)

| Model | OA (%) | mIoU (%) | mF1 (%) |
|-------|--------|----------|---------|
| U-Net (EfficientNet-B4) | 81.18 +/- 0.09 | 59.16 +/- 0.07 | 72.85 +/- 0.06 |
| MiT-B2 + FPN | 80.61 +/- 0.12 | 59.07 +/- 0.07 | 72.73 +/- 0.03 |
| DeepLabV3+ (ResNet-101) | 80.81 +/- 0.06 | 59.05 +/- 0.17 | 72.73 +/- 0.14 |
| **Prithvi-EO 2.0 + LoRA** | **82.75 +/- 0.24** | **61.19 +/- 0.25** | **74.53 +/- 0.21** |

### Zero-Shot Cross-City Generalization (seed 42)

| Model | Lahore OA | Lahore mIoU | Gujranwala OA | Gujranwala mIoU |
|-------|-----------|-------------|---------------|-----------------|
| U-Net | 68.92 | 49.68 | 74.78 | 44.75 |
| DeepLabV3+ | 69.05 | 50.14 | 74.82 | 44.34 |
| MiT-B2 + FPN | 68.83 | 49.72 | 74.84 | 45.14 |
| **Prithvi-EO 2.0** | **70.66** | **51.66** | **75.45** | **46.60** |

## Project Structure

```
prithvi-lulc-punjab/
├── data_download/           # Google Earth Engine export scripts
│   ├── gee_sentinel2_export.js
│   ├── gee_sentinel2_2021_export.js
│   ├── gee_lahore_2026_export.js
│   ├── gee_gujranwala_export.js
│   ├── gee_worldcover_export.js
│   └── gee_hansen_export.js
├── preprocessing/
│   └── prepare_patches.py   # Patch extraction (256x256, 70/15/15 split)
├── training/
│   ├── train.py             # Main training script (all models)
│   ├── train_prithvi.py     # Prithvi-EO 2.0 + LoRA training
│   ├── dataset.py           # Multi-band Sentinel-2 dataset loader
│   ├── losses.py            # Dice + Focal loss (0.5 each)
│   ├── run_multiseed.py     # Multi-seed experiment runner
│   ├── run_lora_ablation.py # LoRA rank ablation (r=8,16,32,64)
│   └── watch_progress.py    # Training progress monitor
├── evaluation/
│   ├── inference.py         # Single-model inference
│   ├── inference_multiseed.py
│   ├── evaluate_true_metrics.py
│   ├── significance_tests.py
│   ├── generate_figures.py
│   └── generate_panel_4models.py
├── webapp/                  # Interactive Leaflet.js dashboard
│   ├── index.html
│   ├── comparison.html
│   ├── app.js
│   └── styles.css
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with 6+ GB VRAM (tested on RTX 4050)
- CUDA 11.8+

### Installation

```bash
pip install -r requirements.txt
```

### Data Download

All satellite imagery is exported via Google Earth Engine:

1. Open [code.earthengine.google.com](https://code.earthengine.google.com)
2. Run scripts from `data_download/` to export Sentinel-2 composites and ESA WorldCover labels
3. Download the exported GeoTIFFs from Google Drive into `data/`

**Input data:**
- Sentinel-2 Level-2A, 10 spectral bands + NDVI (11 bands total)
- 10 m resolution
- Lahore composites: Oct-Mar dry/Rabi season
- ESA WorldCover 2021 ground truth, remapped to 5 classes

**LULC Classes:** Vegetation, Built-up, Cropland, Water, Barren

### Training

```bash
# Multi-seed baseline training (U-Net, DeepLabV3+, MiT-B2+FPN)
python training/run_multiseed.py

# Prithvi-EO 2.0 + LoRA training
python training/train_prithvi.py

# LoRA rank ablation
python training/run_lora_ablation.py
```

### Evaluation

```bash
# Full-city inference + zero-shot Gujranwala
python evaluation/inference.py

# Compute metrics
python evaluation/evaluate_true_metrics.py

# Statistical significance tests
python evaluation/significance_tests.py
```

## Models

| Model | Encoder | Total Params | Trainable | Library |
|-------|---------|-------------|-----------|---------|
| U-Net | EfficientNet-B4 | 20M | 20M (100%) | SMP v0.6.0 |
| DeepLabV3+ | ResNet-101 | 46M | 46M (100%) | SMP v0.6.0 |
| MiT-B2 + FPN | MiT-B2 | 26M | 26M (100%) | SMP v0.6.0 |
| Prithvi-EO 2.0 | ViT (24 blocks) | 323M | 121M (37.6%) | Terratorch + PEFT |

**Prithvi adaptation strategy:** 16 frozen transformer blocks with LoRA adapters (r=64, alpha=64) + 8 fully unfrozen blocks + trainable decoder.

## Citation

If you use this code, please cite:

```bibtex
@article{azeem2026benchmarking,
  title={Benchmarking a {LoRA}-Adapted {EO} Foundation Model for Land Cover Mapping and Zero-Shot Cross-City Generalization in {Punjab}, {Pakistan}},
  author={Azeem, Muhammad Hamza},
  journal={Remote Sensing},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
