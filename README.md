<div align="center">

# Benchmarking a LoRA-Adapted EO Foundation Model for Land Cover Mapping and Zero-Shot Cross-City Generalization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-MDPI_Remote_Sensing-blue)](https://www.mdpi.com/journal/remotesensing)
[![Sentinel-2](https://img.shields.io/badge/Data-Sentinel--2_L2A-orange?logo=satellite&logoColor=white)](#data)

**Can a foundation model pre-trained on global satellite imagery outperform task-specific architectures in South Asian urban landscapes -- and generalize to unseen cities without retraining?**

[Paper (under review)](#citation) | [Interactive Dashboard](#web-dashboard) | [Getting Started](#setup)

</div>

---

## Overview

We benchmark **NASA/IBM Prithvi-EO 2.0** (300M params), fine-tuned via **LoRA + selective block unfreezing**, against U-Net, DeepLabV3+, and MiT-B2+FPN for 5-class land cover segmentation over **Lahore, Pakistan** using 11-band Sentinel-2 imagery. We then test zero-shot cross-city transfer to **Gujranwala** -- no retraining, no fine-tuning, just inference on an unseen city.

<div align="center">
<img src="docs/methodology.png" alt="End-to-end methodology" width="95%"/>
<br/>
<em>Figure 1: End-to-end methodology -- from data acquisition through model benchmarking to change detection and deployment.</em>
</div>

---

## Key Findings

| | Finding | Detail |
|---|---------|--------|
| 1 | **Prithvi wins on all metrics** | +2.03 pp mIoU over the best baseline (U-Net) across 3 seeds |
| 2 | **Zero-shot transfer works** | Prithvi scores 46.60% mIoU on Gujranwala without seeing a single training sample from it |
| 3 | **LoRA is efficient** | Only 37.6% of parameters are trainable; diminishing returns above rank 32 |
| 4 | **Runs on consumer hardware** | All training on a single RTX 4050 (6 GB VRAM) with FP16 mixed precision |
| 5 | **Urbanization quantified** | +108 km2 built-up expansion in Lahore between 2018 and 2026 |

---

## Architecture

<div align="center">
<img src="docs/architecture.png" alt="Prithvi-EO 2.0 + LoRA architecture" width="95%"/>
<br/>
<em>Figure 2: Prithvi-EO 2.0 adaptation -- 16 frozen blocks with LoRA (r=64) + 8 fully unfrozen blocks + trainable decoder.</em>
</div>

---

## Results

### Multi-Seed Patch-Level Evaluation (Lahore test set, seeds 42/123/7)

| Model | Encoder | Params | OA (%) | mIoU (%) | mF1 (%) |
|:------|:--------|-------:|-------:|---------:|--------:|
| U-Net | EfficientNet-B4 | 20M | 81.18 +/- 0.09 | 59.16 +/- 0.07 | 72.85 +/- 0.06 |
| MiT-B2 + FPN | MiT-B2 | 26M | 80.61 +/- 0.12 | 59.07 +/- 0.07 | 72.73 +/- 0.03 |
| DeepLabV3+ | ResNet-101 | 46M | 80.81 +/- 0.06 | 59.05 +/- 0.17 | 72.73 +/- 0.14 |
| **Prithvi-EO 2.0** | **ViT + LoRA** | **323M** | **82.75 +/- 0.24** | **61.19 +/- 0.25** | **74.53 +/- 0.21** |

### Zero-Shot Cross-City Generalization (seed 42)

| Model | Lahore OA | Lahore mIoU | Gujranwala OA | Gujranwala mIoU |
|:------|----------:|------------:|--------------:|----------------:|
| U-Net | 68.92 | 49.68 | 74.78 | 44.75 |
| DeepLabV3+ | 69.05 | 50.14 | 74.82 | 44.34 |
| MiT-B2 + FPN | 68.83 | 49.72 | 74.84 | 45.14 |
| **Prithvi-EO 2.0** | **70.66** | **51.66** | **75.45** | **46.60** |

### LoRA Rank Ablation

| Rank | Trainable Params | Train % | mIoU (%) | OA (%) |
|-----:|-----------------:|--------:|---------:|-------:|
| 8 | 106.7M | 34.6% | 60.40 | 82.07 |
| 16 | 109.6M | 35.2% | 60.67 | 82.63 |
| 32 | 115.4M | 36.4% | 61.12 | 82.62 |
| **64** | **121.4M** | **37.6%** | **61.19** | **82.70** |

---

## Visual Results

### Prediction Comparison (Lahore test patches)

<div align="center">
<img src="docs/predictions_panel.png" alt="Prediction comparison across models" width="95%"/>
<br/>
<em>Qualitative comparison: RGB input, ground truth, and predictions from all four models.</em>
</div>

### Change Detection (2018 - 2026)

<div align="center">
<img src="docs/change_detection.png" alt="Change detection maps" width="95%"/>
<br/>
<em>Multi-temporal land cover change in Lahore and Gujranwala, detected via post-classification comparison using the best-performing Prithvi model.</em>
</div>

---

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
│   └── prepare_patches.py   # Patch extraction (256x256, 70/15/15 stratified split)
├── training/
│   ├── train.py             # Baseline training (U-Net, DeepLabV3+, MiT-B2+FPN)
│   ├── train_prithvi.py     # Prithvi-EO 2.0 + LoRA fine-tuning
│   ├── dataset.py           # 11-band Sentinel-2 dataset loader
│   ├── losses.py            # Dice + Focal loss (0.5 each)
│   ├── run_multiseed.py     # Multi-seed experiment runner (seeds 42, 123, 7)
│   └── run_lora_ablation.py # LoRA rank sweep (r = 8, 16, 32, 64)
├── evaluation/
│   ├── inference.py         # Full-city inference + zero-shot transfer
│   ├── evaluate_true_metrics.py  # OA, mIoU, mF1, per-class metrics
│   ├── significance_tests.py     # Statistical significance analysis
│   └── generate_panel_4models.py # Figure generation
├── webapp/                  # Interactive Leaflet.js dashboard
├── docs/                    # README figures
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with 6+ GB VRAM (tested on RTX 4050 Laptop)
- CUDA 11.8+

### Installation

```bash
git clone https://github.com/ZentaCros/prithvi-lulc-punjab.git
cd prithvi-lulc-punjab
pip install -r requirements.txt
```

### Data

All satellite imagery is sourced from [Google Earth Engine](https://earthengine.google.com/):

| Dataset | Source | Resolution | Use |
|---------|--------|-----------|-----|
| Sentinel-2 L2A | Copernicus/ESA | 10 m | 10 spectral bands + NDVI |
| ESA WorldCover 2021 | ESA | 10 m | Ground truth (remapped to 5 classes) |
| Hansen Forest Change | UMD/Google | 30 m | Validation |

**LULC Classes:** Vegetation - Built-up - Cropland - Water - Barren

Run the scripts in `data_download/` to export composites, then download from Google Drive.

### Training

```bash
# Multi-seed baselines (U-Net, DeepLabV3+, MiT-B2+FPN)
python training/run_multiseed.py

# Prithvi-EO 2.0 + LoRA
python training/train_prithvi.py

# LoRA rank ablation
python training/run_lora_ablation.py
```

### Evaluation

```bash
# Full-city inference + zero-shot Gujranwala transfer
python evaluation/inference.py

# Compute metrics
python evaluation/evaluate_true_metrics.py

# Statistical significance
python evaluation/significance_tests.py
```

---

## Models

| Model | Encoder | Total Params | Trainable | Strategy | Library |
|-------|---------|:-----------:|:---------:|----------|---------|
| U-Net | EfficientNet-B4 | 20M | 20M (100%) | Full fine-tune | SMP v0.6.0 |
| DeepLabV3+ | ResNet-101 | 46M | 46M (100%) | Full fine-tune | SMP v0.6.0 |
| MiT-B2 + FPN | MiT-B2 | 26M | 26M (100%) | Full fine-tune | SMP v0.6.0 |
| Prithvi-EO 2.0 | ViT (24 blocks) | 323M | 121M (37.6%) | LoRA (r=64) + 8 unfrozen blocks | Terratorch + PEFT |

**Training config:** 50 epochs, AdamW optimizer, OneCycleLR scheduler, Dice + Focal loss (0.5 each), batch size 2, FP16 mixed precision.

---

## Web Dashboard

An interactive Leaflet.js dashboard for exploring LULC maps and change detection results across both cities and all time periods.

```bash
# Serve locally
cd webapp
python -m http.server 8000
# Open http://localhost:8000
```

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{azeem2026benchmarking,
  title={Benchmarking a {LoRA}-Adapted {EO} Foundation Model for Land Cover
         Mapping and Zero-Shot Cross-City Generalization in {Punjab}, {Pakistan}},
  author={Azeem, Muhammad Hamza},
  journal={Remote Sensing},
  year={2026},
  note={Under review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with limited resources, unlimited curiosity.</sub>
</div>
