"""
Generate normalized confusion matrices for all 4 models (seed 42).
Uses the current 11-band pipeline.

GPU Required: yes (model inference on test patches)

Usage:
    python evaluation/generate_confusion_matrices.py
"""

import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import LULCDataset
from training.train import create_model
from training.train_prithvi import (
    load_prithvi_backbone, PrithviSegmentation,
    adapt_input_to_11bands, setup_training_params
)


def plot_confusion_matrix(y_true, y_pred, classes, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        vmin=0, vmax=1, square=True, linewidths=0.5,
        annot_kws={"size": 12}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def load_baseline_model(model_name, ckpt_path, device):
    model = create_model(model_name, in_channels=11, num_classes=5)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_prithvi_model(ckpt_path, device, lora_rank=64, unfreeze_blocks=8):
    backbone, feat_dim, b_type = load_prithvi_backbone()
    adapt_input_to_11bands(backbone, b_type)
    num_classes = 5
    model = PrithviSegmentation(backbone, feat_dim, b_type, num_classes)
    setup_training_params(model, lora_rank=lora_rank, unfreeze_last_n=unfreeze_blocks)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type != 'cuda':
        print("WARNING: Running on CPU. This will be slow.")

    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    patches_dir = PROJECT_ROOT / "data" / "patches"
    with open(patches_dir / "metadata.json") as f:
        metadata = json.load(f)

    classes = metadata["class_names"]
    test_ds = LULCDataset(patches_dir, "test", metadata, augment=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    models_config = {
        "U-Net": ("unet", ckpt_dir / "unet_seed42" / "best_model.pt"),
        "MiT-B2+FPN": ("segformer", ckpt_dir / "segformer_seed42" / "best_model.pt"),
        "DeepLabV3+": ("deeplabv3", ckpt_dir / "deeplabv3_seed42" / "best_model.pt"),
        "Prithvi-EO 2.0": ("prithvi", ckpt_dir / "prithvi_v2_seed42" / "best_model.pt"),
    }

    all_targets = []
    all_preds = {name: [] for name in models_config}

    for name, (model_type, ckpt_path) in models_config.items():
        print(f"\nLoading {name} from {ckpt_path.name}...")
        if not ckpt_path.exists():
            print(f"  SKIP: checkpoint not found")
            continue

        if model_type == "prithvi":
            model = load_prithvi_model(ckpt_path, device)
        else:
            model = load_baseline_model(model_type, ckpt_path, device)

        print(f"  Running inference on {len(test_ds)} test patches...")
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy().flatten()
                all_preds[name].append(preds)

                if name == list(models_config.keys())[0]:
                    all_targets.append(targets.numpy().flatten())

        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    y_true = np.concatenate(all_targets)

    for name in models_config:
        if not all_preds[name]:
            continue
        y_pred = np.concatenate(all_preds[name])
        safe_name = name.replace(" ", "_").replace("+", "plus").replace(".", "")
        out_path = out_dir / f"cm_{safe_name}.png"
        plot_confusion_matrix(
            y_true, y_pred, classes,
            f"{name} - Normalized Confusion Matrix (Seed 42)",
            out_path
        )

    print(f"\nAll confusion matrices saved to {out_dir}")


if __name__ == "__main__":
    main()
