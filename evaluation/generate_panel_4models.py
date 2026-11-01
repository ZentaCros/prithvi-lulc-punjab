"""
Generate predictions panel with ALL 4 models for the paper figure.
Uses seed 42 checkpoints for all models.
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import LULCDataset
import segmentation_models_pytorch as smp
from training.train_prithvi import load_prithvi_backbone, PrithviSegmentation, adapt_input_to_11bands, setup_training_params


def get_rgb(image_tensor, metadata):
    rgb = image_tensor[[2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0)
    means = np.array([metadata["band_means"][i] for i in [2, 1, 0]])
    stds = np.array([metadata["band_stds"][i] for i in [2, 1, 0]])
    rgb = (rgb * stds) + means
    p2, p98 = np.percentile(rgb, (2, 98), axis=(0, 1))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    return rgb


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    patches_dir = PROJECT_ROOT / "data" / "patches"
    with open(patches_dir / "metadata.json") as f:
        metadata = json.load(f)

    classes = metadata["class_names"]
    cmap = plt.cm.colors.ListedColormap(['#2ca02c', '#d62728', '#bcbd22', '#1f77b4', '#7f7f7f'])
    norm = plt.cm.colors.BoundaryNorm(np.arange(-0.5, len(classes) + 0.5, 1), cmap.N)

    test_ds = LULCDataset(patches_dir, "test", metadata, augment=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)

    models_to_load = {
        "U-Net": {
            "ckpt": PROJECT_ROOT / "checkpoints" / "unet_seed42" / "best_model.pt",
            "build": lambda: smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None, in_channels=11, classes=5),
        },
        "MiT-B2+FPN": {
            "ckpt": PROJECT_ROOT / "checkpoints" / "segformer_seed42" / "best_model.pt",
            "build": lambda: smp.FPN(encoder_name="mit_b2", encoder_weights=None, in_channels=11, classes=5),
        },
        "DeepLabV3+": {
            "ckpt": PROJECT_ROOT / "checkpoints" / "deeplabv3_seed42" / "best_model.pt",
            "build": lambda: smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=11, classes=5),
        },
        "Prithvi-EO 2.0": {
            "ckpt": PROJECT_ROOT / "checkpoints" / "prithvi_v2_seed42" / "best_model.pt",
            "build": "prithvi",
        },
    }

    models = {}
    for name, cfg in models_to_load.items():
        print(f"Loading {name}...")
        if cfg["build"] == "prithvi":
            backbone, feat_dim, b_type = load_prithvi_backbone()
            adapt_input_to_11bands(backbone, b_type)
            model = PrithviSegmentation(backbone, feat_dim, b_type, metadata["num_classes"])
            setup_training_params(model, lora_rank=64, unfreeze_last_n=8)
        else:
            model = cfg["build"]()

        state = torch.load(cfg["ckpt"], map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models[name] = model
        print(f"  Loaded from {cfg['ckpt'].name}")

    print("\nRunning inference on test patches...")
    model_names = list(models.keys())
    num_cols = 2 + len(model_names)  # RGB + GT + 4 models = 6

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if batch_idx > 0:
                break
            images, targets = images.to(device), targets.to(device)
            num_samples = min(4, images.size(0))

            preds = {}
            for name, model in models.items():
                if "Prithvi" in name:
                    # batch_size=1 for Prithvi to fit in VRAM
                    batch_preds = []
                    for i in range(num_samples):
                        out = model(images[i:i+1])
                        batch_preds.append(out.argmax(dim=1).cpu().numpy()[0])
                    preds[name] = batch_preds
                else:
                    out = model(images[:num_samples])
                    preds[name] = [out.argmax(dim=1).cpu().numpy()[i] for i in range(num_samples)]

            print("Generating panel figure...")
            fig, axes = plt.subplots(num_samples, num_cols, figsize=(4 * num_cols, 4 * num_samples))

            for i in range(num_samples):
                rgb = get_rgb(images[i], metadata)
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title("RGB Sentinel-2", fontsize=14)
                axes[i, 0].axis("off")

                axes[i, 1].imshow(targets[i].cpu(), cmap=cmap, norm=norm, interpolation='nearest')
                axes[i, 1].set_title("Ground Truth (WorldCover)", fontsize=14)
                axes[i, 1].axis("off")

                for j, name in enumerate(model_names):
                    axes[i, j + 2].imshow(preds[name][i], cmap=cmap, norm=norm, interpolation='nearest')
                    axes[i, j + 2].set_title(name, fontsize=14)
                    axes[i, j + 2].axis("off")

            plt.tight_layout()
            out_path = out_dir / "predictions_panel.png"
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {out_path} ({os.path.getsize(out_path) // 1024} KB)")

    print("Done!")


if __name__ == "__main__":
    main()
