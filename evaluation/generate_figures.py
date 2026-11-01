"""
Generate publication-quality figures:
1. Confusion Matrices for all 3 models
2. Side-by-side visual predictions panel
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import LULCDataset
import segmentation_models_pytorch as smp
from training.train_prithvi import load_prithvi_backbone, PrithviSegmentation, adapt_input_to_10bands, setup_training_params

def get_rgb(image_tensor, metadata):
    """Convert 10-band normalized tensor to visible RGB image for plotting"""
    # RGB are bands 2, 1, 0 (Red, Green, Blue) in Sentinel-2
    # For HLS, it's RED(2), GREEN(1), BLUE(0)
    # Our band order: BLUE(0), GREEN(1), RED(2)
    rgb = image_tensor[[2, 1, 0], :, :].cpu().numpy().transpose(1, 2, 0)
    # Denormalize
    means = np.array([metadata["band_means"][i] for i in [2, 1, 0]])
    stds = np.array([metadata["band_stds"][i] for i in [2, 1, 0]])
    rgb = (rgb * stds) + means
    
    # Clip for display
    # 98th percentile stretch for better visualization
    p2, p98 = np.percentile(rgb, (2, 98), axis=(0,1))
    rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
    return rgb

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Replace NaNs with 0
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.ylabel('True Class', fontsize=14)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    patches_dir = PROJECT_ROOT / "data" / "patches"
    with open(patches_dir / "metadata.json") as f:
        metadata = json.load(f)

    classes = metadata["class_names"]
    
    # Use a fixed colormap mapping 5 classes
    # 0: Vegetation (Green), 1: Built-up (Red), 2: Cropland (Yellow), 3: Water (Blue), 4: Barren/Other (Gray)
    cmap = plt.cm.colors.ListedColormap(['#2ca02c', '#d62728', '#bcbd22', '#1f77b4', '#7f7f7f'])
    norm = plt.cm.colors.BoundaryNorm(np.arange(-0.5, len(classes)+0.5, 1), cmap.N)

    test_ds = LULCDataset(patches_dir, "test", metadata, augment=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)

    models = {}
    models_to_test = {
        "SegFormer": PROJECT_ROOT / "checkpoints" / "segformer_seed123" / "best_model.pt",
        "DeepLabV3+": PROJECT_ROOT / "checkpoints" / "deeplabv3_seed456" / "best_model.pt",
        "Prithvi_v2": PROJECT_ROOT / "checkpoints" / "prithvi_v2_seed42" / "best_model.pt",
    }

    print("Loading models...")
    for name, ckpt_path in models_to_test.items():
        if "Prithvi" in name:
            backbone, feat_dim, b_type = load_prithvi_backbone()
            adapt_input_to_10bands(backbone, b_type)
            model = PrithviSegmentation(backbone, feat_dim, b_type, metadata["num_classes"])
            setup_training_params(model, lora_rank=32, unfreeze_last_n=0)
        elif "SegFormer" in name:
            model = smp.FPN(encoder_name="mit_b2", encoder_weights=None, in_channels=10, classes=5)
        elif "DeepLab" in name:
            model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=10, classes=5)
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        models[name] = model

    print("Evaluating test set and saving predictions...")
    all_targets = []
    all_preds = {name: [] for name in models.keys()}

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images, targets = images.to(device), targets.to(device)
            all_targets.append(targets.cpu().numpy().flatten())
            
            for name, model in models.items():
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds[name].append(preds.flatten())

            # Save visual panel for the first batch only
            if batch_idx == 0:
                print("Generating visual panel for sample patches...")
                num_samples = min(4, images.size(0))
                fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
                
                for i in range(num_samples):
                    rgb = get_rgb(images[i], metadata)
                    axes[i, 0].imshow(rgb)
                    axes[i, 0].set_title("RGB Sentinel-2")
                    axes[i, 0].axis("off")
                    
                    im = axes[i, 1].imshow(targets[i].cpu(), cmap=cmap, norm=norm, interpolation='nearest')
                    axes[i, 1].set_title("Ground Truth (WorldCover)")
                    axes[i, 1].axis("off")
                    
                    for j, (name, model) in enumerate(models.items()):
                        pred = all_preds[name][-1].reshape(-1, 256, 256)[i]
                        axes[i, j+2].imshow(pred, cmap=cmap, norm=norm, interpolation='nearest')
                        axes[i, j+2].set_title(f"{name} Pred")
                        axes[i, j+2].axis("off")
                        
                plt.tight_layout()
                fig.savefig(out_dir / "predictions_panel.png", dpi=300)
                plt.close(fig)

    print("Computing and plotting confusion matrices...")
    y_true = np.concatenate(all_targets)
    for name in models.keys():
        y_pred = np.concatenate(all_preds[name])
        plot_confusion_matrix(y_true, y_pred, classes, f"{name} Normalized Confusion Matrix", out_dir / f"cm_{name}.png")

    print(f"\u2728 All figures saved to {out_dir}")

if __name__ == "__main__":
    main()
