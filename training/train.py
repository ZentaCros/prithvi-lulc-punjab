"""
LULC Lahore — Main Training Script
Trains SegFormer or DeepLabV3+ on Lahore LULC patches.

Usage:
    python training/train.py --model segformer --epochs 50 --batch_size 8
    python training/train.py --model deeplabv3 --epochs 50 --batch_size 8
    python training/train.py --model segformer --epochs 1 --batch_size 2 --quick_test
"""

import os
import sys
import json
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, jaccard_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from training.losses import CombinedLoss


def create_model(model_name, in_channels=11, num_classes=5):
    """Factory function to create the specified model."""
    import segmentation_models_pytorch as smp

    if model_name == "unet":
        # U-Net with EfficientNet-B4 encoder (original proposal baseline)
        model = smp.Unet(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        print(f"  Created U-Net (EfficientNet-B4) | in={in_channels}, classes={num_classes}")

    elif model_name == "segformer":
        # MixTransformer-B2 encoder + FPN decoder (lightweight, SegFormer-style)
        model = smp.FPN(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        print(f"  Created SegFormer (MIT-B2 + FPN) | in={in_channels}, classes={num_classes}")

    elif model_name == "deeplabv3":
        # ResNet-101 encoder + ASPP decoder
        model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        print(f"  Created DeepLabV3+ (ResNet-101) | in={in_channels}, classes={num_classes}")

    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'unet', 'segformer', or 'deeplabv3'.")

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    return model


def compute_metrics(preds, targets, num_classes=5):
    """Compute per-class and mean IoU, F1, and overall accuracy."""
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    # Overall accuracy
    oa = (preds_flat == targets_flat).sum() / len(targets_flat)

    # Per-class IoU and F1
    iou_per_class = jaccard_score(targets_flat, preds_flat, average=None,
                                   labels=range(num_classes), zero_division=0)
    f1_per_class = f1_score(targets_flat, preds_flat, average=None,
                             labels=range(num_classes), zero_division=0)

    return {
        "oa": float(oa),
        "miou": float(np.mean(iou_per_class)),
        "mf1": float(np.mean(f1_per_class)),
        "iou_per_class": [float(x) for x in iou_per_class],
        "f1_per_class": [float(x) for x in f1_per_class],
    }


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(images)  # [B, C, H, W]
            loss = criterion(outputs, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=5):
    """Evaluate model. Returns loss and metrics dict."""
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = compute_metrics(all_preds, all_targets, num_classes)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main(args):
    # Seed + deterministic flags
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  LULC LAHORE TRAINING")
    print(f"  Model: {args.model} | Device: {device} | Seed: {args.seed}")
    print(f"{'='*60}")

    # Paths
    patches_dir = PROJECT_ROOT / "data" / "patches"
    exp_name = f"{args.model}_seed{args.seed}"
    results_dir = PROJECT_ROOT / "results" / exp_name
    ckpt_dir = PROJECT_ROOT / "checkpoints" / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("\nLoading data...")
    import platform
    num_workers = 0 if (args.quick_test or platform.system() == "Windows") else 4
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        patches_dir, batch_size=args.batch_size, num_workers=num_workers
    )

    # Model
    print("\nCreating model...")
    num_classes = metadata["num_classes"]
    num_bands = metadata["num_bands"]
    model = create_model(args.model, in_channels=num_bands, num_classes=num_classes)
    model = model.to(device)

    # Loss with class weights
    class_weights = metadata.get("class_weights", None)
    if class_weights:
        print(f"  Class weights: {[f'{w:.3f}' for w in class_weights]}")
    criterion = CombinedLoss(
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps) if not args.quick_test else None

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # CSV log
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_miou", "val_mf1", "val_oa"])

    # Training loop
    best_miou = 0.0
    patience_counter = 0
    patience = args.patience
    class_names = metadata["class_names"]

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Patience: {patience} | LR: {args.lr} | Batch: {args.batch_size}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        elapsed = time.time() - t0
        val_loss = val_metrics["loss"]
        val_miou = val_metrics["miou"]
        val_mf1 = val_metrics["mf1"]
        val_oa = val_metrics["oa"]

        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                           f"{val_miou:.4f}", f"{val_mf1:.4f}", f"{val_oa:.4f}"])

        # Checkpoint
        improved = ""
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            improved = " *BEST*"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"mIoU: {val_miou:.4f} | mF1: {val_mf1:.4f} | "
              f"OA: {val_oa:.4f} | {elapsed:.1f}s{improved}")

        # Per-class IoU (every 10 epochs)
        if epoch % 10 == 0 or epoch == args.epochs:
            print("         Per-class IoU: " +
                  " | ".join(f"{n}: {v:.3f}" for n, v in zip(class_names, val_metrics["iou_per_class"])))

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Final test evaluation
    print(f"\n{'='*60}")
    print("  FINAL TEST EVALUATION")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(ckpt_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    print(f"\n  Overall Accuracy: {test_metrics['oa']:.4f}")
    print(f"  Mean IoU:         {test_metrics['miou']:.4f}")
    print(f"  Mean F1:          {test_metrics['mf1']:.4f}")
    print(f"\n  Per-Class Results:")
    print(f"  {'Class':<15} {'IoU':>8} {'F1':>8}")
    print(f"  {'-'*33}")
    for i, name in enumerate(class_names):
        print(f"  {name:<15} {test_metrics['iou_per_class'][i]:>8.4f} {test_metrics['f1_per_class'][i]:>8.4f}")

    # Save test results
    test_path = results_dir / "test_results.json"
    with open(test_path, 'w') as f:
        json.dump({
            "model": args.model,
            "seed": args.seed,
            "best_val_miou": best_miou,
            **test_metrics,
            "class_names": class_names,
        }, f, indent=2)
    print(f"\n  Results saved to {test_path}")

    # Save final summary
    summary_path = results_dir / "final_summary.csv"
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["model", "seed", "test_miou", "test_mf1", "test_oa", "best_val_miou"])
        w.writerow([args.model, args.seed, f"{test_metrics['miou']:.4f}",
                    f"{test_metrics['mf1']:.4f}", f"{test_metrics['oa']:.4f}", f"{best_miou:.4f}"])

    print(f"\n{'='*60}")
    print(f"  DONE | Best Val mIoU: {best_miou:.4f} | Test mIoU: {test_metrics['miou']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LULC Lahore Training")
    parser.add_argument("--model", type=str, default="unet",
                        choices=["unet", "segformer", "deeplabv3"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--quick_test", action="store_true",
                        help="Quick test mode (no scheduler, 0 workers)")
    args = parser.parse_args()
    main(args)
