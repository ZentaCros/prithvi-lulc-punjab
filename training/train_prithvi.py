"""
LULC Lahore — Prithvi-EO-2.0 Fine-Tuning Script (v2 - Improved)
Improvements over v1:
  1. Uses ALL 10 bands (adapts input projection layer)
  2. Unfreezes last N transformer blocks for deeper fine-tuning
  3. Higher LoRA rank (64) on frozen layers
  4. Multi-scale UPerNet-style decoder

Usage:
    python training/train_prithvi.py --epochs 50 --batch_size 2
"""

import os
import sys
import json
import csv
import time
import argparse
import platform
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, jaccard_score

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.losses import CombinedLoss


def load_prithvi_backbone():
    """Load Prithvi-EO-2.0-300M backbone via terratorch."""
    try:
        from terratorch.registry import BACKBONE_REGISTRY
        backbone = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300",
            pretrained=True,
            num_frames=1,
            in_chans=6,  # Load with original 6 bands first
        )
        print("  [OK] Loaded Prithvi-EO-2.0-300M via TerraTorch")
        return backbone, 1024, "prithvi"
    except Exception as e:
        print(f"  [FAIL] TerraTorch failed: {e}")
        print("  Falling back to ViT-Base")
        import timm
        backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=True,
            num_classes=0, in_chans=6, img_size=256,
        )
        return backbone, 768, "vit_fallback"


def adapt_input_to_11bands(backbone, backbone_type):
    """
    Adapt the 6-band input projection to accept 11 bands (10 Sentinel-2 + NDVI).

    Prithvi-EO 2.0 pretrained bands (HLS order):
        [0]=B2(490nm), [1]=B3(560nm), [2]=B4(665nm),
        [3]=B8A(865nm), [4]=B11(1610nm), [5]=B12(2190nm)

    Our 11 input bands:
        [0]=B2, [1]=B3, [2]=B4, [3]=B5(705nm), [4]=B6(740nm), [5]=B7(783nm),
        [6]=B8(842nm), [7]=B8A(865nm), [8]=B11(1610nm), [9]=B12(2190nm), [10]=NDVI

    Spectral mapping:
        pos 0-2 (B2,B3,B4) <- exact pretrained match
        pos 3-5 (B5,B6,B7) <- wavelength-interpolated between B4 and B8A
        pos 6   (B8)       <- from B8A (spectrally adjacent, 842 vs 865 nm)
        pos 7   (B8A)      <- exact pretrained match
        pos 8   (B11)      <- exact pretrained match
        pos 9   (B12)      <- exact pretrained match
        pos 10  (NDVI)     <- from B8A (NIR component)
    """
    if backbone_type != "prithvi":
        return False

    patch_embed = None
    for name, module in backbone.named_modules():
        if hasattr(module, 'proj') and isinstance(module.proj, (nn.Conv2d, nn.Conv3d)):
            if module.proj.in_channels == 6:
                patch_embed = module
                break

    if patch_embed is None:
        if hasattr(backbone, 'patch_embed'):
            patch_embed = backbone.patch_embed
        else:
            for name, module in backbone.named_modules():
                if 'patch' in name.lower() and 'embed' in name.lower():
                    patch_embed = module
                    break

    if patch_embed is None or not hasattr(patch_embed, 'proj'):
        print("  [WARN] Could not find patch_embed.proj to adapt")
        return False

    old_proj = patch_embed.proj
    is_3d = isinstance(old_proj, nn.Conv3d)
    old_weight = old_proj.weight.data

    if is_3d:
        out_c, _, t, h, w = old_weight.shape
        new_proj = nn.Conv3d(11, out_c, kernel_size=old_proj.kernel_size,
                             stride=old_proj.stride, padding=old_proj.padding,
                             bias=old_proj.bias is not None)
        new_weight = torch.zeros(out_c, 11, t, h, w)
    else:
        out_c, _, h, w = old_weight.shape
        new_proj = nn.Conv2d(11, out_c, kernel_size=old_proj.kernel_size,
                             stride=old_proj.stride, padding=old_proj.padding,
                             bias=old_proj.bias is not None)
        new_weight = torch.zeros(out_c, 11, h, w)

    # Pretrained indices: 0=B2, 1=B3, 2=B4, 3=B8A, 4=B11, 5=B12
    PT_B2, PT_B3, PT_B4, PT_B8A, PT_B11, PT_B12 = 0, 1, 2, 3, 4, 5

    # Exact spectral matches
    new_weight[:, 0]  = old_weight[:, PT_B2]    # B2  <- B2
    new_weight[:, 1]  = old_weight[:, PT_B3]    # B3  <- B3
    new_weight[:, 2]  = old_weight[:, PT_B4]    # B4  <- B4
    new_weight[:, 7]  = old_weight[:, PT_B8A]   # B8A <- B8A
    new_weight[:, 8]  = old_weight[:, PT_B11]   # B11 <- B11
    new_weight[:, 9]  = old_weight[:, PT_B12]   # B12 <- B12

    # Red Edge interpolation (wavelength-based between B4=665nm and B8A=865nm)
    new_weight[:, 3] = 0.80 * old_weight[:, PT_B4] + 0.20 * old_weight[:, PT_B8A]  # B5 705nm
    new_weight[:, 4] = 0.625 * old_weight[:, PT_B4] + 0.375 * old_weight[:, PT_B8A]  # B6 740nm
    new_weight[:, 5] = 0.41 * old_weight[:, PT_B4] + 0.59 * old_weight[:, PT_B8A]  # B7 783nm

    # B8 (842nm) from B8A (865nm) - spectrally very close
    new_weight[:, 6]  = old_weight[:, PT_B8A]   # B8  <- B8A
    # NDVI from NIR
    new_weight[:, 10] = old_weight[:, PT_B8A]   # NDVI <- B8A

    new_proj.weight.data = new_weight
    if old_proj.bias is not None:
        new_proj.bias.data = old_proj.bias.data.clone()

    patch_embed.proj = new_proj
    proj_type = "Conv3d" if is_3d else "Conv2d"
    print(f"  [OK] Adapted patch_embed {proj_type}: 6 -> 11 bands (spectral mapping)")
    print(f"       B2,B3,B4,B8A,B11,B12 = exact match | B5-B7 = interpolated | B8,NDVI from B8A")
    return True


class LightweightDecoder(nn.Module):
    """
    Efficient decoder: reduces channels at backbone resolution (16x16),
    then upsamples. Avoids heavy ops on full 256x256 with 1024 channels.
    """
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        # Process at low-res (16x16) with 1024 channels
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Upsample 4x (16->64)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Upsample 4x (64->256)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x, target_size):
        x = self.reduce(x)   # (B, 256, 16, 16)
        x = self.up1(x)      # (B, 128, 64, 64)
        x = self.up2(x)      # (B, 64, 256, 256)
        if x.shape[2] != target_size[0] or x.shape[3] != target_size[1]:
            x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.head(x)


class PrithviSegmentation(nn.Module):
    """Prithvi backbone + lightweight decoder."""

    def __init__(self, backbone, feat_dim, backbone_type, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.backbone_type = backbone_type
        self.feat_dim = feat_dim
        self.decoder = LightweightDecoder(feat_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone(x)

        if isinstance(features, (list, tuple)):
            features = features[-1]

        if features.dim() == 3:
            N, D = features.shape[1], features.shape[2]
            h = w = int(N ** 0.5)
            if h * w != N:
                features = features[:, 1:, :]
                N = N - 1
                h = w = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, D, h, w)

        # DON'T upsample here - let decoder handle it efficiently
        return self.decoder(features, (H, W))


def setup_training_params(model, lora_rank=64, unfreeze_last_n=8):
    """
    Fine-tuning strategy:
    - Fully unfreeze last N transformer blocks
    - Apply LoRA to remaining frozen attention layers
    - Decoder always fully trainable
    - Input projection always trainable
    """
    # Step 1: Freeze entire backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Step 2: Unfreeze last N transformer blocks
    blocks_unfrozen = 0
    # Try to find transformer blocks
    block_list = None
    for name, module in model.backbone.named_modules():
        if hasattr(module, '__len__') and name in ['blocks', 'encoder.blocks']:
            block_list = module
            break

    if block_list is None:
        # Search for block-like sequential containers
        for name, module in model.backbone.named_children():
            if 'block' in name.lower() or 'layer' in name.lower():
                block_list = module
                break

    if block_list is not None and hasattr(block_list, '__len__'):
        total_blocks = len(block_list)
        start_unfreeze = max(0, total_blocks - unfreeze_last_n)
        for i in range(start_unfreeze, total_blocks):
            for param in block_list[i].parameters():
                param.requires_grad = True
            blocks_unfrozen += 1
        print(f"  [OK] Unfroze last {blocks_unfrozen}/{total_blocks} transformer blocks")
    else:
        # Fallback: unfreeze last N named children that look like blocks
        block_names = []
        for name, child in model.backbone.named_children():
            if any(k in name.lower() for k in ['block', 'layer', 'encoder']):
                block_names.append((name, child))

        if block_names:
            for name, child in block_names[-unfreeze_last_n:]:
                for param in child.parameters():
                    param.requires_grad = True
                blocks_unfrozen += 1
            print(f"  [OK] Unfroze last {blocks_unfrozen} backbone components")
        else:
            print(f"  [WARN] Could not identify transformer blocks, unfreezing last 30% of params")
            all_params = list(model.backbone.parameters())
            unfreeze_count = max(1, len(all_params) * 30 // 100)
            for param in all_params[-unfreeze_count:]:
                param.requires_grad = True

    # Step 3: Always unfreeze input projection (patch_embed)
    for name, param in model.backbone.named_parameters():
        if 'patch_embed' in name or 'patch' in name.lower():
            param.requires_grad = True

    # Step 4: Apply LoRA to remaining FROZEN attention layers
    if lora_rank == 0:
        print(f"  [OK] LoRA disabled (rank=0)")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        pct = 100 * trainable / total
        print(f"  Parameters: {total:,} total | {trainable:,} trainable ({pct:.1f}%) | {frozen:,} frozen")
        return

    targets = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Linear) and any(
            k in name for k in ["qkv", "proj", "fc1", "fc2", "attn"]
        ):
            has_frozen_params = any(not p.requires_grad for p in module.parameters())
            if has_frozen_params:
                targets.append((name, module))

    lora_params = []
    for name, module in targets:
        in_f, out_f = module.in_features, module.out_features
        lora_A = nn.Linear(in_f, lora_rank, bias=False)
        lora_B = nn.Linear(lora_rank, out_f, bias=False)
        nn.init.kaiming_uniform_(lora_A.weight)
        nn.init.zeros_(lora_B.weight)

        safe_name = name.replace(".", "_")
        model.register_module(f"lora_A_{safe_name}", lora_A)
        model.register_module(f"lora_B_{safe_name}", lora_B)
        lora_params.extend([lora_A.weight, lora_B.weight])

        original_forward = module.forward
        def make_lora_fwd(orig, A, B):
            def fwd(x):
                return orig(x) + B(A(x))
            return fwd
        module.forward = make_lora_fwd(original_forward, lora_A, lora_B)

    print(f"  [OK] LoRA (rank={lora_rank}) applied to {len(targets)} frozen layers")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    pct = 100 * trainable / total
    print(f"  Parameters: {total:,} total | {trainable:,} trainable ({pct:.1f}%) | {frozen:,} frozen")


def compute_metrics(preds, targets, num_classes=5):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    oa = (preds_flat == targets_flat).sum() / len(targets_flat)
    iou = jaccard_score(targets_flat, preds_flat, average=None,
                         labels=range(num_classes), zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average=None,
                   labels=range(num_classes), zero_division=0)
    return {"oa": float(oa), "miou": float(np.mean(iou)), "mf1": float(np.mean(f1)),
            "iou_per_class": [float(x) for x in iou], "f1_per_class": [float(x) for x in f1]}


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(images)
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
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_targets.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_metrics(all_preds, all_targets, num_classes)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  PRITHVI-EO-2.0 v2 (IMPROVED)")
    print(f"  ALL 10 bands | Unfrozen blocks | LoRA-64 | Multi-scale")
    print(f"  Device: {device} | Seed: {args.seed}")
    print(f"{'='*60}")

    patches_dir = PROJECT_ROOT / "data" / "patches"
    exp_name = os.environ.get("LULC_EXP_NAME", f"prithvi_v2_seed{args.seed}")
    results_dir = PROJECT_ROOT / "results" / exp_name
    ckpt_dir = PROJECT_ROOT / "checkpoints" / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(patches_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Use ALL 10 bands - no band selection wrapper needed!
    print("\nLoading data (ALL 10 bands)...")
    from training.dataset import LULCDataset
    num_workers = 0 if platform.system() == "Windows" else 4

    train_ds = LULCDataset(patches_dir, "train", metadata, augment=True)
    val_ds = LULCDataset(patches_dir, "val", metadata, augment=False)
    test_ds = LULCDataset(patches_dir, "test", metadata, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Using ALL 11 bands (10 Sentinel-2 + NDVI, vs 6 in v1)")

    # Load backbone and adapt to 11 bands
    print("\nLoading Prithvi backbone...")
    backbone, feat_dim, backbone_type = load_prithvi_backbone()

    print("\nAdapting input to 11 bands...")
    adapt_input_to_11bands(backbone, backbone_type)

    # Build model
    num_classes = metadata["num_classes"]
    model = PrithviSegmentation(backbone, feat_dim, backbone_type, num_classes)

    # Setup fine-tuning (unfreeze + LoRA)
    print(f"\nSetting up fine-tuning strategy...")
    setup_training_params(model, lora_rank=args.lora_rank,
                         unfreeze_last_n=args.unfreeze_blocks)

    model = model.to(device)

    # Loss
    class_weights = metadata.get("class_weights", None)
    criterion = CombinedLoss(num_classes=num_classes, class_weights=class_weights).to(device)

    # Optimizer with differential learning rates
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'decoder' not in n and 'lora' not in n]
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and 'lora' in n]
    decoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and 'decoder' in n]

    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.1},  # Lower LR for backbone
        {"params": lora_params, "lr": args.lr * 0.5},       # Medium LR for LoRA
        {"params": decoder_params, "lr": args.lr},           # Full LR for decoder
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=[args.lr*0.1, args.lr*0.5, args.lr],
                          total_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_miou", "val_mf1", "val_oa"])

    best_miou = 0.0
    patience_counter = 0
    class_names = metadata["class_names"]

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  LR: backbone={args.lr*0.1:.1e}, lora={args.lr*0.5:.1e}, decoder={args.lr:.1e}")
    print(f"  Batch: {args.batch_size} | Patience: {args.patience}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        elapsed = time.time() - t0

        val_miou = val_metrics["miou"]
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{val_metrics['loss']:.4f}",
                                    f"{val_miou:.4f}", f"{val_metrics['mf1']:.4f}", f"{val_metrics['oa']:.4f}"])

        improved = ""
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            improved = " *BEST*"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_metrics['loss']:.4f} | "
              f"mIoU: {val_miou:.4f} | mF1: {val_metrics['mf1']:.4f} | "
              f"OA: {val_metrics['oa']:.4f} | {elapsed:.1f}s{improved}")

        if epoch % 10 == 0 or epoch == args.epochs:
            print("         Per-class IoU: " +
                  " | ".join(f"{n}: {v:.3f}" for n, v in zip(class_names, val_metrics["iou_per_class"])))

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Final test
    print(f"\n{'='*60}")
    print("  FINAL TEST EVALUATION (Prithvi-EO-2.0 v2)")
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

    with open(results_dir / "test_results.json", 'w') as f:
        json.dump({
            "model": "prithvi_eo_v2_300_improved",
            "version": "v2",
            "seed": args.seed,
            "best_val_miou": best_miou,
            "lora_rank": args.lora_rank,
            "unfreeze_blocks": args.unfreeze_blocks,
            "bands_used": 10,
            **test_metrics,
            "class_names": class_names,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE | Best Val mIoU: {best_miou:.4f} | Test mIoU: {test_metrics['miou']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prithvi-EO-2.0 v2 LULC Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--unfreeze_blocks", type=int, default=8)
    args = parser.parse_args()
    main(args)
