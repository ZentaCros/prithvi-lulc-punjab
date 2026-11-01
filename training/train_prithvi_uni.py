"""
LULC Lahore — Prithvi-EO-2.0 Heavy Fine-Tuning Script (Optimized for RTX 4080 16GB)
Improvements:
  1. Uses ALL 10 bands (adapts input projection layer)
  2. Unfreezes last 12 transformer blocks (massive learning capacity)
  3. High LoRA rank (64) on remaining frozen layers
  4. Multi-scale ASPP-style decoder (high capacity)

Usage on Uni PC (Ubuntu):
    python train_prithvi_uni.py --epochs 50 --batch_size 4
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

# Assuming the user runs this from the Capstone_2_LULC_Segmentation folder on Uni PC
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.losses import CombinedLoss

def load_prithvi_backbone():
    try:
        from terratorch.registry import BACKBONE_REGISTRY
        backbone = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_300",
            pretrained=True,
            num_frames=1,
            in_chans=6,
        )
        print("  [OK] Loaded Prithvi-EO-2.0-300M via TerraTorch")
        return backbone, 1024, "prithvi"
    except Exception as e:
        print(f"  [FAIL] TerraTorch failed: {e}")
        sys.exit(1)

def adapt_input_to_10bands(backbone, backbone_type):
    if backbone_type == "prithvi":
        patch_embed = None
        for name, module in backbone.named_modules():
            if hasattr(module, 'proj') and isinstance(module.proj, (nn.Conv2d, nn.Conv3d)):
                if module.proj.in_channels == 6:
                    patch_embed = module
                    break

        if patch_embed is not None and hasattr(patch_embed, 'proj'):
            old_proj = patch_embed.proj
            is_3d = isinstance(old_proj, nn.Conv3d)

            if is_3d:
                old_weight = old_proj.weight.data
                out_c = old_weight.shape[0]
                t, h, w = old_weight.shape[2], old_weight.shape[3], old_weight.shape[4]
                new_proj = nn.Conv3d(10, out_c, kernel_size=(old_proj.kernel_size),
                                     stride=old_proj.stride, padding=old_proj.padding,
                                     bias=old_proj.bias is not None)
                new_weight = torch.zeros(out_c, 10, t, h, w)
            else:
                old_weight = old_proj.weight.data
                out_c = old_weight.shape[0]
                h, w = old_weight.shape[2], old_weight.shape[3]
                new_proj = nn.Conv2d(10, out_c, kernel_size=old_proj.kernel_size,
                                     stride=old_proj.stride, padding=old_proj.padding,
                                     bias=old_proj.bias is not None)
                new_weight = torch.zeros(out_c, 10, h, w)

            new_weight[:, :6] = old_weight
            new_weight[:, 6] = old_weight[:, 3]
            new_weight[:, 7] = old_weight[:, 4]
            new_weight[:, 8] = old_weight[:, 5]
            new_weight[:, 9] = old_weight[:, 5]

            new_proj.weight.data = new_weight
            if old_proj.bias is not None:
                new_proj.bias.data = old_proj.bias.data.clone()

            patch_embed.proj = new_proj
            print(f"  [OK] Adapted patch_embed: 6 -> 10 bands")
            return True
    return False

class MultiScaleDecoder(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.scale1 = nn.Sequential(nn.Conv2d(feat_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.scale2 = nn.Sequential(nn.Conv2d(feat_dim, 256, 3, padding=2, dilation=2), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.scale3 = nn.Sequential(nn.Conv2d(feat_dim, 256, 3, padding=4, dilation=4), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(feat_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(256 * 4, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        s1, s2, s3, s4 = self.scale1(x), self.scale2(x), self.scale3(x), self.scale4(x)
        s4 = nn.functional.interpolate(s4, size=(H, W), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([s1, s2, s3, s4], dim=1))

class PrithviSegmentation(nn.Module):
    def __init__(self, backbone, feat_dim, backbone_type, num_classes=5):
        super().__init__()
        self.backbone = backbone
        self.decoder = MultiScaleDecoder(feat_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        features = self.backbone(x)
        if isinstance(features, (list, tuple)): features = features[-1]
        
        if features.dim() == 3:
            N, D = features.shape[1], features.shape[2]
            h = w = int(N ** 0.5)
            if h * w != N:
                features = features[:, 1:, :]
                N = N - 1
                h = w = int(N ** 0.5)
            features = features.permute(0, 2, 1).reshape(B, D, h, w)
            
        if features.shape[2] != H or features.shape[3] != W:
            features = nn.functional.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)
        return self.decoder(features)

def setup_training_params(model, lora_rank=64, unfreeze_last_n=12):
    for param in model.backbone.parameters():
        param.requires_grad = False

    blocks_unfrozen = 0
    block_list = None
    for name, module in model.backbone.named_modules():
        if hasattr(module, '__len__') and name in ['blocks', 'encoder.blocks']:
            block_list = module
            break
            
    if block_list is not None and hasattr(block_list, '__len__'):
        total_blocks = len(block_list)
        start_unfreeze = max(0, total_blocks - unfreeze_last_n)
        for i in range(start_unfreeze, total_blocks):
            for param in block_list[i].parameters():
                param.requires_grad = True
            blocks_unfrozen += 1
    
    for name, param in model.backbone.named_parameters():
        if 'patch_embed' in name or 'patch' in name.lower():
            param.requires_grad = True

    targets = []
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in ["qkv", "proj", "fc1", "fc2", "attn"]):
            if any(not p.requires_grad for p in module.parameters()):
                targets.append((name, module))

    for name, module in targets:
        in_f, out_f = module.in_features, module.out_features
        lora_A = nn.Linear(in_f, lora_rank, bias=False)
        lora_B = nn.Linear(lora_rank, out_f, bias=False)
        nn.init.kaiming_uniform_(lora_A.weight)
        nn.init.zeros_(lora_B.weight)
        
        safe_name = name.replace(".", "_")
        model.register_module(f"lora_A_{safe_name}", lora_A)
        model.register_module(f"lora_B_{safe_name}", lora_B)
        
        original_forward = module.forward
        def make_lora_fwd(orig, A, B):
            def fwd(x): return orig(x) + B(A(x))
            return fwd
        module.forward = make_lora_fwd(original_forward, lora_A, lora_B)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [OK] Unfroze last {blocks_unfrozen} blocks. LoRA(rank={lora_rank}) on {len(targets)} layers.")
    print(f"  Parameters: {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")

def compute_metrics(preds, targets, num_classes=5):
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    oa = (preds_flat == targets_flat).sum() / len(targets_flat)
    iou = jaccard_score(targets_flat, preds_flat, average=None, labels=range(num_classes), zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average=None, labels=range(num_classes), zero_division=0)
    return {"oa": float(oa), "miou": float(np.mean(iou)), "mf1": float(np.mean(f1)),
            "iou_per_class": [float(x) for x in iou]}

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
        if scheduler: scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=5):
    model.eval()
    total_loss, all_preds, all_targets = 0, [], []
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\n  PRITHVI-EO-2.0 v2 (RTX 4080 16GB Edition)\n  Device: {device}\n{'='*60}")

    patches_dir = PROJECT_ROOT / "data" / "patches"
    exp_name = f"prithvi_v2_heavy_seed{args.seed}"
    results_dir = PROJECT_ROOT / "results" / exp_name
    ckpt_dir = PROJECT_ROOT / "checkpoints" / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(patches_dir / "metadata.json") as f:
        metadata = json.load(f)

    from training.dataset import LULCDataset
    num_workers = 4 if platform.system() != "Windows" else 0
    train_ds = LULCDataset(patches_dir, "train", metadata, augment=True)
    val_ds = LULCDataset(patches_dir, "val", metadata, augment=False)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    backbone, feat_dim, b_type = load_prithvi_backbone()
    adapt_input_to_10bands(backbone, b_type)
    model = PrithviSegmentation(backbone, feat_dim, b_type, metadata["num_classes"])
    setup_training_params(model, lora_rank=args.lora_rank, unfreeze_last_n=args.unfreeze_blocks)
    model = model.to(device)

    criterion = CombinedLoss(num_classes=metadata["num_classes"], class_weights=metadata.get("class_weights")).to(device)
    
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and 'decoder' not in n and 'lora' not in n]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora' in n]
    decoder_params = [p for n, p in model.named_parameters() if p.requires_grad and 'decoder' in n]
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": lora_params, "lr": args.lr * 0.5},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=1e-4)
    
    scheduler = OneCycleLR(optimizer, max_lr=[args.lr*0.1, args.lr*0.5, args.lr], total_steps=len(train_loader)*args.epochs)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_miou, patience_counter = 0.0, 0
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_miou", "val_mf1", "val_oa"])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device, metadata["num_classes"])
        
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            imp = " *BEST*"
        else:
            patience_counter += 1
            imp = ""
            
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_metrics['loss'], val_metrics['miou'], val_metrics['mf1'], val_metrics['oa']])
            
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f}/{val_metrics['loss']:.4f} | mIoU: {val_metrics['miou']:.4f} | {time.time()-t0:.1f}s{imp}")
        if patience_counter >= args.patience: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--unfreeze_blocks", type=int, default=12)
    main(parser.parse_args())
