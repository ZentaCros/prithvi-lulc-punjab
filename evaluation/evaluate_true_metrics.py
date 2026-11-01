"""
Evaluate currently trained models on the perfectly aligned 2021 Test Set.
This reveals the TRUE metrics without the temporal gap penalty!
"""

import os
import sys
import json
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import LULCDataset
import segmentation_models_pytorch as smp

# Prithvi imports
from training.train_prithvi import (
    load_prithvi_backbone, PrithviSegmentation, adapt_input_to_11bands, setup_training_params, evaluate
)

class DummyCriterion:
    def __call__(self, x, y):
        return torch.tensor(0.0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    patches_dir = PROJECT_ROOT / "data" / "patches"
    with open(patches_dir / "metadata.json") as f:
        metadata = json.load(f)

    test_ds = LULCDataset(patches_dir, "test", metadata, augment=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)

    print("\n" + "="*60)
    print("  EVALUATING TRUE METRICS ON ALIGNED 2021 DATASET")
    print("="*60)

    results = {}
    criterion = DummyCriterion()

    models_to_test = {
        "SegFormer": PROJECT_ROOT / "checkpoints" / "segformer_seed123" / "best_model.pt",
        "DeepLabV3+": PROJECT_ROOT / "checkpoints" / "deeplabv3_seed456" / "best_model.pt",
        "Prithvi (v2)": PROJECT_ROOT / "checkpoints" / "prithvi_v2_seed42" / "best_model.pt",
    }

    for name, ckpt_path in models_to_test.items():
        print(f"\nEvaluating {name}...")
        try:
            if "Prithvi" in name:
                backbone, feat_dim, b_type = load_prithvi_backbone()
                adapt_input_to_11bands(backbone, b_type)
                model = PrithviSegmentation(backbone, feat_dim, b_type, metadata["num_classes"])
                setup_training_params(model, lora_rank=64, unfreeze_last_n=8)
            elif "SegFormer" in name:
                model = smp.FPN(encoder_name="mit_b2", encoder_weights=None, in_channels=10, classes=5)
            elif "DeepLab" in name:
                model = smp.DeepLabV3Plus(encoder_name="resnet101", encoder_weights=None, in_channels=10, classes=5)

            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.to(device)

            metrics = evaluate(model, test_loader, criterion, device, num_classes=5)
            results[name] = metrics
            
            print(f"  Overall Accuracy: {metrics['oa']*100:.2f}%")
            print(f"  Mean IoU:         {metrics['miou']*100:.2f}%")
            print(f"  Mean F1:          {metrics['mf1']*100:.2f}%")
        except Exception as e:
            print(f"  Failed: {e}")

    print("\n" + "="*60)
    print("  FINAL COMPARISON (TRUE METRICS)")
    print("="*60)
    print(f"  {'Model':<15} {'Accuracy':>10} {'mIoU':>10}")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"  {name:<15} {metrics['oa']*100:>9.2f}% {metrics['miou']*100:>9.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
