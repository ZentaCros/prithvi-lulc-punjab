"""Count parameters of every Prithvi configuration exactly as trained, and prove it by strict-loading its checkpoint.

Build order matches training/train_prithvi.py main(): load backbone -> adapt_input_to_11bands -> PrithviSegmentation
-> setup_training_params. Runs on CPU, one model at a time.
"""
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "training"))
from train_prithvi import load_prithvi_backbone, adapt_input_to_11bands, PrithviSegmentation, setup_training_params

CONFIGS = [  # (checkpoint dir, lora_rank, unfreeze_blocks)
    ("prithvi_v2_fix6_r8_seed42", 8, 8),
    ("prithvi_v2_fix6_r16_seed42", 16, 8),
    ("prithvi_v2_fix6_r32_seed42", 32, 8),
    ("prithvi_v2_fix6_seed42", 64, 8),
    ("prithvi_v2_fix6_lora_only_seed42", 64, 0),
    ("prithvi_v2_fix6_unfreeze_only_seed42", 0, 8),
]

rows = []
for name, rank, unf in CONFIGS:
    backbone, feat_dim, b_type = load_prithvi_backbone()
    adapt_input_to_11bands(backbone, b_type)
    model = PrithviSegmentation(backbone, feat_dim, b_type, num_classes=5)
    setup_training_params(model, lora_rank=rank, unfreeze_last_n=unf)
    state = torch.load(PROJECT_ROOT / "checkpoints" / name / "best_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_names = sorted({n.split("lora_A_")[1] for n, _ in model.named_modules() if n.startswith("lora_A_")})
    block0 = [n for n in lora_names if "blocks_0_" in n or n.startswith("encoder_blocks_0_") or "_0_" in n.split("blocks")[-1][:3]]
    rows.append({"config": name, "lora_rank": rank, "unfreeze_blocks": unf, "total": total, "trainable": trainable,
                 "pct": round(100 * trainable / total, 2), "lora_layers": len(lora_names), "lora_block0": block0})
    print(f"{name:40s} strict-load OK | total {total:,} | trainable {trainable:,} ({100*trainable/total:.2f}%) | "
          f"{trainable/1e6:.1f}M / {total/1e6:.1f}M | LoRA layers {len(lora_names)}")
    if block0:
        print(f"    LoRA targets in first block: {block0}")
    del model, backbone, state

out = PROJECT_ROOT / "results" / "prithvi_param_counts.json"
out.write_text(json.dumps(rows, indent=2))
print(f"Saved {out}")
