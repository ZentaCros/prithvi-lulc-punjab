"""
Multi-Seed Training — All 4 Models × 3 Seeds = 12 Runs

Trains U-Net, MiT-B2+FPN (formerly SegFormer), DeepLabV3+, and Prithvi-EO-2.0
with seeds [42, 123, 456] for statistical robustness.

Usage:
    python training/run_multiseed.py                    # Run all missing experiments
    python training/run_multiseed.py --models unet      # Run only U-Net
    python training/run_multiseed.py --dry_run           # Print what would run
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

MODELS = ["unet", "segformer", "deeplabv3"]
SEEDS = [42, 123, 7]
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4

PRITHVI_BATCH_SIZE = 2
PRITHVI_LR = 1e-4
PRITHVI_LORA_RANK = 64
PRITHVI_UNFREEZE = 8


def check_existing(model, seed):
    """Check if a run already has results."""
    if model == "prithvi":
        results_path = PROJECT_ROOT / "results" / f"prithvi_v2_seed{seed}" / "test_results.json"
    else:
        results_path = PROJECT_ROOT / "results" / f"{model}_seed{seed}" / "test_results.json"
    return results_path.exists()


def run_smp_model(model, seed):
    """Train a segmentation_models_pytorch model."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train.py"),
        "--model", model,
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
    ]
    return subprocess.run(cmd)


def run_prithvi(seed):
    """Train Prithvi-EO-2.0 with LoRA."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train_prithvi.py"),
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--batch_size", str(PRITHVI_BATCH_SIZE),
        "--lr", str(PRITHVI_LR),
        "--lora_rank", str(PRITHVI_LORA_RANK),
        "--unfreeze_blocks", str(PRITHVI_UNFREEZE),
    ]
    return subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed Training Suite")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["unet", "segformer", "deeplabv3", "prithvi"],
                        help="Models to train (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Seeds to use (default: 42 123 456)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without running")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    args = parser.parse_args()

    models = args.models or (MODELS + ["prithvi"])
    seeds = args.seeds or SEEDS

    # Build run plan
    runs = []
    for model in models:
        for seed in seeds:
            exists = check_existing(model, seed)
            if exists and not args.force:
                runs.append((model, seed, "SKIP (exists)"))
            else:
                runs.append((model, seed, "PENDING"))

    # Print plan
    print("=" * 60)
    print("  MULTI-SEED TRAINING PLAN")
    print("=" * 60)
    pending = sum(1 for _, _, s in runs if s == "PENDING")
    skipping = sum(1 for _, _, s in runs if s != "PENDING")
    print(f"  Total: {len(runs)} | Pending: {pending} | Skipping: {skipping}")
    print()
    for model, seed, status in runs:
        print(f"  {model:<12} seed {seed:<5} {status}")
    print()

    if args.dry_run:
        print("  (dry run — nothing executed)")
        return

    if pending == 0:
        print("  All runs already complete. Use --force to re-run.")
        return

    # Execute
    results = []
    for model, seed, status in runs:
        if status != "PENDING":
            continue

        print(f"\n{'=' * 60}")
        print(f"  RUNNING: {model} | seed {seed}")
        print(f"{'=' * 60}")

        t0 = time.time()
        if model == "prithvi":
            result = run_prithvi(seed)
        else:
            result = run_smp_model(model, seed)
        elapsed = (time.time() - t0) / 60

        ok = result.returncode == 0
        results.append((model, seed, ok, elapsed))
        print(f"  {'OK' if ok else 'FAILED'} in {elapsed:.1f} min")

    # Summary
    print(f"\n{'=' * 60}")
    print("  MULTI-SEED TRAINING SUMMARY")
    print(f"{'=' * 60}")
    for model, seed, ok, elapsed in results:
        status = "OK" if ok else "FAILED"
        print(f"  {model:<12} seed {seed:<5} {status:<8} {elapsed:.1f} min")
    ok_count = sum(1 for _, _, ok, _ in results if ok)
    print(f"\n  {ok_count}/{len(results)} experiments completed successfully.")


if __name__ == "__main__":
    main()
