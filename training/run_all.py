"""
LULC Lahore — Run All Experiments
Trains SegFormer and DeepLabV3+ with 3 seeds each.

Usage (on uni PC with RTX 4080):
    python training/run_all.py
"""

import subprocess
import sys
import time

MODELS = ["segformer", "deeplabv3"]
SEEDS = [42, 123, 456]
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4

def run_experiment(model, seed):
    cmd = [
        sys.executable, "training/train.py",
        "--model", model,
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
    ]

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {model} | Seed {seed}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = (time.time() - start) / 60

    status = "OK" if result.returncode == 0 else f"FAIL (exit {result.returncode})"
    print(f"  Finished: {status} in {elapsed:.1f} min")
    return result.returncode

def main():
    print("="*60)
    print("  LULC LAHORE — FULL EXPERIMENT SUITE")
    print(f"  Models: {MODELS}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total runs: {len(MODELS) * len(SEEDS)}")
    print("="*60)

    results = []
    for model in MODELS:
        for seed in SEEDS:
            rc = run_experiment(model, seed)
            results.append((model, seed, rc))

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for model, seed, rc in results:
        status = "OK" if rc == 0 else "FAILED"
        print(f"  {model} seed {seed}: {status}")

    ok = sum(1 for _, _, rc in results if rc == 0)
    print(f"\n  {ok}/{len(results)} experiments completed successfully.")

if __name__ == "__main__":
    main()
