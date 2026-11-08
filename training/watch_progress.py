"""
Capstone 2 - GPU Task Progress Monitor
Live dashboard for all training, ablation, and evaluation tasks.
Usage: python training/watch_progress.py [--refresh 10]
"""
import os, sys, csv, time, json, argparse
from pathlib import Path

PROJECT = Path(r"D:\MY\Study\Research\Paper 2")
RESULTS = PROJECT / "results"
CHECKPOINTS = PROJECT / "checkpoints"
OUTPUTS = PROJECT / "outputs" / "figures"
TOTAL_EPOCHS = 50

C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[92m", "cyan": "\033[96m", "yellow": "\033[93m",
    "red": "\033[91m", "magenta": "\033[95m", "white": "\033[97m",
    "bg_green": "\033[42m", "bg_yellow": "\033[43m", "bg_gray": "\033[100m",
}

# All experiments grouped by phase
PHASES = [
    ("Multi-Seed Benchmark", [
        ("unet_seed42",       "U-Net s42"),
        ("unet_seed123",      "U-Net s123"),
        ("unet_seed7",        "U-Net s7"),
        ("segformer_seed42",  "MiT-B2 s42"),
        ("segformer_seed123", "MiT-B2 s123"),
        ("segformer_seed7",   "MiT-B2 s7"),
        ("deeplabv3_seed42",  "DLv3+ s42"),
        ("deeplabv3_seed123", "DLv3+ s123"),
        ("deeplabv3_seed7",   "DLv3+ s7"),
        ("prithvi_v2_seed42", "Prithvi s42"),
        ("prithvi_v2_seed123","Prithvi s123"),
        ("prithvi_v2_seed7",  "Prithvi s7"),
    ]),
    ("LoRA Rank Ablation", [
        ("prithvi_lora_r8_seed42",  "LoRA r=8"),
        ("prithvi_lora_r16_seed42", "LoRA r=16"),
        ("prithvi_lora_r32_seed42", "LoRA r=32"),
    ]),
    ("Review Fix: Attribution Ablation", [
        ("prithvi_v2_lora_only_r64_seed42",     "Fix #4a LoRA-only"),
        ("prithvi_v2_unfreeze_only_b8_seed42",  "Fix #4b Unfreeze-only"),
    ]),
    ("Review Fix: Band Init", [
        ("prithvi_v2_bandfix_seed42",           "Fix #6 BandFix s42"),
        ("prithvi_v2_bandfix_lora_r8_seed42",   "Fix #6 BandFix r=8"),
        ("prithvi_v2_bandfix_lora_r16_seed42",  "Fix #6 BandFix r=16"),
        ("prithvi_v2_bandfix_lora_r32_seed42",  "Fix #6 BandFix r=32"),
        ("prithvi_v2_bandfix_lora_r64_seed42",  "Fix #6 BandFix r=64"),
    ]),
]

# Non-training tasks (check by file existence)
FILE_TASKS = [
    ("Evaluation Outputs", [
        (OUTPUTS / "cm_U-Net.png",             "Fix #8 U-Net CM"),
        (OUTPUTS / "cm_MiT-B2plusFPN.png",     "Fix #8 MiT-B2 CM"),
        (OUTPUTS / "cm_DeepLabV3plus.png",     "Fix #8 DLv3+ CM"),
        (OUTPUTS / "cm_Prithvi-EO_20.png",     "Fix #8 Prithvi CM"),
    ]),
    ("Inference & Change Detection", [
        (PROJECT / "data" / "inference" / "gujranwala" / "gujranwala_sentinel2_2021_dryseason_10bands.tif",
                                               "Fix #5 Dry-season GEE"),
        (RESULTS / "pipeline" / "gujranwala" / "evaluation_gujranwala_dryseason.json",
                                               "Fix #5 Dry-season eval"),
        (RESULTS / "pipeline" / "lahore" / "lulc_lahore_prithvi_only_2018.tif",
                                               "Fix #2 Prithvi TIF 2018"),
        (RESULTS / "pipeline" / "lahore" / "lulc_lahore_prithvi_only_2026.tif",
                                               "Fix #2 Prithvi TIF 2026"),
    ]),
]


def read_training_run(folder):
    """Read training run status from results or checkpoints."""
    d = RESULTS / folder
    test_file = d / "test_results.json"
    csv_file = d / "metrics.csv"

    if test_file.exists():
        with open(test_file) as f:
            r = json.load(f)
        return {"epoch": TOTAL_EPOCHS, "done": True,
                "miou": r.get("miou", 0), "oa": r.get("oa", 0)}

    if csv_file.exists():
        try:
            rows = []
            with open(csv_file) as f:
                for row in csv.DictReader(f):
                    rows.append(row)
            if rows:
                last = rows[-1]
                best_miou = max(float(r.get("val_miou", 0)) for r in rows)
                ep = int(last["epoch"])
                return {"epoch": ep, "done": False, "miou": best_miou, "oa": 0}
        except Exception:
            return {"epoch": 0, "done": False, "miou": 0, "oa": 0}

    ckpt = CHECKPOINTS / folder / "best_model.pt"
    if ckpt.exists():
        return {"epoch": TOTAL_EPOCHS, "done": True, "miou": 0, "oa": 0}

    return None


def bar(pct, width=25, done=False):
    filled = int(width * pct)
    if done:
        return f"{C['bg_green']}{' ' * width}{C['reset']}"
    elif pct > 0:
        return (f"{C['bg_green']}{' ' * filled}"
                f"{C['bg_yellow']}{' ' * 1}"
                f"{C['bg_gray']}{' ' * max(0, width - filled - 1)}{C['reset']}")
    else:
        return f"{C['bg_gray']}{' ' * width}{C['reset']}"


def fmt_pct(val):
    if val > 1:
        return f"{val:.2f}%"
    elif val > 0:
        return f"{val*100:.2f}%"
    return "-"


def render(args):
    os.system("cls" if os.name == "nt" else "clear")

    print(f"{C['bold']}{C['cyan']}{'=' * 72}")
    print(f"  CAPSTONE 2 - GPU Task Progress Monitor")
    print(f"{'=' * 72}{C['reset']}\n")

    total_done = 0
    total_running = 0
    total_pending = 0
    running_epochs_left = 0

    # Training phases
    for phase_name, runs in PHASES:
        print(f"  {C['cyan']}-- {phase_name} --{C['reset']}")

        for folder, label in runs:
            info = read_training_run(folder)

            if info is None:
                total_pending += 1
                print(f"  {label:<22} {bar(0)} {C['dim']}--  not started{C['reset']}")
                continue

            epoch = info["epoch"]
            pct = epoch / TOTAL_EPOCHS

            if info["done"]:
                total_done += 1
                miou = fmt_pct(info["miou"])
                oa = fmt_pct(info["oa"])
                extra = f"  mIoU {C['bold']}{miou}{C['reset']}"
                if info["oa"] > 0:
                    extra += f"  OA {oa}"
                print(f"  {label:<22} {bar(1.0, done=True)} {C['green']}DONE{C['reset']}{extra}")
            else:
                total_running += 1
                running_epochs_left += (TOTAL_EPOCHS - epoch)
                miou = fmt_pct(info["miou"])
                print(f"  {label:<22} {bar(pct)} {C['yellow']}{epoch:>2}/{TOTAL_EPOCHS}{C['reset']}"
                      f"  best mIoU {C['bold']}{miou}{C['reset']}")

    # File-based tasks
    for phase_name, tasks in FILE_TASKS:
        print(f"\n  {C['cyan']}-- {phase_name} --{C['reset']}")

        for filepath, label in tasks:
            if filepath.exists():
                total_done += 1
                sz = filepath.stat().st_size
                if sz > 1024 * 1024:
                    sz_str = f"{sz / (1024*1024):.1f} MB"
                else:
                    sz_str = f"{sz / 1024:.0f} KB"
                print(f"  {label:<22} {bar(1.0, done=True)} {C['green']}DONE{C['reset']}  {sz_str}")
            else:
                total_pending += 1
                print(f"  {label:<22} {bar(0)} {C['dim']}--  not started{C['reset']}")

    # Summary
    total = total_done + total_running + total_pending
    print(f"\n{C['dim']}{'=' * 72}{C['reset']}")

    parts = []
    if total_done:
        parts.append(f"{C['green']}{total_done} done{C['reset']}")
    if total_running:
        parts.append(f"{C['yellow']}{total_running} running{C['reset']}")
    if total_pending:
        parts.append(f"{C['dim']}{total_pending} pending{C['reset']}")
    print(f"  {C['bold']}{total}{C['reset']} tasks:  {'  '.join(parts)}")

    if running_epochs_left > 0:
        # ~45s per epoch on RTX 4050 for Prithvi
        eta_min = running_epochs_left * 0.75
        if eta_min > 60:
            print(f"  ETA: ~{eta_min/60:.1f} hours ({running_epochs_left} epochs remaining)")
        else:
            print(f"  ETA: ~{eta_min:.0f} min ({running_epochs_left} epochs remaining)")

    print(f"\n  {C['dim']}Refreshes every {args.refresh}s. Press Ctrl+C to stop.{C['reset']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", type=int, default=10)
    args = parser.parse_args()

    try:
        while True:
            render(args)
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print(f"\n{C['dim']}Stopped.{C['reset']}")
