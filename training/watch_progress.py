"""Live training progress bar for Capstone 2 experiments."""
import os, sys, csv, time, json
from pathlib import Path

RESULTS = Path(r"D:\MY\Study\Research\Paper 2\results")
TOTAL_EPOCHS = 50

COLORS = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "green": "\033[92m", "cyan": "\033[96m", "yellow": "\033[93m",
    "red": "\033[91m", "magenta": "\033[95m", "white": "\033[97m",
    "bg_green": "\033[42m", "bg_gray": "\033[100m",
}

RUNS = [
    ("unet_seed42", "U-Net s42"),
    ("unet_seed123", "U-Net s123"),
    ("unet_seed7", "U-Net s7"),
    ("segformer_seed42", "MiT-B2 s42"),
    ("segformer_seed123", "MiT-B2 s123"),
    ("segformer_seed7", "MiT-B2 s7"),
    ("deeplabv3_seed42", "DLv3+ s42"),
    ("deeplabv3_seed123", "DLv3+ s123"),
    ("deeplabv3_seed7", "DLv3+ s7"),
    ("prithvi_v2_seed42", "Prithvi s42"),
    ("prithvi_v2_seed123", "Prithvi s123"),
    ("prithvi_v2_seed7", "Prithvi s7"),
    ("prithvi_lora_r8_seed42", "LoRA r=8"),
    ("prithvi_lora_r16_seed42", "LoRA r=16"),
    ("prithvi_lora_r32_seed42", "LoRA r=32"),
]


def read_run(folder):
    d = RESULTS / folder
    test_file = d / "test_results.json"
    csv_file = d / "metrics.csv"

    if test_file.exists():
        with open(test_file) as f:
            r = json.load(f)
        return {"epoch": TOTAL_EPOCHS, "done": True,
                "miou": r.get("miou", 0), "oa": r.get("oa", 0), "mf1": r.get("mf1", 0)}

    if csv_file.exists():
        rows = []
        with open(csv_file) as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if rows:
            last = rows[-1]
            best_miou = max(float(r.get("val_miou", 0)) for r in rows)
            return {"epoch": int(last["epoch"]), "done": False,
                    "miou": best_miou, "oa": 0, "mf1": 0}

    if not d.exists():
        return None
    return {"epoch": 0, "done": False, "miou": 0, "oa": 0, "mf1": 0}


def bar(pct, width=30):
    filled = int(width * pct)
    return f"{COLORS['bg_green']}{' ' * filled}{COLORS['reset']}{COLORS['bg_gray']}{' ' * (width - filled)}{COLORS['reset']}"


def render():
    c = COLORS
    os.system("cls" if os.name == "nt" else "clear")

    print(f"{c['bold']}{c['cyan']}{'=' * 68}")
    print(f"   CAPSTONE 2 — Training Progress Monitor")
    print(f"{'=' * 68}{c['reset']}\n")

    done_count = 0
    total_count = 0

    for folder, label in RUNS:
        info = read_run(folder)
        if info is None:
            status = f"  {c['dim']}{'—':>12}  not started{c['reset']}"
            print(f"  {label:<14} {status}")
            continue

        total_count += 1
        epoch = info["epoch"]
        pct = epoch / TOTAL_EPOCHS

        if info["done"]:
            done_count += 1
            miou_str = f"{info['miou']*100:.2f}%"
            oa_str = f"{info['oa']*100:.2f}%"
            print(f"  {label:<14} {bar(1.0)} {c['green']}DONE{c['reset']}  "
                  f"mIoU {c['bold']}{miou_str}{c['reset']}  OA {oa_str}")
        else:
            miou_str = f"{info['miou']*100:.2f}%" if info["miou"] > 0 else "—"
            print(f"  {label:<14} {bar(pct)} {c['yellow']}{epoch:>2}/{TOTAL_EPOCHS}{c['reset']}  "
                  f"best mIoU {c['bold']}{miou_str}{c['reset']}")

    # Summary
    print(f"\n{c['dim']}{'─' * 68}{c['reset']}")
    running = total_count - done_count
    print(f"  {c['green']}{done_count} done{c['reset']}  "
          f"{c['yellow']}{running} running{c['reset']}  "
          f"{c['dim']}{len(RUNS) - total_count} pending{c['reset']}")
    print(f"\n  {c['dim']}Refreshes every 10s. Press Ctrl+C to stop.{c['reset']}\n")


if __name__ == "__main__":
    try:
        while True:
            render()
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"\n{COLORS['dim']}Stopped.{COLORS['reset']}")
