"""Check every table in a manuscript (.tex) against the raw result files.

Tables 1-2: results/{model}_seed{42,123,7}/test_results.json (mean, sample std)
Tables 3, 4, 6: results/pipeline_fix6_fullcov/fullcov_results.json
Table 5: results/pipeline_fix6_fullcov/change_stats.json (exact pixel counts)
Table 7: results/prithvi_v2_fix6_*_seed42/test_results.json + results/prithvi_param_counts.json
Usage: python -I evaluation/verify_manuscript_numbers.py path/to/main.tex
"""
import re
import sys
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
tex = Path(sys.argv[1]).read_text(encoding="utf-8")
lines = tex.splitlines()
ok = bad = 0


def check(label, got, want):
    global ok, bad
    if f"{got}" == f"{want}":
        ok += 1
    else:
        bad += 1
        print(f"MISMATCH {label}: paper {got} vs source {want}")


def row(prefix, need):
    c = [l for l in lines if l.strip().startswith(prefix) and need(l)]
    if len(c) != 1:
        raise SystemExit(f"row '{prefix}' found {len(c)} times")
    return [x for x in re.findall(r"[0-9]+\.[0-9]+", c[0].split("&", 1)[1])]


pct = lambda x: x * 100 if x <= 1.0 else x
f2 = lambda x: f"{x:.2f}"
f1 = lambda x: f"{x:.1f}"
# Tables 1 and 2
T1 = {"unet": "U-Net (EffNet-B4)", "deeplabv3": "DeepLabV3+ (ResNet-101)", "segformer": "MiT-B2+FPN               &", "prithvi_v2_fix6": "Prithvi-EO 2.0 (LoRA)"}
T2 = {"unet": "U-Net        &", "deeplabv3": "DeepLabV3+   &", "segformer": "MiT-B2+FPN   &", "prithvi_v2_fix6": "Prithvi 2.0   &"}
for k in T1:
    ds = [json.loads((RES / f"{k}_seed{s}" / "test_results.json").read_text()) for s in (42, 123, 7)]
    want = []
    for key in ("oa", "miou", "mf1"):
        v = [pct(d[key]) for d in ds]
        want += [f2(st.mean(v)), f2(st.stdev(v))]
    check(f"T1 {k}", row(T1[k].rstrip("&").strip(), lambda l: l.count("pm") == 3), want)
    want = []
    for i in range(5):
        v = [pct(d["iou_per_class"][i]) for d in ds]
        want += [f2(st.mean(v)), f2(st.stdev(v))]
    check(f"T2 {k}", row(T2[k].rstrip("&").strip(), lambda l: l.count("pm") == 5), want)

# Tables 3, 4, 6
F = json.loads((RES / "pipeline_fix6_fullcov" / "fullcov_results.json").read_text())
g = lambda job, m: F[f"{job}/{m}"]["metrics_table3_mask"]
P = {"unet": "U-Net        &", "deeplabv3": "DeepLabV3+   &", "segformer": "MiT-B2+FPN   &", "prithvi": "Prithvi 2.0   &"}
C = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]
for m, pre in P.items():
    pre = pre.rstrip("&").strip()
    l, gu, d = g("lahore_2021", m), g("gujranwala_2021", m), g("dryseason_2021", m)
    t3 = [f2(l["OA"]), f2(l["mIoU"]), f2(gu["OA"]), f2(gu["mIoU"])]
    t6 = [f2(gu["OA"]), f2(gu["mIoU"]), f2(d["OA"]), f2(d["mIoU"])]
    four = [r for r in lines if r.strip().startswith(pre) and "pm" not in r and r.count("&") == 4]
    got = [re.findall(r"[0-9]+\.[0-9]+", r.split("&", 1)[1]) for r in four]
    check(f"T3 {m}", got[0], t3)
    check(f"T6 {m}", got[1], t6)
    check(f"T4 {m}", row(pre, lambda r: "pm" not in r and r.count("&") == 5), [f2(gu["per_class_IoU"][c]) for c in C])

# Table 5 (exact pixel counts from the saved maps, 10 m pixels)
import numpy as np, rasterio
from decimal import Decimal, ROUND_HALF_UP
r1 = lambda px: str((Decimal(px) / Decimal(10000)).quantize(Decimal("0.1"), ROUND_HALF_UP))
CNT = {}
for city in ("lahore", "gujranwala"):
    a = rasterio.open(RES / "pipeline_fix6_fullcov" / city / f"lulc_{city}_prithvi_2018.tif").read(1)
    b = rasterio.open(RES / "pipeline_fix6_fullcov" / city / f"lulc_{city}_prithvi_2026.tif").read(1)
    v = (a < 5) & (b < 5)
    CNT[city] = [(int(((a == k) & v).sum()), int(((b == k) & v).sum())) for k in range(5)]
T5 = ["Vegetation   &", "Built-up     &", "Cropland     &", "Water        &", "Barren/Other &"]
for k, pre in enumerate(T5):
    want = []
    for city in ("lahore", "gujranwala"):
        x, y = CNT[city][k]
        want += [r1(x), r1(y), r1(abs(y - x))]
    check(f"T5 {pre}", row(pre.rstrip("&").strip(), lambda l: l.count("&") == 6), want)

# Table 7
PC = {d["config"]: d for d in json.loads((RES / "prithvi_param_counts.json").read_text())}
T7 = [("\\quad $r{=}8$", "prithvi_v2_fix6_r8_seed42"), ("\\quad $r{=}16$", "prithvi_v2_fix6_r16_seed42"),
      ("\\quad $r{=}32$", "prithvi_v2_fix6_r32_seed42"), ("\\quad $r{=}64$", "prithvi_v2_fix6_seed42"),
      ("\\quad LoRA + unfreezing", "prithvi_v2_fix6_seed42"), ("\\quad LoRA only", "prithvi_v2_fix6_lora_only_seed42"),
      ("\\quad Unfreezing only", "prithvi_v2_fix6_unfreeze_only_seed42")]
for pre, cfg in T7:
    d = json.loads((RES / cfg / "test_results.json").read_text())
    pc = next(v for k, v in PC.items() if k == cfg or (cfg == "prithvi_v2_fix6_seed42" and v["lora_rank"] == 64 and v["unfreeze_blocks"] == 8))
    want = [f1(pc["trainable"] / 1e6), f1(pc["pct"]), f2(pct(d["miou"])), f2(pct(d["oa"]))]
    c = [l for l in lines if l.strip().startswith(pre)]
    check(f"T7 {cfg}", re.findall(r"[0-9]+\.[0-9]+", c[0].split("&", 1)[1]), want)

print(f"{sys.argv[1]}: {ok} rows match, {bad} mismatches")
sys.exit(1 if bad else 0)
