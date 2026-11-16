"""Print Tables 3, 4, 6 and every derived number used in the text, from
results/pipeline_fix6_fullcov/fullcov_results.json."""
import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent.parent / "results" / "pipeline_fix6_fullcov" / "fullcov_results.json").read_text())
M = ["unet", "deeplabv3", "segformer", "prithvi"]
N = {"unet": "U-Net", "deeplabv3": "DeepLabV3+", "segformer": "MiT-B2+FPN", "prithvi": "Prithvi"}
C = ["Vegetation", "Built-up", "Cropland", "Water", "Barren/Other"]
g = lambda job, m: R[f"{job}/{m}"]["metrics_table3_mask"]

for job in ("lahore_2021", "gujranwala_2021", "dryseason_2021"):
    for m in M:
        a, b = g(job, m), R[f"{job}/{m}"]["metrics_anyband_mask"]
        assert (a["OA"], a["mIoU"]) == (b["OA"], b["mIoU"]), (job, m, a, b)

print("Table 3 (Lahore OA, mIoU | Gujranwala OA, mIoU)")
for m in M:
    l, gu = g("lahore_2021", m), g("gujranwala_2021", m)
    print(f"  {N[m]:12s} {l['OA']:.2f} {l['mIoU']:.2f} | {gu['OA']:.2f} {gu['mIoU']:.2f}")
print("Table 4 (Gujranwala per-class IoU)")
for m in M:
    print(f"  {N[m]:12s}", " ".join(f"{g('gujranwala_2021', m)['per_class_IoU'][c]:.2f}" for c in C))
print("Table 6 (full-year OA, mIoU | dry OA, mIoU)")
for m in M:
    f, d = g("gujranwala_2021", m), g("dryseason_2021", m)
    print(f"  {N[m]:12s} {f['OA']:.2f} {f['mIoU']:.2f} | {d['OA']:.2f} {d['mIoU']:.2f}")

gm = {m: g("gujranwala_2021", m)["mIoU"] for m in M}
lm = {m: g("lahore_2021", m)["mIoU"] for m in M}
base = [m for m in M if m != "prithvi"]
margins = sorted(round(gm["prithvi"] - gm[m], 2) for m in base)
print("\nZero-shot margin over baselines (pp):", margins, "best baseline:", N[max(base, key=gm.get)], gm[max(base, key=gm.get)])
drops = {m: round(lm[m] - gm[m], 2) for m in M}
print("Lahore -> Gujranwala mIoU drop:", drops, "baseline range", min(drops[m] for m in base), "-", max(drops[m] for m in base))
pc = {m: g("gujranwala_2021", m)["per_class_IoU"] for m in M}
for c in ("Barren/Other", "Vegetation"):
    v = [pc[m][c] for m in M]
    print(f"{c} IoU range: {min(v):.2f}-{max(v):.2f}")
print(f"Prithvi Veg IoU {pc['prithvi']['Vegetation']:.2f}, over best baseline {pc['prithvi']['Vegetation'] - max(pc[m]['Vegetation'] for m in base):+.2f} ({N[max(base, key=lambda m: pc[m]['Vegetation'])]})")
print(f"Prithvi Built IoU {pc['prithvi']['Built-up']:.2f}, over baselines {pc['prithvi']['Built-up'] - max(pc[m]['Built-up'] for m in base):+.2f} to {pc['prithvi']['Built-up'] - min(pc[m]['Built-up'] for m in base):+.2f}")
best_class = {c: N[max(M, key=lambda m: pc[m][c])] for c in C}
print("Best model per class:", best_class)
sea = {m: round(g("dryseason_2021", m)["mIoU"] - gm[m], 2) for m in M}
print("Dry - full mIoU (pp):", sea, "max |diff|", max(abs(x) for x in sea.values()))
print("Dry-season best:", N[max(M, key=lambda m: g('dryseason_2021', m)['mIoU'])], g("dryseason_2021", "prithvi")["mIoU"])
print("Full-city Lahore mIoU range:", min(lm.values()), "-", max(lm.values()))
