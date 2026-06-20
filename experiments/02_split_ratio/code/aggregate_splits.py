#!/usr/bin/env python3
"""
Aggregate the split-ratio robustness experiment. Reads
outputs_split/ratio_<RR>/<model>/corruption_results_A.json for every available
ratio, builds the mCE@top-1 (and clean test-acc) table model x ratio, and tests
whether the robustness RANKING is consistent across split ratios via pairwise
Kendall's tau and Spearman rho, plus the mean-rank ordering.

Writes outputs_split/split_ratio_summary.csv and prints a report.
"""
import sys, json, glob, itertools
from pathlib import Path
import numpy as np

try:
    from scipy.stats import kendalltau, spearmanr
except Exception:
    kendalltau = spearmanr = None

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs_split")
MODELS = ["vgg16_bn", "googlenet", "swin_t", "efficientnet_b0", "squeezenet1_0",
          "convnext_tiny", "regnet_x_400mf", "shufflenet_v2_x0_5", "mnasnet0_5"]
DISP = {"vgg16_bn": "VGG16_BN", "googlenet": "GoogLeNet", "swin_t": "Swin-Tiny",
        "efficientnet_b0": "EfficientNet-B0", "squeezenet1_0": "SqueezeNet1_0",
        "convnext_tiny": "ConvNeXt-Tiny", "regnet_x_400mf": "RegNet-X-400MF",
        "shufflenet_v2_x0_5": "ShuffleNetV2-0.5", "mnasnet0_5": "MnasNet0_5"}

ratios = sorted(int(p.name.split("_")[1]) for p in OUT.glob("ratio_*") if p.is_dir())
mce, clean = {}, {}
complete = []
for rr in ratios:
    rdir = OUT / f"ratio_{rr}"
    vals, cl = {}, {}
    ok = True
    for m in MODELS:
        f = rdir / m / "corruption_results_A.json"
        if not f.exists():
            ok = False; continue
        d = json.load(open(f))
        vals[m] = d["mce"]["top1"]
        cl[m] = d["clean"]["top1"]
    mce[rr] = vals; clean[rr] = cl
    status = "complete" if ok else f"partial ({len(vals)}/9)"
    print(f"ratio {rr}:{100-rr:>3}  -> {status}")
    if ok:
        complete.append(rr)

if len(complete) < 2:
    print("\nNeed >=2 complete ratios to test rank consistency. "
          f"Complete so far: {complete}")
    sys.exit(0)

# ---- mCE@top-1 table (rows=model sorted by mean mCE across complete ratios) ----
def meanmce(m): return np.mean([mce[rr][m] for rr in complete])
rows = sorted(MODELS, key=meanmce)
hdr = "Model".ljust(18) + "".join(f"{rr}:{100-rr}".rjust(9) for rr in complete) + "    mean   rank-std"
print("\n=== mCE@top-1 by train:test ratio (lower=better) ===")
print(hdr); print("-" * len(hdr))
# rank within each ratio (1=most robust)
rankvec = {rr: {m: r + 1 for r, m in enumerate(sorted(MODELS, key=lambda x: mce[rr][x]))}
           for rr in complete}
csv = ["model," + ",".join(f"mce_{rr}" for rr in complete) + ",mean_mce,"
       + ",".join(f"rank_{rr}" for rr in complete) + ",rank_std"]
for m in rows:
    rline = [rankvec[rr][m] for rr in complete]
    rstd = np.std(rline)
    print(DISP[m].ljust(18) + "".join(f"{mce[rr][m]:9.2f}" for rr in complete)
          + f"  {meanmce(m):6.2f}   {rstd:6.2f}")
    csv.append(f"{m}," + ",".join(f"{mce[rr][m]:.4f}" for rr in complete)
               + f",{meanmce(m):.4f}," + ",".join(str(rankvec[rr][m]) for rr in complete)
               + f",{rstd:.3f}")

# ---- pairwise rank-correlation across ratios ----
print("\n=== pairwise ranking agreement across split ratios ===")
if kendalltau is None:
    print("scipy unavailable; skipping correlation")
else:
    taus, rhos = [], []
    for r1, r2 in itertools.combinations(complete, 2):
        v1 = [mce[r1][m] for m in MODELS]
        v2 = [mce[r2][m] for m in MODELS]
        t = kendalltau(v1, v2).correlation
        s = spearmanr(v1, v2).correlation
        taus.append(t); rhos.append(s)
        print(f"  {r1}:{100-r1} vs {r2}:{100-r2}   Kendall tau={t:+.3f}   Spearman rho={s:+.3f}")
    print(f"\n  mean Kendall tau = {np.mean(taus):+.3f} (min {np.min(taus):+.3f})")
    print(f"  mean Spearman rho = {np.mean(rhos):+.3f} (min {np.min(rhos):+.3f})")

# ---- consensus ordering ----
print("\n=== consensus robustness order (by mean mCE@top-1) ===")
print("  " + "  <  ".join(f"{DISP[m]}({meanmce(m):.2f})" for m in rows))

(OUT / "split_ratio_summary.csv").write_text("\n".join(csv) + "\n")
print(f"\nsaved {OUT/'split_ratio_summary.csv'}")
