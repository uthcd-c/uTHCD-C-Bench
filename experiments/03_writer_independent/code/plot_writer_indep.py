#!/usr/bin/env python3
"""Two-panel writer-independent figure mirroring plot_split_combined.py: clean
Top-1 (linear, top) and mCE@top-1 (log, bottom) as PAIRED bars per model --
writer-DEPENDENT 70:30 (ratio_70) vs writer-INDEPENDENT 70:30 (mean over 3
seeds, error bars = std). Message: clean accuracy is essentially unchanged
(negligible writer gap) while the corruption-robustness ordering is preserved,
with ShuffleNetV2 the one architecture whose apparent mild-noise robustness was
partly writer-dependent.
"""
import json, statistics as st
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

SEEDS = [1, 42, 123]
DISP = {"vgg16_bn": "VGG16_BN", "convnext_tiny": "ConvNeXt-Tiny", "swin_t": "Swin-Tiny",
        "googlenet": "GoogLeNet", "efficientnet_b0": "EfficientNet-B0",
        "squeezenet1_0": "SqueezeNet1_0", "regnet_x_400mf": "RegNet-X-400MF",
        "shufflenet_v2_x0_5": "ShuffleNetV2-0.5", "mnasnet0_5": "MnasNet0_5"}


def wd(m, key):
    d = json.load(open(f"outputs_split/ratio_70/{m}/corruption_results_A.json"))
    return d["clean"]["top1"] if key == "clean" else d["mce"]["top1"]


def wi(m, key):
    vals = []
    for s in SEEDS:
        d = json.load(open(f"outputs_writer/seed_{s}/{m}/corruption_results_A.json"))
        vals.append(d["clean"]["top1"] if key == "clean" else d["mce"]["top1"])
    return st.mean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)

# order robust -> fragile by writer-independent mCE
order = sorted(DISP, key=lambda m: wi(m, "mce")[0])
labels = [DISP[m] for m in order]
x = np.arange(len(order)); w = 0.38
CWD, CWI = "#9ecae1", "#08519c"   # writer-dependent (light) vs independent (dark)

clean_wd = [wd(m, "clean") for m in order]
clean_wi = [wi(m, "clean") for m in order]
mce_wd = [wd(m, "mce") for m in order]
mce_wi = [wi(m, "mce") for m in order]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.6), sharex=True,
                               gridspec_kw={"hspace": 0.16})

ax1.bar(x - w/2, clean_wd, w, color=CWD, label="Writer-dependent 70:30")
ax1.bar(x + w/2, [c[0] for c in clean_wi], w, yerr=[c[1] for c in clean_wi],
        color=CWI, capsize=3, label="Writer-independent 70:30 (mean$\\pm$std, 3 seeds)")
ax1.set_ylim(92, 99)
ax1.set_ylabel("Clean Top-1 accuracy (\\%)", fontsize=11)
ax1.set_title("(a) Clean accuracy: nearly identical on unseen writers (mean gap $0.12\\%$)",
              loc="left", fontsize=10.5)
ax1.grid(True, axis="y", ls=":", alpha=0.4)
ax1.legend(fontsize=9, framealpha=0.95, loc="lower left")

ax2.bar(x - w/2, mce_wd, w, color=CWD)
ax2.bar(x + w/2, [m[0] for m in mce_wi], w, yerr=[m[1] for m in mce_wi],
        color=CWI, capsize=3)
ax2.set_yscale("log")
ax2.set_yticks([0.8, 1.0, 1.5, 2, 3, 5, 8])
ax2.get_yaxis().set_major_formatter(mt.ScalarFormatter())
ax2.axhspan(0.0, 1.30, color="#1b7837", alpha=0.05, zorder=0)
ax2.set_ylabel("mCE@top-1  (VGG16\\_BN $=1.0$)", fontsize=11)
ax2.set_title("(b) Corruption robustness: ranking preserved (Kendall $\\tau=0.67$); "
              "ShuffleNetV2 worsens most", loc="left", fontsize=10.5)
ax2.grid(True, which="both", axis="y", ls=":", alpha=0.4)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9.5)

fig.subplots_adjust(left=0.085, right=0.97, top=0.95, bottom=0.13)
import os
os.makedirs("results/figures", exist_ok=True)
out = "results/figures/writer_indep_combined.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
print("order (robust->fragile):", order)
