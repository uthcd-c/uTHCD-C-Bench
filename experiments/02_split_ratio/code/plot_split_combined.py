#!/usr/bin/env python3
"""Two-panel split-ratio figure: clean Top-1 accuracy (linear, top) and
mCE@top-1 (log, bottom), sharing the x-axis (train:test ratio). Highlights that
clean accuracy is flat and uninformative while robustness ranking is preserved.
"""
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

RATIOS = [50, 60, 70, 80, 90]
XLAB = ["50:50", "60:40", "70:30", "80:20", "90:10"]
DISP = {"vgg16_bn": "VGG16_BN", "googlenet": "GoogLeNet", "swin_t": "Swin-Tiny",
        "efficientnet_b0": "EfficientNet-B0", "squeezenet1_0": "SqueezeNet1_0",
        "convnext_tiny": "ConvNeXt-Tiny", "regnet_x_400mf": "RegNet-X-400MF",
        "shufflenet_v2_x0_5": "ShuffleNetV2-0.5", "mnasnet0_5": "MnasNet0_5"}
STYLE = {
    "swin_t": ("#1b7837", "o", 2.0), "vgg16_bn": ("#000000", "D", 2.6),
    "convnext_tiny": ("#2166ac", "o", 2.0), "efficientnet_b0": ("#7570b3", "s", 1.6),
    "squeezenet1_0": ("#666666", "s", 1.6), "googlenet": ("#999999", "s", 1.6),
    "regnet_x_400mf": ("#f4a582", "^", 1.8), "shufflenet_v2_x0_5": ("#d6604d", "^", 1.8),
    "mnasnet0_5": ("#b2182b", "^", 2.0),
}

clean = {m: [] for m in DISP}
for rr in RATIOS:
    for m in DISP:
        clean[m].append(json.load(open(f"outputs_split/ratio_{rr}/{m}/corruption_results_A.json"))["clean"]["top1"])
mce = {}
for r in csv.DictReader(open("outputs_split/split_ratio_summary.csv")):
    mce[r["model"]] = [float(r[f"mce_{x}"]) for x in RATIOS]

order = sorted(DISP, key=lambda m: sum(mce[m]) / len(mce[m]))   # robust first
x = list(range(len(RATIOS)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 7.4), sharex=True,
                               gridspec_kw={"hspace": 0.13})

for m in order:
    c, mk, lw = STYLE[m]
    ls = "--" if m == "vgg16_bn" else "-"
    z = 3 if m == "vgg16_bn" else 2
    ax1.plot(x, clean[m], ls=ls, color=c, marker=mk, lw=lw, ms=6, label=DISP[m], zorder=z)
    ax2.plot(x, mce[m], ls=ls, color=c, marker=mk, lw=lw, ms=6, label=DISP[m], zorder=z)

ax1.set_ylim(93, 99)
ax1.set_ylabel("Clean Top-1 accuracy (\\%)", fontsize=11)
ax1.set_title("(a) Clean accuracy: high and nearly flat at every ratio", loc="left", fontsize=10.5)
ax1.grid(True, axis="y", ls=":", alpha=0.4)

ax2.set_yscale("log")
ax2.set_yticks([0.8, 1.0, 1.5, 2, 3, 5, 8])
ax2.get_yaxis().set_major_formatter(mt.ScalarFormatter())
ax2.axhspan(0.0, 1.30, color="#1b7837", alpha=0.05, zorder=0)
ax2.set_ylabel("mCE@top-1  (VGG16\\_BN $=1.0$)", fontsize=11)
ax2.set_title("(b) Corruption robustness: ranking preserved (mean Kendall $\\tau$ = 0.72)", loc="left", fontsize=10.5)
ax2.grid(True, which="both", axis="y", ls=":", alpha=0.4)
ax2.set_xticks(x); ax2.set_xticklabels(XLAB)
ax2.set_xlabel("Train : test split ratio", fontsize=11)

h, l = ax2.get_legend_handles_labels()
fig.legend(h, l, loc="center left", bbox_to_anchor=(0.805, 0.5), fontsize=8.7, framealpha=0.95)
fig.subplots_adjust(left=0.085, right=0.80, top=0.95, bottom=0.07)
import os
os.makedirs("results/figures", exist_ok=True)
out = "results/figures/split_ratio_combined.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
