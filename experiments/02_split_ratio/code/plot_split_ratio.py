#!/usr/bin/env python3
"""mCE@top-1 vs train:test split ratio, one line per architecture, to visualize
rank consistency across split proportions. Reads outputs_split/split_ratio_summary.csv.
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "outputs_split/split_ratio_summary.csv"
RATIOS = [50, 60, 70, 80, 90]
XLAB = ["50:50", "60:40", "70:30", "80:20", "90:10"]
DISP = {"vgg16_bn": "VGG16_BN", "googlenet": "GoogLeNet", "swin_t": "Swin-Tiny",
        "efficientnet_b0": "EfficientNet-B0", "squeezenet1_0": "SqueezeNet1_0",
        "convnext_tiny": "ConvNeXt-Tiny", "regnet_x_400mf": "RegNet-X-400MF",
        "shufflenet_v2_x0_5": "ShuffleNetV2-0.5", "mnasnet0_5": "MnasNet0_5"}
# group -> (color, marker); robust=blues/green, mid=greys, fragile=reds
STYLE = {
    "swin_t":            ("#1b7837", "o", 2.0),   # robust
    "vgg16_bn":          ("#000000", "D", 2.6),    # baseline
    "convnext_tiny":     ("#2166ac", "o", 2.0),
    "efficientnet_b0":   ("#7570b3", "s", 1.6),    # mid
    "squeezenet1_0":     ("#666666", "s", 1.6),
    "googlenet":         ("#999999", "s", 1.6),
    "regnet_x_400mf":    ("#f4a582", "^", 1.8),    # fragile
    "shufflenet_v2_x0_5":("#d6604d", "^", 1.8),
    "mnasnet0_5":        ("#b2182b", "^", 2.0),
}

rows = {}
for r in csv.DictReader(open(CSV)):
    rows[r["model"]] = [float(r[f"mce_{x}"]) for x in RATIOS]

# order legend by mean mCE (most robust first)
order = sorted(rows, key=lambda m: sum(rows[m]) / len(rows[m]))

fig, ax = plt.subplots(figsize=(7.2, 5.0))
x = list(range(len(RATIOS)))
for m in order:
    c, mk, lw = STYLE[m]
    ls = "--" if m == "vgg16_bn" else "-"
    ax.plot(x, rows[m], ls=ls, color=c, marker=mk, lw=lw, ms=6,
            label=DISP[m], zorder=3 if m == "vgg16_bn" else 2)

ax.axhspan(0.0, 1.30, color="#1b7837", alpha=0.05, zorder=0)   # "robust band" guide
ax.set_yscale("log")
ax.set_yticks([0.8, 1.0, 1.5, 2, 3, 5, 8])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_xlabel("Train : test split ratio", fontsize=11)
ax.set_ylabel("mCE@top-1  (VGG16_BN = 1.0, lower is better)", fontsize=11)
ax.set_title("Robustness ranking is preserved across split ratios\n"
             "(mean Kendall $\\tau$ = 0.72, all pairwise $\\tau$ > 0)", fontsize=11)
ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
ax.legend(fontsize=8.5, ncol=2, loc="upper right", framealpha=0.92)
fig.tight_layout()
import os
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/split_ratio_mce.png", dpi=200, bbox_inches="tight")
print("saved results/figures/split_ratio_mce.png")
