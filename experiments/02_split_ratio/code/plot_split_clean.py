#!/usr/bin/env python3
"""Clean Top-1 accuracy vs train:test split ratio, one line per architecture.
Reads outputs_split/ratio_<RR>/<model>/corruption_results_A.json -> clean.top1.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        d = json.load(open(f"outputs_split/ratio_{rr}/{m}/corruption_results_A.json"))
        clean[m].append(d["clean"]["top1"])

# print table
print(f"{'Model':18s} " + " ".join(f"{x}".rjust(7) for x in XLAB) + "   spread")
for m in sorted(DISP, key=lambda x: -sum(clean[x]) / len(clean[x])):
    vals = clean[m]
    print(f"{DISP[m]:18s} " + " ".join(f"{v:7.2f}" for v in vals)
          + f"   {max(vals)-min(vals):.2f}")

fig, ax = plt.subplots(figsize=(7.2, 5.0))
x = list(range(len(RATIOS)))
order = sorted(DISP, key=lambda m: -sum(clean[m]) / len(clean[m]))
for m in order:
    c, mk, lw = STYLE[m]
    ls = "--" if m == "vgg16_bn" else "-"
    ax.plot(x, clean[m], ls=ls, color=c, marker=mk, lw=lw, ms=6, label=DISP[m],
            zorder=3 if m == "vgg16_bn" else 2)
ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_ylim(93, 99)
ax.set_xlabel("Train : test split ratio", fontsize=11)
ax.set_ylabel("Clean Top-1 accuracy (\\%)", fontsize=11)
ax.set_title("Clean accuracy is uniformly high and nearly flat across split ratios\n"
             "(every model $\\geq 94\\%$ even at 50:50, despite divergent robustness)", fontsize=11)
ax.grid(True, axis="y", ls=":", alpha=0.4)
ax.legend(fontsize=8.5, ncol=2, loc="lower right", framealpha=0.92)
fig.tight_layout()
import os
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/split_ratio_clean.png", dpi=200, bbox_inches="tight")
print("\nsaved results/figures/split_ratio_clean.png")
