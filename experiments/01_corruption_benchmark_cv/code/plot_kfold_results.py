#!/usr/bin/env python3
"""Plot k-fold CV results: clean accuracy and mCE robustness."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

KFOLD_DIR = Path("outputs_kfold")
OUT_DIR = Path("outputs_kfold/figures")
OUT_DIR.mkdir(exist_ok=True)

# ── display names ─────────────────────────────────────────────────────────────
NAMES = {
    "vgg16_bn":            "VGG-16-BN",
    "googlenet":           "GoogLeNet",
    "efficientnet_b0":     "EfficientNet-B0",
    "regnet_x_400mf":      "RegNet-X-400MF",
    "mnasnet0_5":          "MNASNet-0.5",
    "shufflenet_v2_x0_5":  "ShuffleNet-V2",
    "squeezenet1_0":       "SqueezeNet-1.0",
    "convnext_tiny":       "ConvNeXt-Tiny",
    "swin_t":              "Swin-T",
}

# colour: modern vs classic families
MODERN = {"convnext_tiny", "swin_t", "efficientnet_b0",
          "regnet_x_400mf", "mnasnet0_5", "shufflenet_v2_x0_5"}

acc  = pd.read_csv(KFOLD_DIR / "kfold_clean_accuracy.csv")
mce  = pd.read_csv(KFOLD_DIR / "kfold_mce_summary.csv")

# sort by clean accuracy descending
acc  = acc.sort_values("clean_top1_mean", ascending=False).reset_index(drop=True)
mce  = mce.set_index("model").loc[acc["model"]].reset_index()

labels   = [NAMES[m] for m in acc["model"]]
colours  = ["#2196F3" if m in MODERN else "#FF9800" for m in acc["model"]]

# ── Figure 1: Clean Accuracy ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(labels, acc["clean_top1_mean"], yerr=acc["clean_top1_std"],
              color=colours, capsize=4, edgecolor="white", linewidth=0.6,
              error_kw=dict(elinewidth=1.2, ecolor="#333333"))

ax.set_ylim(94, 99)
ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
ax.set_title("5-Fold Cross-Validation — Clean Test Accuracy\n(mean ± std, 5 folds, uTHCD-C 156-class dataset)",
             fontsize=11, pad=10)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)
plt.xticks(rotation=25, ha="right")

# value labels on bars
for bar, val, std in zip(bars, acc["clean_top1_mean"], acc["clean_top1_std"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.axhline(97.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)

legend_handles = [
    mpatches.Patch(color="#2196F3", label="Modern architectures"),
    mpatches.Patch(color="#FF9800", label="Classic architectures"),
]
ax.legend(handles=legend_handles, fontsize=9, framealpha=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR / "kfold_clean_accuracy.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "kfold_clean_accuracy.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: kfold_clean_accuracy.pdf/.png")

# ── Figure 2: mCE@top-1 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(labels, mce["mce_top1_mean"], yerr=mce["mce_top1_std"],
              color=colours, capsize=4, edgecolor="white", linewidth=0.6,
              error_kw=dict(elinewidth=1.2, ecolor="#333333"))

ax.set_ylabel("mCE@top-1  (relative to VGG-16-BN, lower = more robust)", fontsize=10)
ax.set_title("5-Fold Cross-Validation — Corruption Robustness (mCE@top-1)\n"
             "(mean ± std across 5 folds, 10 corruption types × 5 severities)",
             fontsize=11, pad=10)
ax.tick_params(axis="x", labelsize=9)
ax.tick_params(axis="y", labelsize=9)
plt.xticks(rotation=25, ha="right")

for bar, val, std in zip(bars, mce["mce_top1_mean"], mce["mce_top1_std"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Baseline (VGG-16-BN)")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=legend_handles + [
    plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.8, label="Baseline (VGG-16-BN)")
], fontsize=9, framealpha=0.8)

plt.tight_layout()
fig.savefig(OUT_DIR / "kfold_mce_top1.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "kfold_mce_top1.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: kfold_mce_top1.pdf/.png")

# ── Figure 3: Combined (accuracy left, mCE right) ────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# left: accuracy
bars1 = ax1.bar(labels, acc["clean_top1_mean"], yerr=acc["clean_top1_std"],
                color=colours, capsize=4, edgecolor="white", linewidth=0.6,
                error_kw=dict(elinewidth=1.2, ecolor="#333333"))
ax1.set_ylim(94, 99)
ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
ax1.set_title("(a) Clean Accuracy", fontsize=12, fontweight="bold")
ax1.tick_params(axis="x", labelsize=8)
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
ax1.axhline(97.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)
for bar, val, std in zip(bars1, acc["clean_top1_mean"], acc["clean_top1_std"]):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.04,
             f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

# right: mCE
bars2 = ax2.bar(labels, mce["mce_top1_mean"], yerr=mce["mce_top1_std"],
                color=colours, capsize=4, edgecolor="white", linewidth=0.6,
                error_kw=dict(elinewidth=1.2, ecolor="#333333"))
ax2.set_ylabel("mCE@top-1  (lower = more robust)", fontsize=11)
ax2.set_title("(b) Corruption Robustness (mCE@top-1)", fontsize=12, fontweight="bold")
ax2.tick_params(axis="x", labelsize=8)
plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax2.grid(axis="y", linestyle=":", alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)
for bar, val, std in zip(bars2, mce["mce_top1_mean"], mce["mce_top1_std"]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.05,
             f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10,
           framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("uTHCD-C Benchmark: 5-Fold Cross-Validation Results  (n=90,959 samples, 156 classes)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "kfold_combined.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "kfold_combined.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: kfold_combined.pdf/.png")

# ── Figure 4: Scatter — accuracy vs robustness ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for m, colour in zip(acc["model"], colours):
    row_acc = acc[acc["model"] == m].iloc[0]
    row_mce = mce[mce["model"] == m].iloc[0]
    ax.errorbar(row_acc["clean_top1_mean"], row_mce["mce_top1_mean"],
                xerr=row_acc["clean_top1_std"], yerr=row_mce["mce_top1_std"],
                fmt="o", color=colour, markersize=8, capsize=4,
                elinewidth=1.2, ecolor="#555555")
    ax.annotate(NAMES[m], (row_acc["clean_top1_mean"], row_mce["mce_top1_mean"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.set_xlabel("Clean Top-1 Accuracy (%, 5-fold mean)", fontsize=11)
ax.set_ylabel("mCE@top-1 (5-fold mean, lower = more robust)", fontsize=11)
ax.set_title("Accuracy vs. Corruption Robustness\n(error bars = ±1 std across 5 folds)", fontsize=11)
ax.grid(linestyle=":", alpha=0.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=legend_handles, fontsize=9, framealpha=0.8)
plt.tight_layout()
fig.savefig(OUT_DIR / "kfold_scatter.pdf", dpi=300, bbox_inches="tight")
fig.savefig(OUT_DIR / "kfold_scatter.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: kfold_scatter.pdf/.png")

print("\nAll figures saved to", OUT_DIR)
