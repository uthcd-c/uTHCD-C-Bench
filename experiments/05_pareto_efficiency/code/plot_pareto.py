#!/usr/bin/env python3
"""Robustness/accuracy vs. efficiency Pareto frontiers (paper Sec. 4.2 / 4.5).

Reads the shipped cross-validation summaries and the measured throughput, and
draws two scatter plots with the Pareto-optimal frontier highlighted:
  * pareto_mce.png       : mCE@top-1 (robustness, lower better) vs throughput
  * pareto_accuracy.png  : clean Top-1 (higher better)          vs throughput

Inputs (results/tables/):
  throughput_A100.json        {model: images_per_second}
  kfold_mce_summary_A.csv     model, mce_top1_mean, mce_top1_std
  kfold_clean_accuracy.csv    model, clean_top1_mean, clean_top1_std, n_folds

Throughput is the efficiency axis used here because it is the measured quantity
shipped with the repo; parameter count can be substituted on the x-axis if
preferred. Run from the repository root.
"""
import json, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISP = {"vgg16_bn": "VGG16_BN", "googlenet": "GoogLeNet", "swin_t": "Swin-Tiny",
        "efficientnet_b0": "EfficientNet-B0", "squeezenet1_0": "SqueezeNet1_0",
        "convnext_tiny": "ConvNeXt-Tiny", "regnet_x_400mf": "RegNet-X-400MF",
        "shufflenet_v2_x0_5": "ShuffleNetV2-0.5", "mnasnet0_5": "MnasNet0_5"}

thr = json.load(open("results/tables/throughput_A100.json"))
mce = {r["model"]: float(r["mce_top1_mean"]) for r in csv.DictReader(open("results/tables/kfold_mce_summary_A.csv"))}
clean = {r["model"]: float(r["clean_top1_mean"]) for r in csv.DictReader(open("results/tables/kfold_clean_accuracy.csv"))}


def pareto(points, lower_is_better):
    """Return the set of models on the efficiency/quality Pareto frontier
    (higher throughput is always better; quality direction set by flag)."""
    front = []
    for m, (x, y) in points.items():
        dom = False
        for m2, (x2, y2) in points.items():
            if m2 == m:
                continue
            better_q = (y2 <= y) if lower_is_better else (y2 >= y)
            if x2 >= x and better_q and (x2 > x or (y2 != y)):
                dom = True
                break
        if not dom:
            front.append(m)
    return front


def draw(quality, lower_is_better, ylabel, title, fname):
    pts = {m: (thr[m], quality[m]) for m in DISP if m in quality and m in thr}
    front = sorted(pareto(pts, lower_is_better), key=lambda m: thr[m])
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for m, (x, y) in pts.items():
        on = m in front
        ax.scatter(x, y, s=90, zorder=3,
                   color="#d6604d" if on else "#4393c3",
                   edgecolor="black" if on else "none", linewidth=0.8)
        ax.annotate(DISP[m], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8.5)
    fx = [thr[m] for m in front]; fy = [quality[m] for m in front]
    ax.plot(fx, fy, "--", color="#d6604d", lw=1.6, zorder=2, label="Pareto frontier")
    ax.set_xscale("log")
    ax.set_xlabel("Throughput (images/s, higher = more efficient)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    import os
    os.makedirs("results/figures", exist_ok=True)
    fig.savefig(f"results/figures/{fname}", dpi=200, bbox_inches="tight")
    print("saved results/figures/" + fname, "| frontier:", [DISP[m] for m in front])


draw(mce, True, "mCE@top-1 (VGG16_BN $=1.0$, lower better)",
     "Robustness–efficiency Pareto frontier", "pareto_mce.png")
draw(clean, False, "Clean Top-1 accuracy (%)",
     "Accuracy–efficiency Pareto frontier", "pareto_accuracy.png")
