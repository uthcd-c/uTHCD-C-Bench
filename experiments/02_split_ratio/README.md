# 02 — Split-ratio sweep

Tests whether the robustness ranking depends on the train:test **proportion**.
Retrains all nine architectures from scratch at five ratios (50:50, 60:40, 70:30,
80:20, 90:10) under a single stratified split per ratio, runs the corruption
suite, and compares the per-ratio mCE@top-1 rankings. *Paper Section 4.3.*

Run from the repository root with `export PYTHONPATH=src`.

## Reproduce

```bash
# 1. Train every model at every ratio
for r in 0.5 0.6 0.7 0.8 0.9; do for m in vgg16_bn googlenet swin_t efficientnet_b0 \
    squeezenet1_0 convnext_tiny regnet_x_400mf shufflenet_v2_x0_5 mnasnet0_5; do
  python3 experiments/02_split_ratio/code/train_split.py --model $m --ratio $r --data_dir dataset
done; done
# (code/run_split_queue.sh provides a resumable multi-GPU job queue)

# 2. Corruption evaluation per ratio
for r in 0.5 0.6 0.7 0.8 0.9; do
  python3 experiments/02_split_ratio/code/eval_split.py --ratio $r
done

# 3. Aggregate + figure
python3 experiments/02_split_ratio/code/aggregate_splits.py
python3 experiments/02_split_ratio/code/plot_split_combined.py   # -> results/figures/split_ratio_combined.png
```

## Shipped results

* `results/tables/split_ratio_summary.csv` — model × ratio mCE@top-1 + ranking stats
* `results/figures/split_ratio_combined.png` — clean accuracy (flat) vs mCE@top-1 (fans out)

## Key finding

The ranking is preserved across all five ratios: every pairwise Kendall's τ is
positive (mean 0.72, min 0.61; Spearman ρ mean 0.83). The three least-robust
models (RegNet, ShuffleNetV2, MnasNet) stay at the bottom at every ratio; clean
accuracy stays > 94% everywhere and is uninformative. Checkpoints (~2.4 GB) not
bundled — see `checkpoints/MANIFEST.md`.
