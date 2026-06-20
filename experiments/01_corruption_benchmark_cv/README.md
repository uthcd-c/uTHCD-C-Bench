# 01 — Corruption benchmark under 5-fold cross-validation

The core benchmark. Trains each of the nine architectures from scratch on every
fold of a 5-fold stratified partition of the full uTHCD pool, evaluates all ten
corruptions × five severities, and reports **mCE@top-1** (VGG16_BN = 1.0) as
mean ± std over folds, plus cross-fold ranking stability (Kendall's τ).
*Paper Section 4.3.*

Run all commands from the repository root with `export PYTHONPATH=src`.

## Reproduce

```bash
# 1. Train: 9 models x 5 folds (distribute across GPUs as you like)
for f in 0 1 2 3 4; do for m in vgg16_bn googlenet swin_t efficientnet_b0 \
    squeezenet1_0 convnext_tiny regnet_x_400mf shufflenet_v2_x0_5 mnasnet0_5; do
  python3 experiments/01_corruption_benchmark_cv/code/kfold_train.py \
      --model $m --fold $f --data_dir dataset --out_dir outputs_kfold
done; done
# (or use code/run_kfold_parallel.py --models ... --data_dir dataset to schedule them)

# 2. Corruption evaluation (version A), per model per fold
for f in 0 1 2 3 4; do for m in <same nine models>; do
  python3 experiments/01_corruption_benchmark_cv/code/run_eval_A.py --model $m --fold $f
done; done

# 3. Aggregate -> summary tables
python3 experiments/01_corruption_benchmark_cv/code/aggregate_kfold.py --kfold_dir outputs_kfold

# 4. Figure
python3 experiments/01_corruption_benchmark_cv/code/plot_kfold_results.py
```

## Shipped results (`results/tables/`)

* `kfold_mce_summary_A.csv` — mCE@top-1 mean ± std per model
* `kfold_clean_accuracy.csv` — clean Top-1 mean ± std per model
* `kfold_ranking_stability.csv` — all pairwise fold Kendall's τ
* `kfold_aggregate.json`, `percorr_acc_A.json` — full aggregate / per-corruption accuracy

## Key numbers (mCE@top-1, mean ± std over 5 folds)

Swin 0.99 · VGG16_BN 1.00 · ConvNeXt 1.01 · GoogLeNet 1.20 · EfficientNet 1.20 ·
SqueezeNet 1.25 · RegNet 1.79 · **ShuffleNetV2 3.19 · MnasNet 4.22**.
Clean Top-1 ≥ 96% for all; ranking stable across folds (Kendall's τ 0.50–0.83).

Checkpoints (~12 GB) are not bundled — see `checkpoints/MANIFEST.md`.
