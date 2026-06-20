# 04 — Mechanism of corruption-induced collapse (MNASNet / BatchNorm)

Explains *why* the most compact BatchNorm models (MnasNet, ShuffleNetV2) collapse
under mild corruption. *Paper Section 4.4.*

**Root cause.** MNASNet's default BatchNorm momentum is `0.0003`. After one epoch
(~1572 steps) the running statistics are only ≈ 37% converged from their initial
(mean 0, var 1) — `(1-0.0003)^1572 ≈ 0.62`. In eval mode, BatchNorm with
unconverged statistics on near-binary inputs collapses all logits to one class.
Fine-tuning from ImageNet inherits and never overwrites this, so it collapses even
with adapted statistics; training from scratch with momentum raised to `0.01`
fixes it. The fix is in `scripts/train_fold.py` (the `finetune_bnfix` /
`random_init` paths set `m.momentum = 0.01`).

This experiment runs a 2 inits × 3 seeds × 5 folds design (30 runs):
`random_init` vs `finetune`, no Gaussian augmentation, and compares the relative
accuracy drop under Gaussian noise with a Welch t-test.

Run from the repository root with `export PYTHONPATH=src`.

## Reproduce

```bash
# single run
python3 experiments/04_collapse_mechanism/scripts/train_fold.py --init random_init --fold 0 --seed 1
python3 experiments/04_collapse_mechanism/scripts/train_fold.py --init finetune    --fold 0 --seed 1

# full 30-run sweep (schedules all init x seed x fold jobs)
python3 experiments/04_collapse_mechanism/code/run_all.py --max-parallel 4

# aggregate -> summary.csv + comparison.csv
python3 experiments/04_collapse_mechanism/scripts/aggregate_results.py
```

`configs/{random_init,finetune}.yaml` hold the per-init hyperparameters;
`code/fold_indices.json` pins the fold partition. `code/bn_adaptation.py`
demonstrates the complementary test-time fix (recomputing BN statistics on
corrupted data recovers much of the lost accuracy without retraining).

## Shipped results

* `results/` — all 30 `per_run/*.json`, plus `summary.csv` and `comparison.csv`
* `results/tables/mnasnet_collapse_{summary,comparison}.csv` (top-level copies)

## Key numbers (relative accuracy drop under Gaussian noise)

Fine-tune: **99.2 ± 0.06%** drop (always collapses). Random-init (BN-fixed):
**20.0 ± 19.9%** drop. Difference significant for clean, every severity, and the
mean (p < 0.001). Checkpoints (~0.2 GB) not bundled — see `checkpoints/MANIFEST.md`.
