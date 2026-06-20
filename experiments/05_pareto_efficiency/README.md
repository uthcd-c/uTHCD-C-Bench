# 05 — Robustness–efficiency Pareto analysis

Places the nine architectures on accuracy/robustness vs. efficiency Pareto
frontiers, to identify models that are both efficient and resilient.
*Paper Sections 4.2 and 4.5.*

Run from the repository root.

## Reproduce (from shipped summaries; no dataset, no GPU)

```bash
python3 experiments/05_pareto_efficiency/code/plot_pareto.py
# -> results/figures/pareto_mce.png       (mCE@top-1 vs throughput)
# -> results/figures/pareto_accuracy.png  (clean Top-1 vs throughput)
```

## Inputs (`results/tables/`)

* `throughput_A100.json` — measured inference throughput (images/s) per model
* `kfold_mce_summary_A.csv` — robustness (mCE@top-1) per model
* `kfold_clean_accuracy.csv` — clean Top-1 per model

The efficiency axis is measured throughput (shipped with the repo); parameter
count can be substituted by editing `plot_pareto.py`. The frontier is computed as
the non-dominated set (higher throughput and better quality).

## Takeaway

ConvNeXt-Tiny and Swin-Tiny achieve top robustness (mCE ≈ 1.0) but at low
throughput; the compact BatchNorm models (ShuffleNetV2, MnasNet) are the most
efficient yet the least robust. The frontier makes the trade-off explicit and
feeds the model-selection guidelines (experiment 07).
