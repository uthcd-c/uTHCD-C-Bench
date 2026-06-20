# 03 — Writer-independent split

The strictest generalisation test: a **writer-disjoint** 70:30 split (no writer
shared between train and test), compared against the conventional
writer-*dependent* 70:30 baseline. *Paper Section 4.3.* This is the only
experiment whose checkpoints and per-model results are **bundled**, so it
reproduces fully offline.

Writer identity is parsed from each filename (`<writer>_<class>.bmp`); the split
uses `sklearn.GroupShuffleSplit` grouped by writer, with the online-capture
suffix (`192s`) merged into its base writer so none can leak. Three independent
writer partitions (seeds 1 / 42 / 123) give mean ± std.

Run from the repository root with `export PYTHONPATH=src`.

## Reproduce the table & figure offline (no dataset, no GPU)

```bash
python3 experiments/03_writer_independent/code/aggregate_writer_indep.py   # uses shipped outputs_writer/ + outputs_split/ratio_70/
python3 experiments/03_writer_independent/code/plot_writer_indep.py        # -> results/figures/writer_indep_combined.png
```

## Re-evaluate from the bundled checkpoints (dataset + GPU)

```bash
bash scripts/link_checkpoints.sh                 # expose checkpoints/ in the nested layout
for s in 1 42 123; do
  python3 experiments/03_writer_independent/code/eval_writer_indep.py --seed $s
done
python3 experiments/03_writer_independent/code/eval_writer_dependent.py --ratio 0.7
python3 experiments/03_writer_independent/code/aggregate_writer_indep.py
```

## Retrain from scratch (dataset + GPU)

```bash
for s in 1 42 123; do for m in <nine models>; do
  python3 experiments/03_writer_independent/code/train_writer_indep.py --model $m --seed $s
done; done
for m in <nine models>; do
  python3 experiments/03_writer_independent/code/train_writer_dependent.py --model $m --ratio 0.7
done
```
If a Swin run diverges (≈0.6% accuracy, stuck from epoch 1), restart it with a
different initialisation while keeping the same writer split:
`... train_writer_indep.py --model swin_t --seed 123 --init_seed 124`.

## Key findings

* Clean accuracy barely changes on unseen writers: mean Top-1 96.93% → 96.81%
  (gap **0.12 pp**; per-model ≤ 0.62 pp).
* Robustness ranking preserved: Kendall's τ = 0.67, Spearman's ρ = 0.78.
* MnasNet's mild-noise collapse persists on unseen writers (≈98% top-1 error);
  ShuffleNetV2's apparent mild-noise robustness was partly writer-dependent
  (mCE@top-1 2.61 → 3.63).

Bundled: 36 checkpoints (`checkpoints/wi_*.pt`, `wd_70_*.pt`) and per-model JSON
under `outputs_writer/`, `outputs_split/ratio_70/`.
