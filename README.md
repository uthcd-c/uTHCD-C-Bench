# uTHCD-C — Corruption-Robustness Benchmark for Tamil Handwritten Character Recognition

Reproducibility repository for the full benchmark study: nine CNN/transformer
architectures are evaluated on the uTHCD Tamil handwritten-character dataset under
a ten-type corruption suite at five severities each, scored with the mean
Corruption Error at top-1 (**mCE@top-1**). The study establishes the robustness
ranking of the architectures, validates it three independent ways
(cross-validation, split-ratio sweep, writer-independent split), explains the
collapse of the most compact BatchNorm models, and analyses the
robustness–efficiency trade-off and the learned representations.

This repository contains everything needed to **reproduce every experiment,
table, and figure** in the paper. It deliberately excludes the manuscript and
author-response documents — it is a standalone guide for researchers.

---

## What it reproduces

| # | Experiment | Paper § | Key result | Artifact |
|---|---|---|---|---|
| 01 | [Corruption benchmark (5-fold CV)](experiments/01_corruption_benchmark_cv) | 4.3 | mCE@top-1 ranking; stable across folds (Kendall τ 0.50–0.83) | `results/tables/kfold_*` |
| 02 | [Split-ratio sweep (50→90)](experiments/02_split_ratio) | 4.3 | ranking preserved across ratios (mean τ 0.72) | `results/figures/split_ratio_combined.png` |
| 03 | [Writer-independent split](experiments/03_writer_independent) | 4.3 | clean gap 0.12 pp; ranking preserved (τ 0.67) | `results/figures/writer_indep_combined.png` |
| 04 | [Collapse mechanism (MNASNet BN)](experiments/04_collapse_mechanism) | 4.4 | BN-momentum root cause; fine-tune drop 99.2% vs random-init 20.0% (p<0.001) | `results/tables/mnasnet_collapse_*` |
| 05 | [Robustness–efficiency Pareto](experiments/05_pareto_efficiency) | 4.2 / 4.5 | Pareto frontiers (accuracy & mCE vs throughput) | `results/figures/pareto_*.png` |
| 06 | [Representation analysis (t-SNE, CKA)](experiments/06_representation_analysis) | 4.6 / 4.7 | confused-class embeddings; cross-architecture similarity | notebook + CKA scripts |
| 07 | [Model-selection guidelines](experiments/07_model_selection) | 4.8 | decision flowchart from the robustness/efficiency results | `results/figures/model_selection_flowchart.png` |

The nine architectures: `vgg16_bn`, `googlenet`, `swin_t`, `efficientnet_b0`,
`squeezenet1_0`, `convnext_tiny`, `regnet_x_400mf`, `shufflenet_v2_x0_5`,
`mnasnet0_5`. VGG16_BN is the mCE normalisation baseline (mCE = 1.0).

---

## Layout

```
.
├── README.md                 # this file
├── requirements.txt
├── data/README.md            # how to obtain & place the dataset
├── src/                      # shared library used by every experiment
│   ├── kfold_train.py            # model zoo, transforms, training loop, early stopping
│   ├── corruptions_A.py          # the 10 corruptions x 5 severities (incl. Zhang-Suen stroke thinning)
│   ├── eval_kfold_corruption.py  # eval helpers: merged dataset, model loading, top-k metrics
│   └── run_eval_A.py             # DetFold: deterministic per-sample corruption
├── experiments/01..07/       # per-experiment code + README
├── checkpoints/              # 36 writer-independent checkpoints (single dir) + MANIFEST.md
├── outputs_writer/ , outputs_split/ratio_70/   # writer-independent per-model JSON (ships, for offline repro)
├── results/
│   ├── tables/                   # every summary CSV/JSON that backs a paper table
│   └── figures/                  # every result figure
└── scripts/link_checkpoints.sh   # expose checkpoints/ in the nested eval layout
```

---

## Installation

```bash
pip install -r requirements.txt
```

A CUDA GPU is needed for training/evaluation. Tables and figures can be
regenerated from the shipped summaries on CPU.

## Dataset

See [`data/README.md`](data/README.md). Place uTHCD at `./dataset/` once; all
experiments read from there.

---

## How to run

**Contract:** run every command **from the repository root** with `src/` on the
import path:

```bash
export PYTHONPATH=src          # makes `import kfold_train`, `corruptions_A`, ... resolve
```

Each experiment directory has its own `README.md` with the exact train → evaluate
→ aggregate → plot commands. Two common entry points:

```bash
# Regenerate the robustness–efficiency figures from shipped summaries (no GPU, no dataset):
python3 experiments/05_pareto_efficiency/code/plot_pareto.py

# Reproduce the writer-independent table & figure offline (no GPU, no dataset):
python3 experiments/03_writer_independent/code/aggregate_writer_indep.py
python3 experiments/03_writer_independent/code/plot_writer_indep.py
```

The writer-independent experiment (03) is fully reproducible **offline** from the
shipped checkpoints and per-model JSON; the other experiments ship their result
summaries (so every table/figure regenerates) and the scripts to retrain from
scratch. See [`checkpoints/MANIFEST.md`](checkpoints/MANIFEST.md) for which
checkpoints are bundled vs. regenerated.

---

## Pretrained checkpoints

> **Double-blind review note.** To preserve author anonymity, the trained
> weights are **not** hosted at a fixed public location during review. The
> complete set of checkpoints will be **made available via an anonymous
> download link upon request** (e.g., through the handling editor).

Every result in the paper is reproducible **without** the checkpoints: each
architecture retrains from scratch with the shipped code and fixed seeds (see the
per-experiment READMEs). The checkpoints are provided only to let reviewers
verify the reported numbers without retraining.

| Archive | Experiment | Size |
|---|---|---|
| `uTHCD-C-ckpts-01-cross_validation.tar` | 01 — 5-fold CV | 12 GB |
| `uTHCD-C-ckpts-02-split_ratio.tar` | 02 — split-ratio | 3.9 GB |
| `uTHCD-C-ckpts-03-writer_independent.tar` | 03 — writer-independent | 2.4 GB |
| `uTHCD-C-ckpts-04-collapse_mechanism.tar` | 04 — MNASNet collapse | 0.14 GB |

Once the anonymous link has been provided, set it in
`scripts/fetch_checkpoints.sh` and run:

```bash
bash scripts/fetch_checkpoints.sh           # all experiments
bash scripts/fetch_checkpoints.sh 03        # just writer-independent (2.4 GB)
```

Each archive unpacks at the repo root into the layout the eval scripts expect and
ships a `SHA256SUMS-*.txt` for integrity checking. See
[`checkpoints/MANIFEST.md`](checkpoints/MANIFEST.md) for which weights each
archive contains.

---

## Protocol (shared by all experiments)

* **Training.** ImageNet-pretrained fine-tuning, AdamW (lr 3e-4, weight decay
  1e-4), batch size 32, up to 30 epochs, early-stopping patience 7. Identical for
  every architecture (deliberately, to isolate architectural effects).
* **Corruptions (version A).** Gaussian/shot/impulse noise, Gaussian/defocus blur,
  stroke thinning (Zhang-Suen skeletonisation + morphological re-dilation),
  elastic, pixelate, contrast, and scale; five severities each, applied
  deterministically per sample so all models see identical corrupted images.
* **Metric.** mCE@top-1, normalised by the VGG16_BN baseline within each split
  (lower = more robust). Only top-1 is used: the strong baseline has near-zero
  top-3/5 error, which makes the normalised mCE@top-3/5 numerically unstable.

## Notes

* **Determinism.** Splits are deterministic given their seeds. Training uses
  `cudnn.benchmark=True`, so retrained weights may differ negligibly from any
  bundled checkpoints while reproducing the same conclusions.
* **Swin-Tiny restarts.** Transformer fine-tuning at this learning rate
  occasionally diverges at initialisation; restart with a different `--init_seed`
  (see experiment 03's README). The bundled `wi_seed123_swin_t.pt` is such a
  restart.
* **Compute used for the reported results.** Three GPUs (one 40 GB and two 24 GB
  cards). Training jobs are independent per `(seed/fold, model)` and distribute
  trivially across devices with `CUDA_VISIBLE_DEVICES`.
