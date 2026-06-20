# 06 — Representation-level analysis (t-SNE + CKA)

Two complementary representation analyses. *Paper Sections 4.6 and 4.7.*

Run from the repository root with `export PYTHONPATH=src`.

## t-SNE of the most-confused classes (Sec. 4.6)

`code/tsne_cluster_analysis.ipynb` extracts penultimate-layer embeddings and
visualises the twelve most-confused character classes, clean vs. corrupted, to
show how robust and fragile models organise their feature space differently.
Open it with Jupyter:

```bash
jupyter notebook experiments/06_representation_analysis/code/tsne_cluster_analysis.ipynb
```

t-SNE is sensitive to its settings; the notebook records the perplexity / seed
used so the embedding is reproducible.

## Centered Kernel Alignment (Sec. 4.7)

CKA quantifies layerwise representational similarity between a robust and a
fragile architecture, on clean and corrupted inputs:

```bash
python3 experiments/06_representation_analysis/code/cka_convnext_mnasnet.py          # clean
python3 experiments/06_representation_analysis/code/cka_convnext_mnasnet_noise.py     # corrupted
python3 experiments/06_representation_analysis/code/cka_convnext_squeezenet.py
python3 experiments/06_representation_analysis/code/cka_convnext_squeezenet_noisy.py
```

These require trained checkpoints (regenerate via experiment 01) and the dataset.

## Per-class top-k error

`code/perclass_topk.py` produces the per-class top-1/top-3 error breakdowns used
to identify the confused classes analysed above.

## Takeaway

The robust LayerNorm-based models keep the confused classes separable under
corruption, while the fragile BatchNorm models collapse them; CKA shows their
representations diverge sharply in the deeper layers under noise.
