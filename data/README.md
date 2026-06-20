# Dataset

This repository does **not** ship the dataset. Obtain the **uTHCD** (unconstrained
Tamil Handwritten Character Database, ≈ 90,950 samples, 156 classes) and place it
at `dataset/` in the repository root, in ImageFolder form:

```
dataset/<split>/<class_id>/<writer>_<class_id>.bmp
# e.g.  dataset/train/001/0292_001.bmp
```

Notes:

* The code merges the `train/`, `val/`, and `test/` split folders into one pool
  and **re-partitions** it for each experiment (cross-validation folds, split
  ratios, or writer-disjoint splits). The dataset's original split assignment is
  therefore irrelevant — any partition into those folders works, and a `val/`
  folder is optional.
* **Writer identity** is the token *before* the underscore in each filename
  (`0292` in `0292_001.bmp`). Only the writer-independent experiment (03) uses it;
  an online-capture suffix such as `192s` is stripped so `192` and `192s` are
  treated as the same writer.
* Images are loaded as RGB and resized to 224×224 (ImageNet-style preprocessing).

Place the dataset once here; every experiment reads from `./dataset` relative to
the repository root.
