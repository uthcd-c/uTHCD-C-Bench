#!/usr/bin/env python3
"""
Compute stratified 5-fold splits over the combined train+val pool.
Run ONCE before any training. Outputs: mnasnet_robustness/fold_indices.json

Split seed is fixed at 0 (independent of any training seed) so all 30 runs
see exactly the same fold boundaries.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import ImageFolder

MNE_ROOT  = Path(__file__).parent.parent
REPO_ROOT = MNE_ROOT.parent
DATASET   = REPO_ROOT / "dataset"

N_SPLITS   = 5
SPLIT_SEED = 0


def main():
    ds_train = ImageFolder(str(DATASET / "train"))
    ds_val   = ImageFolder(str(DATASET / "val"))

    assert ds_train.class_to_idx == ds_val.class_to_idx, (
        "class_to_idx mismatch between train/ and val/ — cannot safely combine."
    )

    all_samples = ds_train.samples + ds_val.samples
    all_labels  = [s[1] for s in all_samples]
    n_total     = len(all_samples)
    n_classes   = len(ds_train.classes)

    print(f"Combined pool: {n_total} samples | {n_classes} classes")

    skf   = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(n_total), all_labels)
    ):
        folds.append({
            "fold":    fold_idx,
            "train":   train_idx.tolist(),
            "val":     val_idx.tolist(),
            "n_train": int(len(train_idx)),
            "n_val":   int(len(val_idx)),
        })
        print(f"  Fold {fold_idx}: {len(train_idx)} train / {len(val_idx)} val")

    out = {
        "n_splits":    N_SPLITS,
        "split_seed":  SPLIT_SEED,
        "n_total":     n_total,
        "n_classes":   n_classes,
        "class_to_idx": ds_train.class_to_idx,
        "folds":       folds,
    }

    out_path = MNE_ROOT / "fold_indices.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
