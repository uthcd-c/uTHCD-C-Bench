#!/usr/bin/env python3
"""
kfold_train.py
--------------
Stratified k-fold cross-validation training for all 9 uTHCD-C architectures.

Merges the existing train/val/test splits into a single pool, then creates
n_splits stratified folds. For fold k:
  - test  : fold k  (~20% of data)
  - val   : 10% of remaining folds (for early stopping)
  - train : remaining ~72%

Fold indices are saved as JSON alongside the checkpoint so that
eval_kfold_corruption.py can reconstruct the exact same test split.

Usage:
  python kfold_train.py --model convnext_tiny --fold 0 --data_dir /path/to/dataset
  # Run for all 9 models × 5 folds (45 jobs total)
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    vgg16_bn, VGG16_BN_Weights,
    googlenet, GoogLeNet_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    regnet_x_400mf, RegNet_X_400MF_Weights,
    mnasnet0_5, MNASNet0_5_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    squeezenet1_0, SqueezeNet1_0_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    swin_t, Swin_T_Weights,
)
from sklearn.model_selection import StratifiedKFold

SUPPORTED_MODELS = [
    "vgg16_bn", "googlenet", "efficientnet_b0", "regnet_x_400mf",
    "mnasnet0_5", "shufflenet_v2_x0_5", "squeezenet1_0",
    "convnext_tiny", "swin_t",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 1):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float):
        if self.best_score is None:
            self.best_score = val_loss
            return
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------
@dataclass
class FoldMetrics:
    fold: int
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    test_loss: float
    test_acc: float
    total_epochs_run: int
    elapsed_sec: float
    train_indices: List[int]
    val_indices: List[int]
    test_indices: List[int]


# ---------------------------------------------------------------------------
# Merged dataset (combines train/ val/ test/ into one flat pool)
# ---------------------------------------------------------------------------
class MergedImageFolder(Dataset):
    """
    Loads images from multiple ImageFolder roots (e.g. train/, val/, test/)
    and presents them as a single flat dataset with a unified class mapping.
    """

    def __init__(self, roots: List[str], transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        # Build unified class mapping from first root (all roots share the same classes)
        ref = ImageFolder(roots[0])
        self.classes = ref.classes
        self.class_to_idx = ref.class_to_idx

        for root in roots:
            ds = ImageFolder(root)
            # Remap class indices to the unified mapping
            for path, local_idx in ds.samples:
                class_name = ds.classes[local_idx]
                unified_idx = self.class_to_idx[class_name]
                self.samples.append((path, unified_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @property
    def targets(self) -> List[int]:
        return [s[1] for s in self.samples]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def build_train_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_eval_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    n = model_name.lower()
    if n == "vgg16_bn":
        m = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif n == "googlenet":
        m = googlenet(weights=GoogLeNet_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        for aux_name in ["aux1", "aux2"]:
            aux = getattr(m, aux_name, None)
            if aux is not None and hasattr(aux, "fc2"):
                aux.fc2 = nn.Linear(aux.fc2.in_features, num_classes)
    elif n == "efficientnet_b0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif n == "regnet_x_400mf":
        m = regnet_x_400mf(weights=RegNet_X_400MF_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == "mnasnet0_5":
        m = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif n == "shufflenet_v2_x0_5":
        m = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == "squeezenet1_0":
        m = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif n == "convnext_tiny":
        m = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif n == "swin_t":
        m = swin_t(weights=Swin_T_Weights.DEFAULT)
        m.head = nn.Linear(m.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {SUPPORTED_MODELS}")
    return m


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, targets)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="K-fold CV training for uTHCD-C")
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-based)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset root containing train/, val/, test/ subdirectories")
    parser.add_argument("--out_dir", type=str, default="outputs_kfold",
                        help="Root output directory; results saved to out_dir/<model>/fold_<k>/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_if_done", action="store_true",
                        help="Exit silently if best.pt already exists for this fold")
    args = parser.parse_args()

    assert 0 <= args.fold < args.n_splits, f"--fold must be in [0, {args.n_splits - 1}]"

    fold_dir  = Path(args.out_dir) / args.model / f"fold_{args.fold}"
    best_ckpt = fold_dir / "checkpoints" / "best.pt"
    if args.skip_if_done and best_ckpt.exists():
        print(f"[{args.model}|fold {args.fold}] Already done — skipping.")
        return

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Fold: {args.fold}/{args.n_splits}")

    # ------------------------------------------------------------------
    # Build merged dataset (no transforms yet — applied per-subset below)
    # ------------------------------------------------------------------
    roots = [
        os.path.join(args.data_dir, "train"),
        os.path.join(args.data_dir, "val"),
        os.path.join(args.data_dir, "test"),
    ]
    full_ds = MergedImageFolder(roots, transform=None)
    num_classes = len(full_ds.classes)
    all_targets = np.array(full_ds.targets)
    all_indices = np.arange(len(full_ds))
    print(f"Full dataset: {len(full_ds)} samples, {num_classes} classes")

    # ------------------------------------------------------------------
    # Stratified k-fold split
    # ------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    splits = list(skf.split(all_indices, all_targets))
    trainval_idx, test_idx = splits[args.fold]

    # Within trainval, hold out 10% for validation (stratified)
    val_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed + args.fold)
    trainval_targets = all_targets[trainval_idx]
    train_local, val_local = next(iter(val_skf.split(trainval_idx, trainval_targets)))
    train_idx = trainval_idx[train_local]
    val_idx   = trainval_idx[val_local]

    print(f"Fold {args.fold}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ------------------------------------------------------------------
    # Build subset datasets with appropriate transforms
    # ------------------------------------------------------------------
    train_tfm = build_train_transforms()
    eval_tfm  = build_eval_transforms()

    class TransformSubset(Dataset):
        def __init__(self, base_ds, indices, transform):
            self.base_ds   = base_ds
            self.indices   = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            path, label = self.base_ds.samples[self.indices[i]]
            from PIL import Image
            img = Image.open(path).convert("RGB")
            return self.transform(img), label

    train_ds = TransformSubset(full_ds, train_idx, train_tfm)
    val_ds   = TransformSubset(full_ds, val_idx,   eval_tfm)
    test_ds  = TransformSubset(full_ds, test_idx,  eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model     = build_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ------------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------------
    fold_dir  = Path(args.out_dir) / args.model / f"fold_{args.fold}"
    ckpt_dir  = fold_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best.pt"

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_loss, best_val_acc, best_epoch = float("inf"), 0.0, 0
    start = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        print(f"[{args.model}|fold {args.fold}] Epoch {epoch+1}/{args.epochs} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_acc  = vl_acc
            best_epoch    = epoch
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "class_to_idx": full_ds.class_to_idx,
                "fold": args.fold,
                "n_splits": args.n_splits,
                "args": vars(args),
            }, best_ckpt)

        early_stopper(vl_loss)
        if early_stopper.should_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    elapsed = time.time() - start

    print(f"Fold {args.fold} test_acc={test_acc:.4f} | elapsed={elapsed:.0f}s")

    # ------------------------------------------------------------------
    # Save fold metrics + indices
    # ------------------------------------------------------------------
    metrics = FoldMetrics(
        fold=args.fold,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc=best_val_acc,
        test_loss=test_loss,
        test_acc=test_acc,
        total_epochs_run=epoch + 1,
        elapsed_sec=elapsed,
        train_indices=train_idx.tolist(),
        val_indices=val_idx.tolist(),
        test_indices=test_idx.tolist(),
    )
    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(asdict(metrics), f, indent=2)

    print(f"Saved checkpoint -> {best_ckpt}")
    print(f"Saved metrics    -> {fold_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
