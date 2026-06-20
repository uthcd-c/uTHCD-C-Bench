#!/usr/bin/env python3
"""
Train MNASNet0_5 for one (init_strategy, fold, seed) combination, then
evaluate the best checkpoint on the held-out test set under clean conditions
and Gaussian noise at severities 1–5.

All outputs land under mnasnet_robustness/:
  checkpoints/{init}/seed_{seed}/fold_{fold}/best.pt
  results/per_run/{init}_seed{seed}_fold{fold}.json
  logs/{init}_seed{seed}_fold{fold}.log

Usage:
  python scripts/train_fold.py --init random_init --fold 0 --seed 1
  python scripts/train_fold.py --init finetune    --fold 2 --seed 42
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mnasnet0_5, MNASNet0_5_Weights

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MNE_ROOT  = Path(__file__).parent.parent   # mnasnet_robustness/
REPO_ROOT = MNE_ROOT.parent                # uTHCD-C/
DATASET   = REPO_ROOT / "dataset"
FOLD_FILE = MNE_ROOT / "fold_indices.json"

# ---------------------------------------------------------------------------
# Per-init hyperparameter configs (loaded from YAML by run_all.py or used directly)
# ---------------------------------------------------------------------------
CONFIGS = {
    "random_init":    dict(lr=3e-4, weight_decay=1e-4, batch_size=32,
                           epochs=30, patience=7, min_delta=0.0, num_workers=4),
    "finetune":       dict(lr=1e-4, weight_decay=1e-4, batch_size=32,
                           epochs=30, patience=7, min_delta=0.0, num_workers=4),
    "finetune_bnfix": dict(lr=1e-4, weight_decay=1e-4, batch_size=32,
                           epochs=30, patience=7, min_delta=0.0, num_workers=4),
}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True   # fastest conv kernel


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def aug_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def clean_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class CombinedDataset(Dataset):
    """Merges dataset/train and dataset/val into a single indexed pool."""

    def __init__(self, transform=None):
        ds_tr = ImageFolder(str(DATASET / "train"))
        ds_vl = ImageFolder(str(DATASET / "val"))
        assert ds_tr.class_to_idx == ds_vl.class_to_idx, \
            "class_to_idx mismatch between train/ and val/"
        self.samples      = ds_tr.samples + ds_vl.samples
        self.class_to_idx = ds_tr.class_to_idx
        self.classes      = ds_tr.classes
        self.transform    = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


class GaussianTestDataset(Dataset):
    """Test ImageFolder with optional Gaussian noise applied before transforms."""

    def __init__(self, severity: int = 0, transform=None):
        ds = ImageFolder(str(DATASET / "test"))
        self.samples   = ds.samples
        self.severity  = severity
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.severity > 0:
            arr   = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 10.0 * self.severity, arr.shape)
            img   = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


def make_test_loader(severity: int) -> DataLoader:
    ds = GaussianTestDataset(severity=severity, transform=clean_transform())
    g  = torch.Generator()
    g.manual_seed(0)
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=4,
                      pin_memory=True, worker_init_fn=_seed_worker, generator=g)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(init: str, num_classes: int) -> nn.Module:
    if init in ("finetune", "finetune_bnfix"):
        m = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT)
    else:
        m = mnasnet0_5(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def load_best_for_eval(ckpt_path: str, num_classes: int) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt.get("model_state_dict", ckpt)
    # Patch missing _version metadata (torchvision >= 0.15 requirement)
    if not hasattr(sd, "_metadata"):
        sd._metadata = {}
    sd._metadata[""] = {"version": 2}
    m = mnasnet0_5(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    missing, unexpected = m.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, \
        f"Checkpoint mismatch — missing: {missing}, unexpected: {unexpected}"
    return m


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0
        self.triggered = False

    def __call__(self, val_loss: float) -> bool:
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best    = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    loss_sum = correct = total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = criterion(logits, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (logits.detach().argmax(1) == targets).sum().item()
        total    += targets.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits    = model(imgs)
        loss      = criterion(logits, targets)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (logits.argmax(1) == targets).sum().item()
        total    += targets.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_top1(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        correct += (model(imgs).argmax(1) == targets).sum().item()
        total   += targets.size(0)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------
def run_robustness_eval(model, device) -> dict:
    res = {}
    res["clean"] = eval_top1(model, make_test_loader(0), device)
    for s in range(1, 6):
        res[f"gaussian_s{s}"] = eval_top1(model, make_test_loader(s), device)
    noisy_vals       = [res[f"gaussian_s{s}"] for s in range(1, 6)]
    res["mean_noisy"]         = float(np.mean(noisy_vals))
    res["relative_drop_pct"]  = 100.0 * (res["clean"] - res["mean_noisy"]) / res["clean"]
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True, choices=["random_init", "finetune", "finetune_bnfix"])
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    cfg      = CONFIGS[args.init]
    run_name = f"{args.init}_seed{args.seed}_fold{args.fold}"

    ckpt_dir   = MNE_ROOT / "checkpoints" / args.init / f"seed_{args.seed}" / f"fold_{args.fold}"
    result_dir = MNE_ROOT / "results" / "per_run"
    log_dir    = MNE_ROOT / "logs"
    for d in [ckpt_dir, result_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Redirect all print output to the log file (run_all.py also redirects stdout)
    log_path = log_dir / f"{run_name}.log"
    log_f    = open(log_path, "w", buffering=1)
    _orig_stdout = sys.stdout

    class Tee:
        def write(self, s): _orig_stdout.write(s); log_f.write(s)
        def flush(self):    _orig_stdout.flush();  log_f.flush()

    sys.stdout = Tee()

    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print(f"Run : {run_name}")
    print(f"Init: {args.init} | Fold: {args.fold} | Seed: {args.seed}")
    print(f"Config: {cfg}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Fold indices --------------------------------------------------
    with open(FOLD_FILE) as f:
        fold_data = json.load(f)
    num_classes = fold_data["n_classes"]
    fold_info   = fold_data["folds"][args.fold]
    train_idx   = fold_info["train"]
    val_idx     = fold_info["val"]
    print(f"Fold {args.fold}: {len(train_idx)} train / {len(val_idx)} val | {num_classes} classes")

    # ---- Datasets & loaders -------------------------------------------
    # Both train and val folds use the augmented transform during the training
    # loop. This is intentional: with random init, BN running stats start at
    # (mean=0, var=1) and are updated from augmented-input batches. Evaluating
    # with a mismatched clean transform in early epochs produces near-chance
    # val accuracy, making early stopping unreliable. The final robustness
    # evaluation always uses the clean/deterministic transform on the held-out
    # test set (see run_robustness_eval), which runs after BN stats have
    # converged over 30 epochs of training.
    aug_pool   = CombinedDataset(transform=aug_transform())

    fold_train = Subset(aug_pool, train_idx)
    fold_val   = Subset(aug_pool, val_idx)

    g_train = torch.Generator()
    g_train.manual_seed(args.seed)

    train_loader = DataLoader(
        fold_train, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True,
        worker_init_fn=_seed_worker, generator=g_train,
    )
    val_loader = DataLoader(
        fold_val, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )

    # ---- Model ---------------------------------------------------------
    model = build_model(args.init, num_classes).to(device)

    # MNASNet's BN momentum is 0.0003 (tuned for ImageNet-scale training with
    # millions of steps). With random init starting from (mean=0, var=1) and
    # only ~47 k training steps total, running stats barely converge by epoch
    # 30, so eval-mode BN is essentially an identity function early on and
    # collapses all predictions to one class. Raising momentum to 0.01 ensures
    # the running stats reflect the actual feature distribution by the end of
    # epoch 1.  For finetune, pretrained stats update at the original slow rate
    # (desirable: keeps ImageNet calibration during early fine-tuning).
    if args.init in ("random_init", "finetune_bnfix"):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.01

    n_params = sum(p.numel() for p in model.parameters())
    print(f"MNASNet0_5 | Init={args.init} | Params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    stopper   = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])

    # ---- Training ------------------------------------------------------
    train_hist = {"loss": [], "acc": []}
    val_hist   = {"loss": [], "acc": []}
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_epoch    = 0
    t0 = time.time()

    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc = eval_one_epoch(model, val_loader,   criterion, device)

        train_hist["loss"].append(tr_loss); train_hist["acc"].append(tr_acc)
        val_hist["loss"].append(vl_loss);   val_hist["acc"].append(vl_acc)

        print(
            f"Ep {epoch+1:02d}/{cfg['epochs']}  "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
            f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}"
        )

        if vl_loss < best_val_loss - cfg["min_delta"]:
            best_val_loss = vl_loss
            best_val_acc  = vl_acc
            best_epoch    = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch":            epoch,
                    "best_val_loss":    best_val_loss,
                    "best_val_acc":     best_val_acc,
                    "run_name":         run_name,
                    "init":             args.init,
                    "fold":             args.fold,
                    "seed":             args.seed,
                    "num_classes":      num_classes,
                    "config":           cfg,
                },
                ckpt_dir / "best.pt",
            )

        if stopper(vl_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed:.0f}s  |  best val_loss={best_val_loss:.4f} @ epoch {best_epoch+1}")

    # ---- Robustness evaluation ----------------------------------------
    print("\n--- Robustness Evaluation on held-out test set ---")
    best_model = load_best_for_eval(str(ckpt_dir / "best.pt"), num_classes).to(device)
    robustness = run_robustness_eval(best_model, device)
    for k, v in robustness.items():
        print(f"  {k:<22}: {v:.2f}")

    # ---- Save result JSON ---------------------------------------------
    result = {
        "run_name":       run_name,
        "init":           args.init,
        "fold":           args.fold,
        "seed":           args.seed,
        "config":         cfg,
        "best_epoch":     best_epoch,
        "best_val_loss":  best_val_loss,
        "best_val_acc":   best_val_acc,
        "elapsed_sec":    elapsed,
        "train_history":  train_hist,
        "val_history":    val_hist,
        "robustness":     robustness,
    }
    out_path = result_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved result → {out_path}")
    print("=" * 60)

    sys.stdout = _orig_stdout
    log_f.close()


if __name__ == "__main__":
    main()
