#!/usr/bin/env python3
"""
eval_kfold_corruption.py
------------------------
Runs the full 10-corruption × 5-severity sweep on a k-fold checkpoint and
computes mCE@top-k (k=1,3,5) relative to the VGG16_BN baseline trained on
the same fold.

Must be run AFTER both the target model AND vgg16_bn have been trained for
the same fold (kfold_train.py), because mCE normalisation requires the
VGG16_BN fold errors.

Usage:
  python eval_kfold_corruption.py \
      --model convnext_tiny \
      --fold 0 \
      --kfold_dir outputs_kfold \
      --data_dir /path/to/dataset

Output: outputs_kfold/<model>/fold_<k>/corruption_results.json
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
import io
from torch.utils.data import DataLoader, Dataset
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

# Ten corruption types matching the paper (Section 3.3)
CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "gaussian_blur",
    "defocus_blur",
    "stroke_thinning",
    "elastic",
    "pixelate",
    "contrast",
    "scale",
]
SEVERITIES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Corruption functions (paper-faithful implementations)
# ---------------------------------------------------------------------------
def apply_corruption(img: Image.Image, corruption: str, severity: int) -> Image.Image:
    s = max(1, min(5, int(severity)))
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]

    if corruption == "gaussian_noise":
        sigma = [8, 12, 18, 26, 38][s - 1]
        noise = np.random.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    if corruption == "shot_noise":
        # Poisson noise: scale controls amount
        scale = [60, 25, 12, 5, 3][s - 1]
        arr_norm = arr / 255.0
        noisy = np.random.poisson(arr_norm * scale) / float(scale)
        return Image.fromarray(np.clip(noisy * 255, 0, 255).astype(np.uint8))

    if corruption == "impulse_noise":
        # Salt-and-pepper
        ratio = [0.03, 0.06, 0.09, 0.17, 0.27][s - 1]
        out = arr.copy().astype(np.uint8)
        mask = np.random.rand(h, w) < ratio
        salt = np.random.rand(h, w) < 0.5
        out[mask & salt]  = 255
        out[mask & ~salt] = 0
        return Image.fromarray(out)

    if corruption == "gaussian_blur":
        radius = [1, 2, 3, 4, 6][s - 1]
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "defocus_blur":
        # Disk-like blur via repeated box filter (approximation)
        iterations = [1, 2, 3, 4, 5][s - 1]
        out = img
        for _ in range(iterations):
            out = out.filter(ImageFilter.BoxBlur(radius=2))
        return out

    if corruption == "stroke_thinning":
        # Morphological erosion of dark strokes on white background
        gray = np.array(img.convert("L"))
        # Threshold to binary: strokes are dark (<128)
        binary = (gray < 128).astype(np.uint8) * 255
        # Erode by shrinking dark regions (erosion of inverted = dilation of background)
        eroded = Image.fromarray(binary).filter(ImageFilter.MinFilter(size=3 + 2 * (s - 1)))
        eroded_arr = np.array(eroded)
        # Reconstruct RGB: thin strokes on white
        out = np.ones_like(arr, dtype=np.uint8) * 255
        out[eroded_arr < 128] = 0
        return Image.fromarray(out)

    if corruption == "elastic":
        # Elastic deformation using random displacement fields
        from PIL import Image as PILImage
        magnitude = [2, 4, 6, 8, 10][s - 1]
        dx = np.random.uniform(-magnitude, magnitude, (h, w)).astype(np.float32)
        dy = np.random.uniform(-magnitude, magnitude, (h, w)).astype(np.float32)
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        new_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
        new_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)
        # Remap via PIL (use affine-style sampling)
        arr_uint = arr.astype(np.uint8)
        out = np.zeros_like(arr_uint)
        # Vectorised nearest-neighbour sampling
        src_x = new_x.astype(int)
        src_y = new_y.astype(int)
        out = arr_uint[src_y, src_x]
        return Image.fromarray(out)

    if corruption == "pixelate":
        block = [16, 10, 7, 5, 3][s - 1]
        small = img.resize((w // block, h // block), Image.BOX)
        return small.resize((w, h), Image.NEAREST)

    if corruption == "contrast":
        from PIL import ImageEnhance
        factor = [0.75, 0.5, 0.35, 0.2, 0.1][s - 1]
        return ImageEnhance.Contrast(img).enhance(factor)

    if corruption == "scale":
        target_size = [128, 64, 32, 16, 8][s - 1]
        small = img.resize((target_size, target_size), Image.BILINEAR)
        return small.resize((w, h), Image.BICUBIC)

    return img


# ---------------------------------------------------------------------------
# Fold-aware dataset: loads samples by their stored global indices
# ---------------------------------------------------------------------------
class FoldDataset(Dataset):
    """
    Loads images identified by their global indices in the merged dataset.
    Indices and path list come from the fold metrics JSON saved during training.
    """

    def __init__(self, all_samples: List[Tuple[str, int]],
                 indices: List[int],
                 corruption: str = "clean",
                 severity: int = 1,
                 transform=None):
        self.samples    = [all_samples[i] for i in indices]
        self.corruption = corruption
        self.severity   = severity
        self.transform  = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.corruption != "clean":
            img = apply_corruption(img, self.corruption, self.severity)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Model factory (matches kfold_train.py — pretrained=False, just architecture)
# ---------------------------------------------------------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    n = model_name.lower()
    if n == "vgg16_bn":
        m = vgg16_bn(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif n == "googlenet":
        m = googlenet(weights=None, aux_logits=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif n == "regnet_x_400mf":
        m = regnet_x_400mf(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == "mnasnet0_5":
        m = mnasnet0_5(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif n == "shufflenet_v2_x0_5":
        m = shufflenet_v2_x0_5(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == "squeezenet1_0":
        m = squeezenet1_0(weights=None)
        m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    elif n == "convnext_tiny":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif n == "swin_t":
        m = swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m


def load_model(model_name: str, ckpt_path: str, num_classes: int,
               device: torch.device) -> nn.Module:
    model = build_model(model_name, num_classes)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Top-k accuracy
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_topk(model: nn.Module, loader: DataLoader,
                  device: torch.device, ks: Tuple[int, ...] = (1, 3, 5)):
    model.eval()
    correct = {k: 0 for k in ks}
    total   = 0
    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits  = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        for k in ks:
            topk_preds = logits.topk(k, dim=1).indices
            correct[k] += (topk_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)
    return {k: 100.0 * correct[k] / max(1, total) for k in ks}


def make_eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Build merged sample list (same order as kfold_train.py)
# ---------------------------------------------------------------------------
def build_merged_samples(data_dir: str) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    roots = [
        os.path.join(data_dir, "train"),
        os.path.join(data_dir, "val"),
        os.path.join(data_dir, "test"),
    ]
    ref = ImageFolder(roots[0])
    class_to_idx = ref.class_to_idx
    all_samples: List[Tuple[str, int]] = []
    for root in roots:
        ds = ImageFolder(root)
        for path, local_idx in ds.samples:
            unified_idx = class_to_idx[ds.classes[local_idx]]
            all_samples.append((path, unified_idx))
    return all_samples, class_to_idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="K-fold corruption evaluation for uTHCD-C")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fold",  type=int, required=True)
    parser.add_argument("--kfold_dir", type=str, default="outputs_kfold")
    parser.add_argument("--data_dir",  type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_if_done", action="store_true",
                        help="Skip if corruption_results.json already exists")
    args = parser.parse_args()

    fold_dir    = Path(args.kfold_dir) / args.model / f"fold_{args.fold}"
    results_out = fold_dir / "corruption_results.json"

    if args.skip_if_done and results_out.exists():
        print(f"Already done: {results_out} — skipping.")
        return

    # Load fold metadata
    with open(fold_dir / "metrics.json") as f:
        fold_meta = json.load(f)
    test_indices = fold_meta["test_indices"]
    num_classes  = 156  # fixed for uTHCD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {args.model} fold {args.fold} | device={device}")

    # Build full merged sample list (same order as training)
    all_samples, class_to_idx = build_merged_samples(args.data_dir)
    tfm = make_eval_transform()

    # Load target model
    ckpt_path = fold_dir / "checkpoints" / "best.pt"
    model = load_model(args.model, str(ckpt_path), num_classes, device)

    # Load VGG16_BN baseline for same fold (needed for mCE normalisation)
    vgg_fold_dir  = Path(args.kfold_dir) / "vgg16_bn" / f"fold_{args.fold}"
    vgg_ckpt_path = vgg_fold_dir / "checkpoints" / "best.pt"
    if not vgg_ckpt_path.exists():
        print(f"WARNING: VGG16_BN fold {args.fold} checkpoint not found at {vgg_ckpt_path}.")
        print("mCE will be stored as raw errors (no normalisation). Run vgg16_bn fold first.")
        vgg_model = None
    elif str(vgg_ckpt_path) == str(ckpt_path):
        # Evaluating vgg16_bn itself — reuse the same object to avoid loading a second copy
        vgg_model = model
    else:
        vgg_model = load_model("vgg16_bn", str(vgg_ckpt_path), num_classes, device)

    results: Dict = {
        "model": args.model,
        "fold":  args.fold,
        "clean": {},
        "corrupted": {},
        "mce": {},
    }

    # ------------------------------------------------------------------
    # Clean evaluation
    # ------------------------------------------------------------------
    clean_ds     = FoldDataset(all_samples, test_indices, "clean", 1, tfm)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)
    clean_topk   = evaluate_topk(model, clean_loader, device)
    results["clean"] = {f"top{k}": v for k, v in clean_topk.items()}
    print(f"  Clean: {clean_topk}")

    vgg_clean_topk = None
    if vgg_model is not None:
        vgg_clean_topk = evaluate_topk(vgg_model, clean_loader, device)

    # ------------------------------------------------------------------
    # Corruption sweep
    # ------------------------------------------------------------------
    # corruption_errors[corruption][severity][k] = error (= 100 - accuracy)
    corruption_errors: Dict = {}
    vgg_errors: Dict        = {}

    for corruption in CORRUPTIONS:
        corruption_errors[corruption] = {}
        vgg_errors[corruption]        = {}
        for sev in SEVERITIES:
            ds     = FoldDataset(all_samples, test_indices, corruption, sev, tfm)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=False)
            topk   = evaluate_topk(model, loader, device)
            corruption_errors[corruption][sev] = {f"top{k}": 100 - v for k, v in topk.items()}

            if vgg_model is not None:
                vgg_topk = evaluate_topk(vgg_model, loader, device)
                vgg_errors[corruption][sev] = {f"top{k}": 100 - v for k, v in vgg_topk.items()}

            print(f"  {corruption:20s} | sev {sev}: top1={topk[1]:.2f}%")

    results["corrupted"] = corruption_errors
    results["vgg_baseline_errors"] = vgg_errors

    # ------------------------------------------------------------------
    # Compute mCE@top-k (paper formula)
    # mCE_avg = mean over k in {1,3,5} of:
    #   (1/C) Σ_c (1/5) Σ_s E_m,c,s / E_VGG,c,s
    # Since we evaluate across the full test set (not per-corruption type
    # as a separate "c"), we use corruptions as c:
    #   CE_m,corr = (1/5) Σ_s E_m,corr,s / E_VGG,corr,s
    #   mCE@k = (1/|CORRUPTIONS|) Σ_corr CE_m,corr  (for top-k errors)
    # ------------------------------------------------------------------
    if vgg_model is not None:
        mce: Dict = {}
        for k_str in ["top1", "top3", "top5"]:
            ce_per_corruption = []
            for corruption in CORRUPTIONS:
                ce_sum = 0.0
                for sev in SEVERITIES:
                    e_m   = corruption_errors[corruption][sev][k_str]
                    e_vgg = vgg_errors[corruption][sev][k_str]
                    if e_vgg > 0:
                        ce_sum += e_m / e_vgg
                    else:
                        ce_sum += 1.0  # both near-zero: ratio = 1
                ce_per_corruption.append(ce_sum / len(SEVERITIES))
            mce[k_str] = sum(ce_per_corruption) / len(CORRUPTIONS)
        mce["avg"] = (mce["top1"] + mce["top3"] + mce["top5"]) / 3.0
        results["mce"] = mce
        print(f"  mCE@top1={mce['top1']:.4f}  mCE@top3={mce['top3']:.4f}  "
              f"mCE@top5={mce['top5']:.4f}  mCE_avg={mce['avg']:.4f}")
    else:
        print("  mCE skipped (no VGG16_BN baseline for this fold).")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {results_out}")


if __name__ == "__main__":
    main()
