#!/usr/bin/env python3
"""
perclass_topk.py
----------------
Per-class top-1 vs top-5 recall under a single corruption+severity.

Tests whether top-k changes the per-class confusion picture: i.e. whether the
top-1 -> top-5 recovery is *concentrated* in specific characters (recoverable
confusions, actionable for downstream OCR re-ranking) rather than uniform.

Reuses the exact corruption / model / transform code from eval_kfold_corruption.py.
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).parent))
from eval_kfold_corruption import apply_corruption, load_model, make_eval_transform


class CorruptImageFolder(ImageFolder):
    def __init__(self, root, corruption, severity, transform):
        super().__init__(root, transform=transform)
        self.corruption = corruption
        self.severity = severity

    def __getitem__(self, i):
        from PIL import Image
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.corruption != "clean":
            img = apply_corruption(img, self.corruption, self.severity)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


@torch.no_grad()
def run(model, loader, device, num_classes):
    # per class: count, in_top1, in_top5
    cnt = np.zeros(num_classes); t1 = np.zeros(num_classes); t3 = np.zeros(num_classes); t5 = np.zeros(num_classes)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        top5 = logits.topk(5, dim=1).indices.cpu().numpy()
        tgt = targets.numpy()
        for j, y in enumerate(tgt):
            cnt[y] += 1
            if top5[j, 0] == y:
                t1[y] += 1
            if y in top5[j, :3]:
                t3[y] += 1
            if y in top5[j]:
                t5[y] += 1
    return cnt, t1, t3, t5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", default="dataset/test")
    ap.add_argument("--corruption", required=True)
    ap.add_argument("--severity", type=int, required=True)
    ap.add_argument("--num_classes", type=int, default=156)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model, args.ckpt, args.num_classes, device)
    tfm = make_eval_transform()
    ds = CorruptImageFolder(args.test_dir, args.corruption, args.severity, tfm)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    cnt, t1, t3, t5 = run(model, loader, device, args.num_classes)
    valid = cnt > 0
    r1 = 100.0 * t1[valid] / cnt[valid]   # per-class top-1 recall
    r3 = 100.0 * t3[valid] / cnt[valid]   # per-class top-3 recall
    r5 = 100.0 * t5[valid] / cnt[valid]   # per-class top-5 recall

    # overall
    overall_t1 = 100.0 * t1.sum() / cnt.sum()
    overall_t3 = 100.0 * t3.sum() / cnt.sum()
    overall_t5 = 100.0 * t5.sum() / cnt.sum()

    # "recoverable" classes: confused at top-1 (recall<50) but recoverable (recall>=80)
    confused = r1 < 50
    summary = {
        "model": args.model, "corruption": args.corruption, "severity": args.severity,
        "overall_top1": round(overall_t1, 2), "overall_top3": round(overall_t3, 2), "overall_top5": round(overall_t5, 2),
        "n_classes": int(valid.sum()),
        "n_confused_top1(<50%)": int(confused.sum()),
        "n_recoverable_top3(top1<50 & top3>=80)": int((confused & (r3 >= 80)).sum()),
        "n_recoverable_top5(top1<50 & top5>=80)": int((confused & (r5 >= 80)).sum()),
        "n_lost_top3(top1<50 & top3<50)": int((confused & (r3 < 50)).sum()),
        "cond_recovery_top3(%)": round(float((overall_t3 - overall_t1) / max(1e-9, 100 - overall_t1) * 100), 2),
        "cond_recovery_top5(%)": round(float((overall_t5 - overall_t1) / max(1e-9, 100 - overall_t1) * 100), 2),
    }
    out = {"summary": summary,
           "per_class_top1": r1.round(2).tolist(),
           "per_class_top3": r3.round(2).tolist(),
           "per_class_top5": r5.round(2).tolist()}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
