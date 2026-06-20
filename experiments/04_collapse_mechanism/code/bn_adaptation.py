#!/usr/bin/env python3
"""
bn_adaptation.py
----------------
Verifies that the corruption-induced collapse of the BatchNorm-based compact models
(MnasNet0_5, ShuffleNetV2-0.5) is caused by BatchNorm covariate shift, not by
training instability or architectural incapacity (reviewer #2, comment 7).

For a given (corruption, severity) it reports, per model, the Top-1 accuracy under:
  1. clean inputs;
  2. corrupted inputs with the STORED (clean-data) BatchNorm running statistics
     (model.eval())  -- the collapse;
  3. corrupted inputs with BatchNorm re-estimated from the corrupted BATCH
     (test-time BN adaptation, no retraining; Schneider et al., NeurIPS 2020)
     -- recovery, if the cause is covariate shift.

A large jump from (2) to (3) shows the learned features are intact and only the
normalisation statistics are mismatched.

Usage:
  python eval_scripts/bn_adaptation.py --corruption gaussian_noise --severity 3 \
      --models mnasnet0_5 shufflenet_v2_x0_5 --fold 0
"""
import sys, json, argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import eval_kfold_corruption as E
from run_eval_A import DetFold          # deterministic per-sample version-A corruption

KFOLD = "outputs_kfold"; DATA = "dataset"; NC = 156


def set_bn_use_batch_stats(model):
    """Keep dropout etc. in eval mode, but let BatchNorm use the current batch's
    statistics instead of the stored running estimates."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
    return model


@torch.no_grad()
def top1(model, ds, device, bs, nw):
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    correct = total = 0
    for x, y in ld:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        o = model(x)
        if isinstance(o, tuple):
            o = o[0]
        correct += (o.argmax(1) == y).sum().item(); total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corruption", default="gaussian_noise")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["mnasnet0_5", "shufflenet_v2_x0_5"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=12)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = json.load(open(f"{KFOLD}/vgg16_bn/fold_{a.fold}/metrics.json"))["test_indices"]
    all_samples, _ = E.build_merged_samples(DATA)
    tfm = E.make_eval_transform()
    ds_clean = DetFold(all_samples, idx, "clean", 1, tfm)
    ds_corr = DetFold(all_samples, idx, a.corruption, a.severity, tfm)

    print(f"Corruption: {a.corruption} severity {a.severity} (fold {a.fold})")
    print(f"{'model':<20}{'clean':>8}{'stored-BN':>11}{'adapted-BN':>12}")
    out = {"corruption": a.corruption, "severity": a.severity, "fold": a.fold, "rows": {}}
    for mname in a.models:
        ckpt = f"{KFOLD}/{mname}/fold_{a.fold}/checkpoints/best.pt"
        m = E.load_model(mname, ckpt, NC, device); m.eval()
        a_clean = top1(m, ds_clean, device, a.batch_size, a.num_workers)
        a_stored = top1(m, ds_corr, device, a.batch_size, a.num_workers)
        set_bn_use_batch_stats(m)
        a_adapt = top1(m, ds_corr, device, a.batch_size, a.num_workers)
        print(f"{mname:<20}{a_clean:>8.1f}{a_stored:>11.1f}{a_adapt:>12.1f}")
        out["rows"][mname] = {"clean": round(a_clean, 2), "stored_bn": round(a_stored, 2),
                              "adapted_bn": round(a_adapt, 2)}
        del m; torch.cuda.empty_cache()
    Path(f"{KFOLD}/bn_adaptation_{a.corruption}_s{a.severity}.json").write_text(json.dumps(out, indent=2))
    print(f"saved {KFOLD}/bn_adaptation_{a.corruption}_s{a.severity}.json")


if __name__ == "__main__":
    main()
