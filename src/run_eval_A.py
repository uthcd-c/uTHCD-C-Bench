#!/usr/bin/env python3
"""
Regenerate corruption results with VERSION A corruptions (corruptions_A.py).

Faithful to eval_kfold_corruption.py's metric/formula, but:
  - applies version A corruptions deterministically per global sample index
    (so every model AND the VGG baseline see identical corrupted images);
  - caches the VGG-baseline errors per fold (_vggA_errors_fold{F}.json) so the
    baseline is not recomputed for all 9 models;
  - writes outputs_kfold/<model>/fold_<F>/corruption_results_A.json
    (version-B files are left untouched).
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import eval_kfold_corruption as E
from corruptions_A import apply_corruption

SEED_BASE = 20000
KFOLD = "outputs_kfold"; DATA = "dataset"; NUM_CLASSES = 156
CORR = E.CORRUPTIONS; SEV = E.SEVERITIES


class DetFold(Dataset):
    def __init__(self, all_samples, indices, corruption, severity, tfm):
        self.s = [all_samples[i] for i in indices]; self.g = list(indices)
        self.c = corruption; self.sev = severity; self.t = tfm
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        p, l = self.s[i]; img = Image.open(p).convert("RGB")
        if self.c != "clean":
            np.random.seed(SEED_BASE + int(self.g[i]))
            img = apply_corruption(img, self.c, self.sev)
        if self.t is not None: img = self.t(img)
        return img, l


def eval_all(model, all_samples, idx, tfm, device, bs, nw):
    def run(c, s):
        ds = DetFold(all_samples, idx, c, s, tfm)
        ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
        return E.evaluate_topk(model, ld, device)
    clean = run("clean", 1)
    err = {c: {s: {f"top{k}": 100 - v for k, v in run(c, s).items()} for s in SEV} for c in CORR}
    return clean, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=256); ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--skip_if_done", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_dir = Path(KFOLD) / a.model / f"fold_{a.fold}"
    out = fold_dir / "corruption_results_A.json"
    if a.skip_if_done and out.exists():
        print(f"skip {out}", flush=True); return
    t0 = time.time()
    test_idx = json.load(open(Path(KFOLD) / "vgg16_bn" / f"fold_{a.fold}" / "metrics.json"))["test_indices"]
    all_samples, _ = E.build_merged_samples(DATA)
    tfm = E.make_eval_transform()

    model = E.load_model(a.model, str(fold_dir / "checkpoints" / "best.pt"), NUM_CLASSES, device)
    clean, err = eval_all(model, all_samples, test_idx, tfm, device, a.batch_size, a.num_workers)
    del model; torch.cuda.empty_cache()

    vgg_cache = Path(KFOLD) / f"_vggA_errors_fold{a.fold}.json"
    if a.model == "vgg16_bn":
        vgg_err = err
        vgg_cache.write_text(json.dumps({"err": err, "clean": clean}))
    elif vgg_cache.exists():
        vgg_err = json.load(open(vgg_cache))["err"]
    else:
        vm = E.load_model("vgg16_bn", str(Path(KFOLD) / "vgg16_bn" / f"fold_{a.fold}" / "checkpoints" / "best.pt"),
                          NUM_CLASSES, device)
        _, vgg_err = eval_all(vm, all_samples, test_idx, tfm, device, a.batch_size, a.num_workers)
        del vm; torch.cuda.empty_cache()
        vgg_cache.write_text(json.dumps({"err": vgg_err}))

    def vget(d, c, s, k):
        sub = d[c]; sub = sub[s] if s in sub else sub[str(s)]; return sub[k]
    mce = {}
    for k in ["top1", "top3", "top5"]:
        ces = []
        for c in CORR:
            cs = 0.0
            for s in SEV:
                em = err[c][s][k]; ev = vget(vgg_err, c, s, k)
                cs += (em / ev) if ev > 0 else 1.0
            ces.append(cs / len(SEV))
        mce[k] = sum(ces) / len(CORR)
    mce["avg"] = (mce["top1"] + mce["top3"] + mce["top5"]) / 3.0

    res = {"model": a.model, "fold": a.fold, "version": "A",
           "clean": {f"top{k}": v for k, v in clean.items()},
           "corrupted": {c: {str(s): err[c][s] for s in SEV} for c in CORR},
           "mce": mce}
    out.write_text(json.dumps(res, indent=2))
    print(f"SAVED {out}  mCE@1={mce['top1']:.4f} avg={mce['avg']:.4f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
