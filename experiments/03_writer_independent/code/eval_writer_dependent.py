#!/usr/bin/env python3
"""
Version-A corruption evaluation for ONE split ratio. Mirrors run_fold_A.py: loads
all 9 models trained at this ratio, applies each corrupted set once (deterministic
per sample), forwards every model on the same batch, and writes per-model
corruption_results_A.json with clean / corrupted (top-k error) / mCE.

Reads the test partition from outputs_split/ratio_<RR>/vgg16_bn/metrics.json
(every model at a ratio shares the same split, so any model's test_indices work).
"""
import sys, json, argparse, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import eval_kfold_corruption as E
from run_eval_A import DetFold

DATA = "dataset"; NC = 156
CORR = E.CORRUPTIONS; SEV = E.SEVERITIES
MODELS = ["vgg16_bn", "googlenet", "swin_t", "efficientnet_b0", "squeezenet1_0",
          "convnext_tiny", "regnet_x_400mf", "shufflenet_v2_x0_5", "mnasnet0_5"]


@torch.no_grad()
def pass_all(models, ds, device, bs, nw):
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    correct = {m: {1: 0, 3: 0, 5: 0} for m in models}
    total = 0
    for images, targets in ld:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        for m, mod in models.items():
            logits = mod(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            for k in (1, 3, 5):
                correct[m][k] += (logits.topk(k, 1).indices == targets.unsqueeze(1)).any(1).sum().item()
        total += targets.size(0)
    return {m: {k: 100.0 * correct[m][k] / total for k in (1, 3, 5)} for m in models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--out_dir", default="outputs_split")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--skip_if_done", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rr = f"{int(round(a.ratio * 100))}"
    rdir = Path(a.out_dir) / f"ratio_{rr}"
    outs = {m: rdir / m / "corruption_results_A.json" for m in MODELS}
    if a.skip_if_done and all(p.exists() for p in outs.values()):
        print(f"ratio {rr} already evaluated -- skip", flush=True); return

    miss = [m for m in MODELS if not (rdir / m / "checkpoints" / "best.pt").exists()]
    if miss:
        print(f"ratio {rr}: missing checkpoints for {miss} -- train first", flush=True); return

    t0 = time.time()
    test_idx = json.load(open(rdir / "vgg16_bn" / "metrics.json"))["test_indices"]
    all_samples, _ = E.build_merged_samples(DATA)
    tfm = E.make_eval_transform()
    models = {m: E.load_model(m, str(rdir / m / "checkpoints" / "best.pt"), NC, device) for m in MODELS}
    print(f"[r{rr}] loaded {len(models)} models, {len(test_idx)} test imgs ({time.time()-t0:.0f}s)", flush=True)

    clean = pass_all(models, DetFold(all_samples, test_idx, "clean", 1, tfm), device, a.batch_size, a.num_workers)
    err = {m: {c: {} for c in CORR} for m in MODELS}
    for c in CORR:
        for s in SEV:
            acc = pass_all(models, DetFold(all_samples, test_idx, c, s, tfm), device, a.batch_size, a.num_workers)
            for m in MODELS:
                err[m][c][s] = {f"top{k}": 100 - acc[m][k] for k in (1, 3, 5)}
            print(f"[r{rr}] {c:16s} s{s} vgg_top1={acc['vgg16_bn'][1]:.2f}", flush=True)

    vgg = err["vgg16_bn"]
    for m in MODELS:
        mce = {}
        for k in ["top1", "top3", "top5"]:
            ces = [sum((err[m][c][s][k] / vgg[c][s][k]) if vgg[c][s][k] > 0 else 1.0 for s in SEV) / len(SEV)
                   for c in CORR]
            mce[k] = sum(ces) / len(CORR)
        mce["avg"] = (mce["top1"] + mce["top3"] + mce["top5"]) / 3.0
        res = {"model": m, "ratio": a.ratio, "version": "A",
               "clean": {f"top{k}": clean[m][k] for k in (1, 3, 5)},
               "corrupted": {c: {str(s): err[m][c][s] for s in SEV} for c in CORR},
               "mce": mce}
        outs[m].write_text(json.dumps(res, indent=2))
    print(f"[r{rr}] DONE 9 models ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
