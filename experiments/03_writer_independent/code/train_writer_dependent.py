#!/usr/bin/env python3
"""
Train ONE model under a single stratified train:test split of a given ratio.
Mirrors the kfold_train protocol exactly (AdamW lr 3e-4, wd 1e-4, bs 32, 30 ep,
early-stop patience 7, ImageNet-pretrained fine-tuning) but replaces the
StratifiedKFold partition with StratifiedShuffleSplit(test_size = 1 - ratio).

The split seed is FIXED across models, so every architecture trained at a given
ratio shares the exact same train/val/test partition. 10% of the train portion is
held out (stratified) for early-stopping validation.

Outputs: outputs_split/ratio_<RR>/<model>/checkpoints/best.pt
         outputs_split/ratio_<RR>/<model>/metrics.json   (incl. test_indices, clean test acc)
"""
import sys, os, json, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, "training_scripts")
sys.path.insert(0, str(Path(__file__).parent))
import kfold_train as KT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=KT.SUPPORTED_MODELS)
    ap.add_argument("--ratio", type=float, required=True, help="train fraction, e.g. 0.7")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out_dir", default="outputs_split")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--skip_if_done", action="store_true")
    a = ap.parse_args()

    rr = f"{int(round(a.ratio * 100))}"
    base = Path(a.out_dir) / f"ratio_{rr}" / a.model
    best_ckpt = base / "checkpoints" / "best.pt"
    metrics_p = base / "metrics.json"
    if a.skip_if_done and best_ckpt.exists() and metrics_p.exists():
        print(f"[{a.model}|r{rr}] already done -- skip", flush=True)
        return

    KT.set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = [os.path.join(a.data_dir, s) for s in ("train", "val", "test")]
    full = KT.MergedImageFolder(roots, transform=None)
    nc = len(full.classes)
    tg = np.array(full.targets)
    idx = np.arange(len(full))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - a.ratio, random_state=a.seed)
    trainval_idx, test_idx = next(sss.split(idx, tg))
    vss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=a.seed)
    tr_loc, vl_loc = next(vss.split(trainval_idx, tg[trainval_idx]))
    train_idx = trainval_idx[tr_loc]
    val_idx = trainval_idx[vl_loc]
    print(f"[{a.model}|r{rr}] split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"(seed={a.seed})", flush=True)

    train_tfm = KT.build_train_transforms()
    eval_tfm = KT.build_eval_transforms()
    from PIL import Image

    class Sub(Dataset):
        def __init__(s, ix, t): s.ix = ix; s.t = t
        def __len__(s): return len(s.ix)
        def __getitem__(s, i):
            p, l = full.samples[s.ix[i]]
            return s.t(Image.open(p).convert("RGB")), l

    trl = DataLoader(Sub(train_idx, train_tfm), a.batch_size, shuffle=True,
                     num_workers=a.num_workers, pin_memory=True)
    vll = DataLoader(Sub(val_idx, eval_tfm), a.batch_size, shuffle=False,
                     num_workers=a.num_workers, pin_memory=True)
    tel = DataLoader(Sub(test_idx, eval_tfm), a.batch_size, shuffle=False,
                     num_workers=a.num_workers, pin_memory=True)

    model = KT.build_model(a.model, nc).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    es = KT.EarlyStopping(patience=a.patience)
    best = float("inf"); best_acc = 0.0; best_ep = 0
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(a.epochs):
        trm = KT.train_one_epoch(model, trl, crit, opt, device)
        vlm = KT.evaluate(model, vll, crit, device)
        print(f"[{a.model}|r{rr}] ep{ep+1}/{a.epochs} train_acc={trm[1]:.4f} "
              f"val_loss={vlm[0]:.4f} val_acc={vlm[1]:.4f}", flush=True)
        if vlm[0] < best:
            best = vlm[0]; best_acc = vlm[1]; best_ep = ep
            torch.save({"model": model.state_dict(), "num_classes": nc,
                        "class_to_idx": full.class_to_idx, "ratio": a.ratio,
                        "seed": a.seed}, best_ckpt)
        es(vlm[0])
        if es.should_stop:
            print(f"[{a.model}|r{rr}] early stop at ep{ep+1}", flush=True)
            break

    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    te = KT.evaluate(model, tel, crit, device)
    metrics = {"model": a.model, "ratio": a.ratio, "seed": a.seed, "num_classes": nc,
               "best_epoch": best_ep, "best_val_loss": best, "best_val_acc": best_acc,
               "test_loss": te[0], "test_acc": te[1],
               "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
               "n_test": int(len(test_idx)), "elapsed_sec": time.time() - t0,
               "test_indices": [int(x) for x in test_idx]}
    metrics_p.write_text(json.dumps(metrics))
    print(f"[{a.model}|r{rr}] DONE test_acc={te[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
