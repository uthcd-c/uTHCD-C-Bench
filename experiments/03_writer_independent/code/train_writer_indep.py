#!/usr/bin/env python3
"""
Train ONE model under a single WRITER-INDEPENDENT 70:30 split.

Identical protocol to train_split.py / kfold_train (ImageNet fine-tune, AdamW
lr 3e-4, wd 1e-4, bs 32, 30 ep, early-stop patience 7) EXCEPT the partition is
writer-disjoint: the train/val/test sets share no writer, so the test set
measures generalisation to UNSEEN handwriting styles (Reviewer #2: "writer-
independent splits").

Writer identity is parsed from the uTHCD filename convention <writer>_<class>.bmp
(e.g. 0292_000.bmp -> writer 0292, 192s_000.bmp -> writer 192s). The trailing
's' (online-capture set) is stripped so <NNN> and <NNNs> are treated as the SAME
person -> a writer can never leak across the split.

For a given --seed the GroupShuffleSplit random_state is FIXED across models, so
all 9 architectures at that seed share the exact same writer partition. Three
seeds (1/42/123) give three independent writer partitions -> mean +/- std over
both the held-out-writer set AND model init.

Outputs: outputs_writer/seed_<S>/<model>/checkpoints/best.pt
         outputs_writer/seed_<S>/<model>/metrics.json
           (incl. test_indices, train/test/val writer counts, writer_overlap=0)
"""
import sys, os, re, json, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "training_scripts")
sys.path.insert(0, str(Path(__file__).parent))
import kfold_train as KT

_WPAT = re.compile(r'^(.+)_(\d+)\.bmp$', re.IGNORECASE)


def writer_of(path):
    """Writer id from <writer>_<class>.bmp; strip trailing 's' so NNN==NNNs."""
    b = os.path.basename(path)
    m = _WPAT.match(b)
    w = m.group(1).lower() if m else b.lower()
    return w[:-1] if w.endswith("s") else w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=KT.SUPPORTED_MODELS)
    ap.add_argument("--seed", type=int, required=True,
                    help="defines the writer-disjoint split (GroupShuffleSplit random_state)")
    ap.add_argument("--init_seed", type=int, default=None,
                    help="model-init/training RNG seed; defaults to --seed. Set "
                         "differently ONLY to restart a diverged run while keeping "
                         "the exact same writer partition.")
    ap.add_argument("--test_frac", type=float, default=0.30, help="held-out test fraction")
    ap.add_argument("--data_dir", default="dataset")
    ap.add_argument("--out_dir", default="outputs_writer")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--skip_if_done", action="store_true")
    a = ap.parse_args()

    base = Path(a.out_dir) / f"seed_{a.seed}" / a.model
    best_ckpt = base / "checkpoints" / "best.pt"
    metrics_p = base / "metrics.json"
    if a.skip_if_done and best_ckpt.exists() and metrics_p.exists():
        print(f"[{a.model}|s{a.seed}] already done -- skip", flush=True)
        return

    init_seed = a.seed if a.init_seed is None else a.init_seed
    KT.set_seed(init_seed)   # split is fixed by --seed below (sklearn random_state), so
                             # init_seed only changes weight init / data order / augmentation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = [os.path.join(a.data_dir, s) for s in ("train", "val", "test")]
    full = KT.MergedImageFolder(roots, transform=None)
    nc = len(full.classes)
    tg = np.array(full.targets)
    idx = np.arange(len(full))
    writers = np.array([writer_of(full.samples[i][0]) for i in range(len(full))])

    # writer-disjoint 70:30, then a writer-disjoint 10% of train for early stop
    gss = GroupShuffleSplit(n_splits=1, test_size=a.test_frac, random_state=a.seed)
    trainval_idx, test_idx = next(gss.split(idx, tg, groups=writers))
    vss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=a.seed)
    tr_loc, vl_loc = next(vss.split(trainval_idx, tg[trainval_idx], groups=writers[trainval_idx]))
    train_idx = trainval_idx[tr_loc]
    val_idx = trainval_idx[vl_loc]

    w_tr, w_vl, w_te = set(writers[train_idx]), set(writers[val_idx]), set(writers[test_idx])
    overlap = len((w_tr | w_vl) & w_te)
    assert overlap == 0, f"WRITER LEAK: {overlap} test writers seen in train/val"
    print(f"[{a.model}|s{a.seed}] writer-indep split: "
          f"train={len(train_idx)}({len(w_tr)}w) val={len(val_idx)}({len(w_vl)}w) "
          f"test={len(test_idx)}({len(w_te)}w) overlap={overlap} "
          f"classes_test={len(set(tg[test_idx]))}/{nc}", flush=True)

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
        print(f"[{a.model}|s{a.seed}] ep{ep+1}/{a.epochs} train_acc={trm[1]:.4f} "
              f"val_loss={vlm[0]:.4f} val_acc={vlm[1]:.4f}", flush=True)
        if vlm[0] < best:
            best = vlm[0]; best_acc = vlm[1]; best_ep = ep
            torch.save({"model": model.state_dict(), "num_classes": nc,
                        "class_to_idx": full.class_to_idx, "seed": a.seed,
                        "init_seed": init_seed, "test_frac": a.test_frac}, best_ckpt)
        es(vlm[0])
        if es.should_stop:
            print(f"[{a.model}|s{a.seed}] early stop at ep{ep+1}", flush=True)
            break

    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck["model"]); model.eval()
    te = KT.evaluate(model, tel, crit, device)
    metrics = {"model": a.model, "seed": a.seed, "init_seed": init_seed,
               "test_frac": a.test_frac,
               "protocol": "writer_independent", "num_classes": nc,
               "best_epoch": best_ep, "best_val_loss": best, "best_val_acc": best_acc,
               "test_loss": te[0], "test_acc": te[1],
               "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
               "n_test": int(len(test_idx)),
               "n_writers_train": len(w_tr), "n_writers_val": len(w_vl),
               "n_writers_test": len(w_te), "writer_overlap": overlap,
               "elapsed_sec": time.time() - t0,
               "test_indices": [int(x) for x in test_idx]}
    metrics_p.write_text(json.dumps(metrics))
    print(f"[{a.model}|s{a.seed}] DONE test_acc={te[1]:.4f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
