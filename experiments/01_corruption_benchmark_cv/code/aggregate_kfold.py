#!/usr/bin/env python3
"""
aggregate_kfold.py
------------------
Collects per-fold corruption results for all 9 models and produces:

  1. kfold_clean_accuracy.csv     — mean ± std of clean top-1 across folds
  2. kfold_mce_summary.csv        — mean ± std of mCE@top-1, mCE_avg across folds
  3. kfold_ranking_stability.csv  — Kendall's tau of mCE rankings across fold pairs
  4. kfold_aggregate.json         — full structured results

Run after all 9 models × 5 folds have been evaluated:
  python aggregate_kfold.py --kfold_dir outputs_kfold --out_dir outputs_kfold
"""

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from scipy.stats import kendalltau
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not found — Kendall tau will be skipped. Install with: pip install scipy")

MODELS = [
    "vgg16_bn", "googlenet", "efficientnet_b0", "regnet_x_400mf",
    "mnasnet0_5", "shufflenet_v2_x0_5", "squeezenet1_0",
    "convnext_tiny", "swin_t",
]
N_SPLITS = 5


def load_fold_results(kfold_dir: Path, model: str, fold: int) -> Dict:
    path = kfold_dir / model / f"fold_{fold}" / "corruption_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_fold_metrics(kfold_dir: Path, model: str, fold: int) -> Dict:
    path = kfold_dir / model / f"fold_{fold}" / "metrics.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def write_csv(path: Path, header: List[str], rows: List[List]):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"Saved -> {path}")


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate k-fold results for uTHCD-C")
    parser.add_argument("--kfold_dir", type=str, default="outputs_kfold")
    parser.add_argument("--out_dir",   type=str, default="outputs_kfold")
    parser.add_argument("--n_splits",  type=int, default=N_SPLITS)
    args = parser.parse_args()

    kfold_dir = Path(args.kfold_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = list(range(args.n_splits))

    # ------------------------------------------------------------------
    # Collect per-model, per-fold results
    # ------------------------------------------------------------------
    # clean_acc[model][fold] = top-1 clean accuracy (%)
    # mce_top1[model][fold]  = mCE@top-1
    # mce_avg[model][fold]   = mCE_avg
    clean_acc: Dict[str, List] = {m: [] for m in MODELS}
    mce_top1:  Dict[str, List] = {m: [] for m in MODELS}
    mce_avg:   Dict[str, List] = {m: [] for m in MODELS}

    aggregate = {}

    missing_models = []
    for model in MODELS:
        aggregate[model] = {}
        for fold in folds:
            cr = load_fold_results(kfold_dir, model, fold)
            fm = load_fold_metrics(kfold_dir, model, fold)

            if not cr or not fm:
                print(f"  MISSING: {model} / fold {fold}")
                missing_models.append(f"{model}/fold_{fold}")
                continue

            fold_clean = cr.get("clean", {}).get("top1", None)
            fold_mce   = cr.get("mce",   {})

            if fold_clean is not None:
                clean_acc[model].append(fold_clean)
            if fold_mce.get("top1") is not None:
                mce_top1[model].append(fold_mce["top1"])
            if fold_mce.get("avg") is not None:
                mce_avg[model].append(fold_mce["avg"])

            aggregate[model][f"fold_{fold}"] = {
                "clean_top1":  fold_clean,
                "test_acc_train": fm.get("test_acc"),
                "mce_top1":    fold_mce.get("top1"),
                "mce_top3":    fold_mce.get("top3"),
                "mce_top5":    fold_mce.get("top5"),
                "mce_avg":     fold_mce.get("avg"),
                "best_epoch":  fm.get("best_epoch"),
                "elapsed_sec": fm.get("elapsed_sec"),
            }

    if missing_models:
        print(f"\nWARNING: {len(missing_models)} fold(s) missing results.")
        print("These will be excluded from summary statistics.\n")

    # ------------------------------------------------------------------
    # Table 1: Clean accuracy — mean ± std (replaces Table 4 header rows)
    # ------------------------------------------------------------------
    clean_rows = []
    for model in MODELS:
        vals = clean_acc[model]
        if not vals:
            clean_rows.append([model, "N/A", "N/A", len(folds) - len(vals)])
        else:
            clean_rows.append([
                model,
                f"{np.mean(vals):.2f}",
                f"{np.std(vals):.2f}",
                len(vals),
            ])

    write_csv(
        out_dir / "kfold_clean_accuracy.csv",
        ["model", "clean_top1_mean", "clean_top1_std", "n_folds"],
        clean_rows,
    )

    # ------------------------------------------------------------------
    # Table 2: mCE summary — mean ± std (replaces Table 5)
    # ------------------------------------------------------------------
    mce_rows = []
    for model in MODELS:
        t1 = mce_top1[model]
        av = mce_avg[model]
        if not t1:
            mce_rows.append([model, "N/A", "N/A", "N/A", "N/A", 0])
        else:
            mce_rows.append([
                model,
                f"{np.mean(t1):.4f}",
                f"{np.std(t1):.4f}",
                f"{np.mean(av):.4f}" if av else "N/A",
                f"{np.std(av):.4f}"  if av else "N/A",
                len(t1),
            ])

    write_csv(
        out_dir / "kfold_mce_summary.csv",
        ["model", "mce_top1_mean", "mce_top1_std",
         "mce_avg_mean", "mce_avg_std", "n_folds"],
        mce_rows,
    )

    # ------------------------------------------------------------------
    # Table 3: Ranking stability — Kendall's tau between fold pairs
    # ------------------------------------------------------------------
    if HAS_SCIPY:
        tau_rows = []
        for fold_a, fold_b in combinations(folds, 2):
            ranks_a = []
            ranks_b = []
            for model in MODELS:
                fa = aggregate[model].get(f"fold_{fold_a}", {}).get("mce_top1")
                fb = aggregate[model].get(f"fold_{fold_b}", {}).get("mce_top1")
                if fa is not None and fb is not None:
                    ranks_a.append(fa)
                    ranks_b.append(fb)

            if len(ranks_a) >= 2:
                tau, pval = kendalltau(ranks_a, ranks_b)
                tau_rows.append([f"fold_{fold_a}", f"fold_{fold_b}",
                                  f"{tau:.4f}", f"{pval:.4f}", len(ranks_a)])

        write_csv(
            out_dir / "kfold_ranking_stability.csv",
            ["fold_a", "fold_b", "kendall_tau", "p_value", "n_models"],
            tau_rows,
        )

        if tau_rows:
            all_taus = [float(r[2]) for r in tau_rows]
            print(f"\nKendall tau (mCE@top-1 rankings) across fold pairs:")
            print(f"  mean={np.mean(all_taus):.4f}  min={np.min(all_taus):.4f}  "
                  f"max={np.max(all_taus):.4f}")
            if np.mean(all_taus) > 0.7:
                print("  → Rankings are highly stable across folds (tau > 0.7).")
            elif np.mean(all_taus) > 0.5:
                print("  → Rankings are moderately stable across folds.")
            else:
                print("  → Rankings show variability across folds — investigate.")

    # ------------------------------------------------------------------
    # Print summary to console (mirrors Tables 4 and 5 in the paper)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLEAN ACCURACY: mean ± std across 5 folds")
    print("=" * 70)
    print(f"{'Model':<25} {'Top-1 Mean (%)':>15} {'Std':>8} {'Folds':>6}")
    print("-" * 70)
    for row in clean_rows:
        print(f"{row[0]:<25} {row[1]:>15} {row[2]:>8} {row[3]:>6}")

    print("\n" + "=" * 70)
    print("MCE SUMMARY: mean ± std across 5 folds (lower is better)")
    print("=" * 70)
    print(f"{'Model':<25} {'mCE@top1 Mean':>15} {'Std':>8} {'mCE_avg':>10} {'Std':>8}")
    print("-" * 70)
    for row in mce_rows:
        print(f"{row[0]:<25} {row[1]:>15} {row[2]:>8} {row[3]:>10} {row[4]:>8}")

    # ------------------------------------------------------------------
    # Save full aggregate JSON
    # ------------------------------------------------------------------
    agg_out = out_dir / "kfold_aggregate.json"
    with open(agg_out, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nFull aggregate data -> {agg_out}")


if __name__ == "__main__":
    main()
