#!/usr/bin/env python3
"""
Reads all per-run JSON files and produces:
  results/summary.csv     — mean ± std per (init, metric)
  results/comparison.csv  — side-by-side random_init vs finetune
  Prints a formatted table + t-test p-values for key metrics.

Usage: python scripts/aggregate_results.py
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

MNE_ROOT   = Path(__file__).parent.parent
RESULT_DIR = MNE_ROOT / "results" / "per_run"

ROBUSTNESS_KEYS = (
    ["clean"]
    + [f"gaussian_s{s}" for s in range(1, 6)]
    + ["mean_noisy", "relative_drop_pct"]
)


def load_all() -> list:
    runs = []
    for p in sorted(RESULT_DIR.glob("*.json")):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def aggregate(runs: list, init: str) -> dict:
    subset = [r for r in runs if r["init"] == init]
    out = {"init": init, "n_runs": len(subset)}
    for key in ROBUSTNESS_KEYS:
        vals = [r["robustness"][key] for r in subset if key in r.get("robustness", {})]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"]  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            out[f"{key}_min"]  = float(np.min(vals))
            out[f"{key}_max"]  = float(np.max(vals))
            out[f"{key}_vals"] = vals
    return out


def ttest(agg_a: dict, agg_b: dict, key: str) -> float:
    """Two-sided independent t-test, returns p-value (or nan)."""
    va = agg_a.get(f"{key}_vals", [])
    vb = agg_b.get(f"{key}_vals", [])
    if len(va) < 2 or len(vb) < 2:
        return float("nan")
    _, p = stats.ttest_ind(va, vb, equal_var=False)
    return float(p)


def main():
    runs = load_all()
    if not runs:
        print(f"No JSON files found under {RESULT_DIR}")
        return

    inits = sorted({r["init"] for r in runs})
    aggs  = {init: aggregate(runs, init) for init in inits}

    n_complete = {init: aggs[init]["n_runs"] for init in inits}
    print(f"\nFound {len(runs)} completed runs: {n_complete}")

    # ---- Per-init summary table ---------------------------------------
    print(f"\n{'':22} " + "  ".join(f"{i:<28}" for i in inits))
    print(f"{'Metric':<22} " + "  ".join(f"{'mean':>8} {'±std':>6} {'min':>7} {'max':>7}" for _ in inits))
    print("-" * (22 + 34 * len(inits)))

    for key in ROBUSTNESS_KEYS:
        row = f"{key:<22} "
        for init in inits:
            a = aggs[init]
            m  = a.get(f"{key}_mean", float("nan"))
            s  = a.get(f"{key}_std",  float("nan"))
            lo = a.get(f"{key}_min",  float("nan"))
            hi = a.get(f"{key}_max",  float("nan"))
            row += f"{m:>8.2f} ±{s:>5.2f} {lo:>7.2f} {hi:>7.2f}  "
        print(row)

    # ---- t-test between init strategies (if both present) -------------
    if "random_init" in aggs and "finetune" in aggs:
        print("\n--- Welch t-test: random_init vs finetune ---")
        print(f"{'Metric':<22} {'p-value':>10}  {'significant (p<0.05)':>22}")
        for key in ["clean", "mean_noisy", "relative_drop_pct"]:
            p = ttest(aggs["random_init"], aggs["finetune"], key)
            sig = "yes" if p < 0.05 else "no"
            print(f"  {key:<20} {p:>10.4f}  {sig:>22}")

    # ---- Save summary CSV ---------------------------------------------
    summary_path = MNE_ROOT / "results" / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["init", "metric", "n_runs", "mean", "std", "min", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for init in inits:
            a = aggs[init]
            for key in ROBUSTNESS_KEYS:
                if f"{key}_mean" in a:
                    w.writerow({
                        "init":   init,
                        "metric": key,
                        "n_runs": a["n_runs"],
                        "mean":   round(a[f"{key}_mean"], 4),
                        "std":    round(a[f"{key}_std"],  4),
                        "min":    round(a[f"{key}_min"],  4),
                        "max":    round(a[f"{key}_max"],  4),
                    })
    print(f"\nSaved summary → {summary_path}")

    # ---- Save side-by-side comparison CSV ----------------------------
    comp_path = MNE_ROOT / "results" / "comparison.csv"
    with open(comp_path, "w", newline="") as f:
        cols = ["metric"]
        for init in inits:
            cols += [f"{init}_mean", f"{init}_std"]
        if len(inits) == 2:
            cols.append("p_value")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for key in ROBUSTNESS_KEYS:
            row = {"metric": key}
            for init in inits:
                a = aggs[init]
                row[f"{init}_mean"] = round(a.get(f"{key}_mean", float("nan")), 4)
                row[f"{init}_std"]  = round(a.get(f"{key}_std",  float("nan")), 4)
            if len(inits) == 2:
                row["p_value"] = round(ttest(aggs[inits[0]], aggs[inits[1]], key), 4)
            w.writerow(row)
    print(f"Saved comparison → {comp_path}")


if __name__ == "__main__":
    main()
