#!/usr/bin/env python3
"""
Parallel launcher for all 30 MNASNet robustness runs.
Runs up to --max-parallel jobs concurrently on the same GPU.

Steps:
  1. Generate fold_indices.json (once, via prepare_folds.py)
  2. Launch all 30 (init × seed × fold) training+eval jobs, 4 at a time
  3. Aggregate results via aggregate_results.py

Usage:
  python run_all.py                      # 4 parallel (default)
  python run_all.py --max-parallel 2     # fewer concurrent jobs
  python run_all.py --dry-run            # print commands only

Job matrix:
  init  : random_init, finetune          (2)
  seeds : 1, 42, 123                     (3)
  folds : 0, 1, 2, 3, 4                 (5)
  Total : 30 runs
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MNE_ROOT   = Path(__file__).parent
FOLD_FILE  = MNE_ROOT / "fold_indices.json"
SCRIPTS    = MNE_ROOT / "scripts"

# A100 node handles finetune only.
# random_init jobs run on the L40 remote node via run_remote.py.
INITS  = ["finetune"]
SEEDS  = [1]          # seeds 42 and 123 run on L40 via run_remote_finetune.py
FOLDS  = list(range(5))


# ---------------------------------------------------------------------------
# Step 1 — fold index generation
# ---------------------------------------------------------------------------
def prepare_folds():
    if FOLD_FILE.exists():
        print(f"[folds] fold_indices.json already exists — skipping.")
        return
    print("[folds] Generating stratified 5-fold indices …")
    subprocess.run(
        [sys.executable, str(SCRIPTS / "prepare_folds.py")],
        cwd=str(MNE_ROOT), check=True,
    )
    print("[folds] Done.")


# ---------------------------------------------------------------------------
# Step 2 — individual job runner (called inside worker process)
# ---------------------------------------------------------------------------
def run_job(args_tuple):
    init, seed, fold, dry_run = args_tuple
    label    = f"{init}_seed{seed}_fold{fold}"
    cmd      = [sys.executable, str(SCRIPTS / "train_fold.py"),
                "--init", init, "--seed", str(seed), "--fold", str(fold)]
    log_path = MNE_ROOT / "logs" / f"{label}.log"
    log_path.parent.mkdir(exist_ok=True)

    if dry_run:
        print(f"[dry-run] {' '.join(cmd)}")
        return label, 0, "0.0min"

    t0 = time.time()
    with open(log_path, "a") as lf:
        proc = subprocess.run(
            cmd, cwd=str(MNE_ROOT),
            stdout=lf, stderr=subprocess.STDOUT,
        )
    elapsed = (time.time() - t0) / 60.0
    return label, proc.returncode, f"{elapsed:.1f}min"


# ---------------------------------------------------------------------------
# Step 3 — aggregate (after all jobs finish)
# ---------------------------------------------------------------------------
def aggregate():
    print("\n[aggregate] Building summary tables …")
    subprocess.run(
        [sys.executable, str(SCRIPTS / "aggregate_results.py")],
        cwd=str(MNE_ROOT), check=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Launch all 30 MNASNet robustness runs.")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Max concurrent training jobs (default 4, fits A100 40 GB)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job commands without executing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip a job if its result JSON already exists")
    args = parser.parse_args()

    prepare_folds()

    # Build full job list
    all_jobs = [
        (init, seed, fold)
        for init  in INITS
        for seed  in SEEDS
        for fold  in FOLDS
    ]

    # Optionally skip already-completed runs
    if args.skip_existing:
        pending = []
        for init, seed, fold in all_jobs:
            name   = f"{init}_seed{seed}_fold{fold}"
            result = MNE_ROOT / "results" / "per_run" / f"{name}.json"
            if result.exists():
                print(f"[skip] {name} — result exists")
            else:
                pending.append((init, seed, fold))
        all_jobs = pending

    total = len(all_jobs)
    if total == 0:
        print("Nothing to run.")
    else:
        print(f"\n{'='*60}")
        print(f"Launching {total} jobs  |  max-parallel={args.max_parallel}")
        print(f"Matrix: {INITS} × seeds {SEEDS} × folds {FOLDS}")
        print(f"(random_init runs separately on L40 via run_remote.py)")
        print(f"Logs → {MNE_ROOT / 'logs'}/")
        print(f"{'='*60}\n")

    job_args = [(init, seed, fold, args.dry_run) for init, seed, fold in all_jobs]

    completed = 0
    failed    = []
    t_global  = time.time()

    with ProcessPoolExecutor(max_workers=args.max_parallel) as pool:
        future_to_job = {pool.submit(run_job, ja): ja for ja in job_args}
        for fut in as_completed(future_to_job):
            label, rc, elapsed = fut.result()
            completed += 1
            status    = "✓" if rc == 0 else "✗  FAILED"
            # ETA estimate
            elapsed_total = time.time() - t_global
            avg_per_job   = elapsed_total / completed
            eta_min       = avg_per_job * (total - completed) / 60.0
            print(
                f"[{completed:02d}/{total}] {status}  {label:<40} "
                f"({elapsed})  ETA ~{eta_min:.0f} min"
            )
            if rc != 0:
                failed.append(label)

    wall = (time.time() - t_global) / 3600.0
    print(f"\nAll {total} jobs done in {wall:.2f} h")
    if failed:
        print(f"  {len(failed)} FAILED: {failed}")
        print(f"  Check logs under {MNE_ROOT / 'logs'}/")
    else:
        print("  All runs completed successfully.")

    if not args.dry_run and completed > 0:
        aggregate()


if __name__ == "__main__":
    main()
