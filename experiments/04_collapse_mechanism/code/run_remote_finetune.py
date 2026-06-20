#!/usr/bin/env python3
"""
Launcher for the L40 remote node — finetune only, seeds 42 and 123.
Runs 10 jobs (2 seeds × 5 folds), 4 concurrent (2 per GPU).

Usage (on the remote node, from ~/uTHCD-C/mnasnet_robustness/):
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate thcr
  python run_remote_finetune.py
  python run_remote_finetune.py --skip-existing   # safe resume
  python run_remote_finetune.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MNE_ROOT = Path(__file__).parent
SCRIPTS  = MNE_ROOT / "scripts"
PYTHON   = sys.executable

INIT   = "finetune"
SEEDS  = [42, 123]
FOLDS  = list(range(5))
GPUS   = [0, 1]
MAX_PER_GPU = 2


def run_job(args_tuple):
    seed, fold, gpu_id, dry_run = args_tuple
    label    = f"{INIT}_seed{seed}_fold{fold}"
    log_path = MNE_ROOT / "logs" / f"{label}.log"
    log_path.parent.mkdir(exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [PYTHON, str(SCRIPTS / "train_fold.py"),
           "--init", INIT, "--seed", str(seed), "--fold", str(fold)]

    if dry_run:
        print(f"[dry-run] GPU={gpu_id}  {' '.join(cmd)}")
        return label, 0, "0.0min"

    t0 = time.time()
    with open(log_path, "a") as lf:
        proc = subprocess.run(cmd, cwd=str(MNE_ROOT),
                              stdout=lf, stderr=subprocess.STDOUT, env=env)
    elapsed = (time.time() - t0) / 60.0
    return label, proc.returncode, f"{elapsed:.1f}min"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    all_jobs = []
    for i, (seed, fold) in enumerate(
        (s, f) for s in SEEDS for f in FOLDS
    ):
        gpu_id = GPUS[i % len(GPUS)]
        all_jobs.append((seed, fold, gpu_id, args.dry_run))

    if args.skip_existing:
        pending = []
        for seed, fold, gpu_id, dr in all_jobs:
            name   = f"{INIT}_seed{seed}_fold{fold}"
            result = MNE_ROOT / "results" / "per_run" / f"{name}.json"
            if result.exists():
                print(f"[skip] {name}")
            else:
                pending.append((seed, fold, gpu_id, dr))
        all_jobs = pending

    total       = len(all_jobs)
    max_workers = len(GPUS) * MAX_PER_GPU   # 4

    print(f"\n{'='*60}")
    print(f"Remote node (L40 ×2) | {INIT} | seeds {SEEDS} | {total} jobs | {max_workers} concurrent")
    print(f"GPU assignment: round-robin across {GPUS}")
    print(f"Logs → {MNE_ROOT / 'logs'}/")
    print(f"{'='*60}\n")

    completed = 0
    failed    = []
    t_global  = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(run_job, j): j for j in all_jobs}
        for fut in as_completed(future_to_job):
            label, rc, elapsed = fut.result()
            completed += 1
            status = "✓" if rc == 0 else "✗  FAILED"
            eta = (time.time() - t_global) / completed * (total - completed) / 60
            print(f"[{completed:02d}/{total}] {status}  {label:<42}  ({elapsed})  ETA ~{eta:.0f}min")
            if rc != 0:
                failed.append(label)

    wall = (time.time() - t_global) / 3600
    print(f"\nDone in {wall:.2f} h  |  failed: {len(failed)}")
    if failed:
        print(f"  {failed}")

    if not args.dry_run and completed > 0:
        print("\n[aggregate] local results only —")
        subprocess.run([PYTHON, str(SCRIPTS / "aggregate_results.py")],
                       cwd=str(MNE_ROOT))


if __name__ == "__main__":
    main()
