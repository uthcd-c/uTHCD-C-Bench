#!/usr/bin/env python3
"""
run_kfold_parallel.py
---------------------
Parallel k-fold launcher. Supports one or more GPUs — each GPU is managed
independently with its own memory budget. Jobs are dispatched to whichever
GPU has the most free memory for the next queued job.

Memory budget per model (MiB, conservative at batch_size=32):
  vgg16_bn           10000
  convnext_tiny       4500
  swin_t              4500
  efficientnet_b0     3000
  regnet_x_400mf      3000
  googlenet           3000
  mnasnet0_5          2000
  shufflenet_v2_x0_5  2000
  squeezenet1_0       2000

GPU headroom reserved: 2000 MiB per GPU.

VGG16_BN is scheduled first (needed as mCE baseline during eval).

Usage — single GPU (default):
  python run_kfold_parallel.py --data_dir dataset/

Usage — two independent GPUs on L40:
  python run_kfold_parallel.py --data_dir dataset/ --gpus 0 1 \
      --models efficientnet_b0 regnet_x_400mf googlenet

Usage — A100 (3 models, GPU 0 only):
  python run_kfold_parallel.py --data_dir dataset/ --gpus 0 \
      --models vgg16_bn convnext_tiny swin_t
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

GPU_HEADROOM_MIB = 2000   # reserved per GPU
POLL_INTERVAL    = 30     # seconds between scheduling loops

MODEL_MEM: Dict[str, int] = {
    "vgg16_bn":            10000,
    "convnext_tiny":        4500,
    "swin_t":               4500,
    "efficientnet_b0":      3000,
    "regnet_x_400mf":       3000,
    "googlenet":            3000,
    "mnasnet0_5":           2000,
    "shufflenet_v2_x0_5":  2000,
    "squeezenet1_0":        2000,
}

N_SPLITS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def checkpoint_exists(kfold_dir: str, model: str, fold: int) -> bool:
    return (Path(kfold_dir) / model / f"fold_{fold}" / "checkpoints" / "best.pt").exists()


def gpu_total_mib(gpu_ids: List[int]) -> Dict[int, int]:
    """Return total memory (MiB) for each requested GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        all_total = [int(x) for x in out.strip().split("\n")]
        return {g: all_total[g] for g in gpu_ids}
    except Exception:
        return {g: 24000 for g in gpu_ids}


def gpu_used_mib(gpu_ids: List[int]) -> Dict[int, int]:
    """Return currently used memory (MiB) for each requested GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        all_used = [int(x) for x in out.strip().split("\n")]
        return {g: all_used[g] for g in gpu_ids}
    except Exception:
        return {g: 0 for g in gpu_ids}


def build_job_list(kfold_dir: str, models: List[str]) -> List[Tuple[str, int]]:
    """
    Jobs ordered: vgg16_bn first (if in models), then others smallest-first.
    Skips already-completed checkpoints.
    """
    jobs = []
    if "vgg16_bn" in models:
        for fold in range(N_SPLITS):
            if not checkpoint_exists(kfold_dir, "vgg16_bn", fold):
                jobs.append(("vgg16_bn", fold))

    other = sorted([m for m in models if m != "vgg16_bn"], key=lambda m: MODEL_MEM[m])
    for model in other:
        for fold in range(N_SPLITS):
            if not checkpoint_exists(kfold_dir, model, fold):
                jobs.append((model, fold))
    return jobs


# ---------------------------------------------------------------------------
# Scheduling loop
# ---------------------------------------------------------------------------
def run_training(args, gpu_ids: List[int], gpu_budgets: Dict[int, int]):
    """
    Dispatch (model, fold) jobs across gpu_ids.
    Uses actual nvidia-smi readings before every launch decision so that
    jobs already running from a previous launcher session are accounted for.
    A small post-launch settle delay lets CUDA allocate memory before we
    query again, preventing over-scheduling.
    """
    SETTLE_SECS = 20   # wait after launching before querying nvidia-smi again

    # running[pid] = (model, fold, gpu_id, Popen, log_file)
    running: Dict[int, tuple] = {}

    job_queue = build_job_list(args.kfold_dir, args.models)
    total_jobs = len(job_queue)
    skipped    = sum(
        1 for m in args.models for f in range(N_SPLITS)
        if checkpoint_exists(args.kfold_dir, m, f)
    )

    print(f"\nJobs already done : {skipped}")
    print(f"Jobs to run       : {total_jobs}")
    print()

    while job_queue or running:
        # ---- Reap finished processes ----
        for pid in list(running):
            model, fold, gpu_id, proc, log_file = running[pid]
            if proc.poll() is not None:
                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                log_file.close()
                print(f"  [done]   GPU{gpu_id} | {model:25s} fold {fold}  {status}")
                del running[pid]

        # ---- Read ACTUAL GPU memory from nvidia-smi ----
        actual_used = gpu_used_mib(gpu_ids)

        # ---- Try to launch one job per loop iteration ----
        launched = False
        for i, (model, fold) in enumerate(job_queue):
            needed = MODEL_MEM[model]
            # Find GPU with most actual free space that fits this job
            best_gpu  = None
            best_free = -1
            for g in gpu_ids:
                free = gpu_budgets[g] - actual_used[g]
                if free >= needed and free > best_free:
                    best_free = free
                    best_gpu  = g

            if best_gpu is None:
                continue  # no GPU has room right now; try next job

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

            log_path = Path(args.kfold_dir) / model / f"fold_{fold}" / "train.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "w")

            cmd = [
                sys.executable, "training_scripts/kfold_train.py",
                "--model",       model,
                "--fold",        str(fold),
                "--data_dir",    args.data_dir,
                "--out_dir",     args.kfold_dir,
                "--batch_size",  str(args.batch_size),
                "--epochs",      str(args.epochs),
                "--num_workers", str(args.workers),
                "--skip_if_done",
            ]
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env)
            running[proc.pid] = (model, fold, best_gpu, proc, log_file)
            job_queue.pop(i)

            print(f"  [launch] GPU{best_gpu} | {model:25s} fold {fold}  "
                  f"({needed} MiB needed, {best_free} MiB was free)")
            launched = True

            # Let CUDA allocate before we read nvidia-smi again
            time.sleep(SETTLE_SECS)
            break  # re-read actual GPU state before next launch

        if not launched:
            if running:
                time.sleep(POLL_INTERVAL)
            elif job_queue:
                print("WARNING: remaining jobs do not fit any GPU. Waiting...")
                time.sleep(POLL_INTERVAL)

    print("\nAll training jobs complete.\n")


# ---------------------------------------------------------------------------
# Eval phase
# ---------------------------------------------------------------------------
def run_eval_phase(args, gpu_ids: List[int]):
    """Run corruption eval sequentially, alternating GPUs."""
    gpu_cycle = 0
    for model in args.models:
        for fold in range(N_SPLITS):
            result_path = Path(args.kfold_dir) / model / f"fold_{fold}" / "corruption_results.json"
            if result_path.exists():
                continue
            if not checkpoint_exists(args.kfold_dir, model, fold):
                print(f"  [eval] Skipping {model}/fold_{fold} — no checkpoint.")
                continue

            gpu_id = gpu_ids[gpu_cycle % len(gpu_ids)]
            env    = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            cmd = [
                sys.executable, "eval_scripts/eval_kfold_corruption.py",
                "--model",       model,
                "--fold",        str(fold),
                "--kfold_dir",   args.kfold_dir,
                "--data_dir",    args.data_dir,
                "--batch_size",  "128",
                "--num_workers", str(args.workers),
                "--skip_if_done",
            ]
            print(f"  [eval] GPU{gpu_id} | {model} fold {fold}")
            subprocess.run(cmd, env=env, check=False)
            gpu_cycle += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU parallel k-fold launcher")
    parser.add_argument("--data_dir",    type=str, required=True)
    parser.add_argument("--kfold_dir",   type=str, default="outputs_kfold")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--gpus",        type=int, nargs="+", default=[0],
                        help="GPU IDs to use, e.g. --gpus 0 1")
    parser.add_argument("--models",      nargs="+", default=list(MODEL_MEM.keys()),
                        choices=list(MODEL_MEM.keys()),
                        help="Subset of models to train on this machine")
    args = parser.parse_args()

    os.makedirs(args.kfold_dir, exist_ok=True)

    # Query actual GPU sizes and compute per-GPU budget
    totals  = gpu_total_mib(args.gpus)
    budgets = {g: totals[g] - GPU_HEADROOM_MIB for g in args.gpus}

    print("=" * 65)
    print("  uTHCD-C Multi-GPU Parallel K-Fold Launcher")
    print(f"  GPUs    : {args.gpus}")
    for g in args.gpus:
        print(f"    GPU{g}  total={totals[g]} MiB   budget={budgets[g]} MiB")
    print(f"  Models  : {args.models}")
    print(f"  Data    : {args.data_dir}")
    print(f"  Output  : {args.kfold_dir}")
    print("=" * 65)

    # Phase 1: Training
    run_training(args, args.gpus, budgets)

    # Phase 2: Corruption evaluation
    print("--- Phase 2: Corruption Evaluation ---")
    run_eval_phase(args, args.gpus)

    # Phase 3: Aggregation
    print("\n--- Phase 3: Aggregation ---")
    subprocess.run([
        sys.executable, "eval_scripts/aggregate_kfold.py",
        "--kfold_dir", args.kfold_dir,
        "--out_dir",   args.kfold_dir,
    ], check=True)

    print("\nDone. Results in", args.kfold_dir)


if __name__ == "__main__":
    main()
