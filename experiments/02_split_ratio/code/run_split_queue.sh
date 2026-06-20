#!/bin/bash
# Dynamic job-queue worker for the split-ratio experiment. Multiple workers (one
# per GPU, possibly on different machines pointing at their own out_dir) pull jobs
# from a shared jobs file via atomic mkdir locks, so heterogeneous model costs
# auto-balance. Resumable: train_split.py --skip_if_done guards completed work,
# and per-job .done markers survive restarts.
#
# Usage (run from repo root):  bash eval_scripts/run_split_queue.sh <GPU> <out_dir> <jobs_file>
#   jobs file: one "ratio:model" per line, e.g.  0.5:vgg16_bn
set -u
GPU=$1; OUT=$2; JOBS=$3
export CUDA_VISIBLE_DEVICES=$GPU
LOCK="$OUT/.locks"; mkdir -p "$LOCK"
echo "[gpu$GPU] worker start $(date +%F_%T)  out=$OUT  jobs=$JOBS"
while true; do
  claimed_any=0
  while read -r job; do
    [ -z "$job" ] && continue
    tag=$(echo "$job" | tr ':/.' '_')
    [ -f "$LOCK/$tag.done" ] && continue
    if mkdir "$LOCK/$tag.run" 2>/dev/null; then
      claimed_any=1
      r=${job%%:*}; m=${job##*:}
      echo "[gpu$GPU] CLAIM $job $(date +%T)"
      if python3 training_scripts/train_split.py --model "$m" --ratio "$r" \
            --out_dir "$OUT" --skip_if_done --num_workers 8; then
        touch "$LOCK/$tag.done"; echo "[gpu$GPU] DONE $job $(date +%T)"
      else
        echo "[gpu$GPU] FAIL $job $(date +%T)"; rmdir "$LOCK/$tag.run" 2>/dev/null
      fi
    fi
  done < "$JOBS"
  [ "$claimed_any" -eq 0 ] && break
done
echo "[gpu$GPU] queue drained $(date +%T)"
