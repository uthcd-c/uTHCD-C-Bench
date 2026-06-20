#!/bin/bash
# Download pretrained checkpoints and unpack them at the repository root so the
# evaluation scripts find every model's weights.
#
#   bash scripts/fetch_checkpoints.sh            # all experiments
#   bash scripts/fetch_checkpoints.sh 03         # only writer-independent (2.4 GB)
#   bash scripts/fetch_checkpoints.sh 01 02      # a subset
#
# Two hosting sources are supported (set SOURCE below):
#   * "drive"  : anonymous Google Drive mirror, for the double-blind review period
#                (upload from an ANONYMOUS Google account; fill the per-archive file ids)
#   * "zenodo" : the citable archival record, for the camera-ready (fill ZENODO_RECORD)
set -eu
cd "$(dirname "$0")/.."                                   # repo root

# ---- EDIT THIS --------------------------------------------------------------
SOURCE="zenodo"                 # "drive" during anonymous review, "zenodo" for final

ZENODO_RECORD=""                # numeric record id -> https://zenodo.org/records/<id>

declare -A DRIVE_ID=(           # Google Drive file id of each archive (anonymous account)
  [01]="" [02]="" [03]="" [04]="" )
# -----------------------------------------------------------------------------

declare -A ARCHIVE=(
  [01]="uTHCD-C-ckpts-01-cross_validation.tar"
  [02]="uTHCD-C-ckpts-02-split_ratio.tar"
  [03]="uTHCD-C-ckpts-03-writer_independent.tar"
  [04]="uTHCD-C-ckpts-04-collapse_mechanism.tar"
)
SELECT=("$@"); [ ${#SELECT[@]} -gt 0 ] || SELECT=(01 02 03 04)

fetch() {  # $1 = experiment id, $2 = output filename
  local id="$1" f="$2"
  case "$SOURCE" in
    zenodo)
      [ -n "$ZENODO_RECORD" ] || { echo "Set ZENODO_RECORD." >&2; exit 1; }
      curl -L --fail "https://zenodo.org/records/${ZENODO_RECORD}/files/${f}?download=1" -o "$f" ;;
    drive)
      local gid="${DRIVE_ID[$id]:-}"
      [ -n "$gid" ] || { echo "Set DRIVE_ID[$id]." >&2; exit 1; }
      command -v gdown >/dev/null || pip install gdown
      gdown --id "$gid" -O "$f" ;;          # gdown handles the large-file confirm token
    *) echo "SOURCE must be 'drive' or 'zenodo'." >&2; exit 1 ;;
  esac
}

for id in "${SELECT[@]}"; do
  f="${ARCHIVE[$id]:-}"
  [ -n "$f" ] || { echo "unknown experiment id: $id (use 01..04)"; continue; }
  echo ">> [$SOURCE] $f"
  fetch "$id" "$f"
  tar -xf "$f"                                            # recreates outputs_*/... at the repo root
  rm -f "$f"
done

# place the collapse-experiment weights next to their scripts
if [ -d mnasnet_robustness/checkpoints ]; then
  mkdir -p experiments/04_collapse_mechanism
  cp -rl mnasnet_robustness/checkpoints experiments/04_collapse_mechanism/ 2>/dev/null \
    || cp -r mnasnet_robustness/checkpoints experiments/04_collapse_mechanism/
fi

echo "Done. Verify with the shipped SHA256SUMS-<exp>.txt, then evaluate, e.g.:"
echo "  PYTHONPATH=src python3 experiments/01_corruption_benchmark_cv/code/aggregate_kfold.py"
