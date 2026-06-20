#!/bin/bash
# Expose the flat writer-independent checkpoints/ in the nested layout the
# evaluation scripts expect (outputs_writer/seed_<S>/<model>/checkpoints/best.pt
# and outputs_split/ratio_70/<model>/checkpoints/best.pt), via symlinks.
# Run from the repository root:  bash scripts/link_checkpoints.sh
set -eu
cd "$(dirname "$0")/.."                        # repo root
MODELS="vgg16_bn googlenet swin_t efficientnet_b0 squeezenet1_0 convnext_tiny regnet_x_400mf shufflenet_v2_x0_5 mnasnet0_5"
for s in 1 42 123; do for m in $MODELS; do
  d="outputs_writer/seed_$s/$m/checkpoints"; mkdir -p "$d"
  ln -sf "$(pwd)/checkpoints/wi_seed${s}_${m}.pt" "$d/best.pt"
done; done
for m in $MODELS; do
  d="outputs_split/ratio_70/$m/checkpoints"; mkdir -p "$d"
  ln -sf "$(pwd)/checkpoints/wd_70_${m}.pt" "$d/best.pt"
done
echo "Linked 36 writer-independent checkpoints into the nested layout."
