#!/bin/bash
# Version-A corruption evaluation for all 9 models across all 5 CV folds.
# Run from the repository root with PYTHONPATH=src and dataset/ in place.
set -eu
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODELS="vgg16_bn googlenet swin_t efficientnet_b0 squeezenet1_0 convnext_tiny regnet_x_400mf shufflenet_v2_x0_5 mnasnet0_5"
for f in 0 1 2 3 4; do for m in $MODELS; do
  python3 experiments/01_corruption_benchmark_cv/code/run_eval_A.py \
      --model "$m" --fold "$f" --batch_size 256 --num_workers 16 --skip_if_done
done; done
