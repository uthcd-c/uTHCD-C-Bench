# 07 — Practical model-selection guidelines

Synthesises the benchmark into a decision procedure for choosing an architecture
given a deployment's accuracy, robustness, and efficiency constraints.
*Paper Section 4.8.*

This experiment is **derivative** — it consumes the results of experiments
01–05 rather than running new training. The flowchart is reproduced in
`results/figures/model_selection_flowchart.png`.

## The logic (summarised)

1. **Need maximum robustness?** → LayerNorm-based ConvNeXt-Tiny or Swin-Tiny
   (mCE@top-1 ≈ 1.0), accepting lower throughput.
2. **Need a balance of robustness and accuracy at moderate cost?** → VGG16_BN
   (baseline) or GoogLeNet / EfficientNet-B0 (within ~25% of baseline mCE).
3. **Need maximum efficiency?** → the compact BatchNorm models (ShuffleNetV2,
   MnasNet) are fastest but collapse under corruption; deploy them **only** with
   the mitigations from experiment 04 (BatchNorm-momentum fix at training time, or
   test-time BN-statistic adaptation), or restrict them to clean-input settings.

The supporting numbers come from `results/tables/` (mCE, clean accuracy,
throughput) and the Pareto frontiers in `results/figures/pareto_*.png`.

The flowchart image itself is a diagram summarising this logic; regenerate or edit
it from the per-model numbers in `results/tables/` if you change the underlying
results.
