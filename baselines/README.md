# Baseline experiments

This folder is intentionally separated from the clean core pipeline.

## Previous InvNeRF-Seg kernel-counting study

The previous study is the principal methodological baseline and has its own repository:

https://github.com/ZJiangsan/Automated-3D-Kernel-Instance-Segmentation-and-Counting

The complete previous repository should be referenced rather than duplicated here.

## `svraster_unscaled_mask_experiment.py`

This is a compact reference evaluator for the supplied unscaled/full incomplete-mask SVRaster ablation. The original notebook-style development script mixed several historical branches, so the clean release keeps the exact recorded per-ear predictions in `../results/unscaled_no_split_predictions.csv`. The supplied source records no-split MAE **80.2222**.

## `fruitnerf_full_mask_experiment.py`

This is a compact provenance/configuration record for the supplied FruitNeRF baseline. The large notebook-style training driver is not part of the proposed method and was intentionally not copied into the clean public pipeline.

## SD3

SD3 appears in Figure 2 as a **secondary comparison**. The original development script used for that comparison is not included in this cleaned repository. Accordingly, the repository's reproducibility claim is limited to the **proposed RGB-only SVRaster → post-hoc mask-lifting → topology-aware counting pipeline and the included ablations**; it does not claim end-to-end reproduction of every comparison method shown in the manuscript.

The SD3 values shown in Figure 2 are retained as reported experimental results from the manuscript study. No replacement SD3 implementation is fabricated here because doing so could differ from the code used to generate those reported values.
