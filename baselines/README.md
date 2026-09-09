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

No SD3 source script was found in the files supplied for this repository build or in the searched file library. Therefore an SD3 implementation is **not invented here**. Add the original SD3 experiment script before claiming that every manuscript comparison is fully reproducible from this repository.
