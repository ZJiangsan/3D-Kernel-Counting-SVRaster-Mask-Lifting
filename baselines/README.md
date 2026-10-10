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

## SA3D

Figure 2 includes a **SA3D post-hoc mask back-projection baseline** (Cen et al., 2023, *Segment Anything in 3D with NeRFs*) as a secondary comparison. In the manuscript experiment, the relevant post-hoc component operates on a pretrained RGB NeRF without retraining the underlying radiance field and transfers 2D masks into 3D using volumetric rendering weights.

The original development script used for this SA3D comparison is not included in the cleaned repository. Accordingly, the repository's reproducibility claim is limited to the **proposed RGB-only SVRaster → post-hoc mask-lifting → topology-aware counting pipeline and the included ablations**; it does not claim end-to-end reproduction of every comparison method shown in the manuscript.

The SA3D values in Figure 2 are retained as reported experimental results. A replacement implementation is intentionally not fabricated after the fact because it might differ from the code used to generate those values.
