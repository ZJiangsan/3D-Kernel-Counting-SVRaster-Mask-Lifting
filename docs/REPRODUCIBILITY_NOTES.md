# Reproducibility notes before public release

This repository package was reconstructed from the manuscript-related Python scripts supplied on 9 September 2026. The goal was to preserve scientific logic while removing duplicated notebook-style code, local-path assumptions, exploratory plots, and non-final experimental branches from the **core** pipeline.

## Confirmed reference outputs

The final supplied counting script records:

- Ground truth: `[716, 337, 729, 314, 539, 463, 659, 496, 724]`
- Proposed pipeline: `[716, 340, 732, 317, 533, 461, 654, 492, 725]`
- MAE: `3.0`
- Proportional masks without optimized/adaptive splitting: `[684, 335, 721, 295, 520, 442, 643, 488, 717]`
- MAE: `14.666666...`

The supplied unscaled-mask SVRaster script records a no-split MAE of `80.222222...`. This repository reports that value directly as `80.22` when rounded to two decimals.

## U-Net protocol

The supplied U-Net script:

- uses a ResNet-34 encoder with ImageNet initialization;
- creates 60 augmented variants per annotated image;
- randomly splits the augmented image pool 80/20;
- initializes the present training run from a previously trained U-Net checkpoint.

The cleaned training CLI therefore exposes checkpoint continuation explicitly through `--resume` instead of silently loading a local file.

The train/validation split after augmentation is preserved because it reflects the source experiment. It should not be presented as an independent segmentation generalization test.

## Final SVRaster geometry protocol

The proposed method uses a **single RGB-only SVRaster reconstruction** followed by post-hoc mask lifting.

The actual research script confirms that the RGB reconstruction:

- reads RGB images from `image_dir_name = "images"`;
- trains for `20,000` iterations;
- uses `bound_scale = 0.85`;
- uses `res_downscale = 1.0`.

Kernel masks are not used to train or fine-tune the final SVRaster representation. A minimal reconstruction override is therefore provided as `configs/svraster_rgb_reconstruction.yaml`.

After RGB reconstruction, `scripts/04_svraster_mask_lifting.py` loads the trained RGB checkpoint, renders depth/transmittance, extracts a TSDF surface, and then projects the 2D kernel masks onto those fixed surface points. This post-hoc lifting step is the only point at which kernel masks interact with the final SVRaster geometry.

## External SVRaster dependency

The mask-lifting code imports internal modules from the official NVlabs/SVRaster codebase. The official repository should be installed separately rather than copied into this repository.

Trained SVRaster model checkpoints are intentionally not distributed because of their very large size. Users should train the RGB geometry themselves using the official SVRaster implementation together with `configs/svraster_rgb_reconstruction.yaml`, then apply the provided mask-lifting and counting scripts.

## Scope of the public reproducibility release

This repository is intended to reproduce the **proposed RGB-only SVRaster reconstruction, post-hoc mask-lifting, and topology-aware counting pipeline**, together with the included ablations and reference prediction tables.

The manuscript also reports SD3 in Figure 2 as a secondary comparison. The original SD3 development script used for that experiment is not included in this cleaned repository. Therefore, the public release does **not** claim end-to-end reproduction of every comparison method in Figure 2. The reported SD3 values are preserved as manuscript results rather than replaced by a newly written implementation that might not match the code used in the experiment.

## Release artifacts not bundled here

The repository does not currently include:

1. the RGB image dataset;
2. trained U-Net weights;
3. trained SVRaster checkpoints (intentionally omitted because of size);
4. the original SD3 comparison script.

The minimal RGB-only SVRaster reconstruction configuration is included. The manuscript Data Availability statement distinguishes between the released proposed-method code/configuration and large model/data artifacts that are not distributed in this repository.
