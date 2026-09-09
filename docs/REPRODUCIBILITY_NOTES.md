# Reproducibility notes before public release

This repository package was reconstructed from the manuscript-related Python scripts supplied on 9 September 2026. The goal was to preserve scientific logic while removing duplicated notebook-style code, local-path assumptions, exploratory plots, and dead/commented experimental branches from the **core** pipeline.

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
- loads `kernelSeg_best_unet_01.pth` before continuing training.

The cleaned training CLI therefore exposes the previous checkpoint explicitly through `--resume` instead of silently loading a local file.

The train/validation split after augmentation is preserved because it reflects the source experiment. It should not be presented as an independent segmentation generalization test.

## SVRaster geometry protocol

The final proposed method uses the **Stage-1 RGB-only SVRaster geometry**.

The actual research script confirms that Stage 1:

- reads RGB images from `image_dir_name = "images"`;
- uses the ordinary mask directory only as dataset metadata (`mask_dir_name = "masks"`);
- trains for `20,000` iterations;
- uses `bound_scale = 0.85`;
- uses `res_downscale = 1.0`.

The final geometry is not mask-optimized. A verified minimal override file is therefore provided as `configs/svraster_rgb_stage1.yaml`, with `lambda_mask = 0.0` explicitly documented.

The same research script also contains a later experimental branch that switches the image target to `masks_sep4` and continues optimization from the Stage-1 checkpoint. That mask-image fine-tuning branch was abandoned after post-hoc mask lifting on the RGB geometry performed better. It is not part of the final proposed method and is intentionally excluded from the clean core pipeline.

## External SVRaster dependency

The mask-lifting code imports internal modules from the official NVlabs/SVRaster codebase. The official repository should be installed separately rather than copied into this repository.

Trained SVRaster model checkpoints are intentionally not distributed because of their very large size. Users should train the RGB geometry themselves using the official SVRaster implementation together with `configs/svraster_rgb_stage1.yaml`, then apply the provided mask-lifting and counting scripts.

## Release artifacts not bundled here

The repository does not currently include:

1. the RGB image dataset;
2. trained U-Net weights;
3. trained SVRaster checkpoints (intentionally omitted because of size);
4. the SD3 comparison script.

The RGB-only Stage-1 SVRaster configuration is now included. The manuscript Data Availability statement should distinguish between code/configuration that is available and large model/data artifacts that are not distributed in this repository.
