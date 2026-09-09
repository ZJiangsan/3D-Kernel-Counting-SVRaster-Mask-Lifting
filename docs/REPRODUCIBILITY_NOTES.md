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

## External SVRaster dependency

The mask-lifting code imports internal modules from the official NVlabs/SVRaster codebase. The official repository should be installed separately rather than copied into this repository.

## Missing release artifacts

The following were not present in the supplied materials:

1. the RGB image dataset;
2. U-Net trained weights;
3. the actual SVRaster training `config.yaml`;
4. the SD3 comparison script.

The manuscript Data Availability statement should not claim that these files are already present until they are actually uploaded.
