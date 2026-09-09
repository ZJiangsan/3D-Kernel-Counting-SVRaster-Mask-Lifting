# 3D Kernel Counting with SVRaster Mask Lifting

Research code for **Geometry-First Proportional Mask Lifting and Topology-Aware Clustering Improve 3D Maize Kernel Counting**.

The repository implements the new pipeline developed as a methodological follow-up to the authors' previous InvNeRF-Seg maize kernel-counting study:

**RGB views → U-Net incomplete surface masks → proportional dot masks → RGB-only SVRaster reconstruction → post-hoc mask lifting → topology-aware clustering → kernel count**

The complete new pipeline produced the nine-ear predictions stored in `results/proposed_predictions.csv` (MAE = 3.00). The corresponding proportional-mask/no-adaptive-split ablation is stored in `results/proportional_no_split_predictions.csv` (MAE = 14.67).

## Repository layout

```text
.
├── scripts/
│   ├── 01_train_unet.py
│   ├── 02_infer_unet_masks.py
│   ├── 03_generate_proportional_masks.py
│   ├── 04_svraster_mask_lifting.py
│   ├── 05_count_kernels.py
│   ├── 06_evaluate_counts.py
│   └── 07_alpha_ablation_counts.py
├── src/kernel3d/
│   ├── masks.py
│   ├── unet.py
│   ├── clustering.py
│   └── metrics.py
├── baselines/
├── configs/
│   └── svraster_rgb_stage1.yaml
├── data/
├── weights/
├── results/
└── docs/
```

## 1. Installation

The core post-processing code is ordinary Python, but geometry reconstruction depends on the official SVRaster implementation.

```bash
# Create a Python environment of your choice, then:
pip install -r requirements.txt

git clone https://github.com/NVlabs/svraster.git external/svraster
# Follow the official SVRaster installation instructions, including its CUDA extension.
```

The scripts were derived from the actual research scripts supplied for the manuscript. Hard-coded local paths, duplicated experimental blocks, and exploratory plotting code were removed from the core pipeline.

## 2. Data layout

A minimal example is:

```text
data/
└── cob1/
    ├── images/          # RGB frames used by COLMAP/SVRaster
    ├── masks/           # U-Net incomplete surface masks
    └── masks_alpha06/   # proportional masks, alpha = 0.6
```

## 3. U-Net mask generation

Four manually annotated RGB images were used in the present study. The fourth image was added as a conservative control against poor 2D mask generation being the downstream bottleneck; its effect was not evaluated separately.

Train/fine-tune the U-Net:

```bash
python scripts/01_train_unet.py \
  --images data/annotations/images \
  --labels data/annotations/labels \
  --output weights/kernel_unet.pth
```

The original development script loaded a previous U-Net checkpoint before continuing training. To reproduce that continuation explicitly, use:

```bash
python scripts/01_train_unet.py \
  --images data/annotations/images \
  --labels data/annotations/labels \
  --resume weights/previous_3image_unet.pth \
  --output weights/kernel_unet.pth
```

Generate incomplete masks for all views:

```bash
python scripts/02_infer_unet_masks.py \
  --images data/cob1/images \
  --checkpoint weights/kernel_unet.pth \
  --output data/cob1/masks
```

## 4. Proportional mask transformation

The manuscript uses distance-transform maxima to preserve kernel centers and scales each local radius by `alpha`.

```bash
python scripts/03_generate_proportional_masks.py \
  --masks data/cob1/masks \
  --output data/cob1/masks_alpha06 \
  --alpha 0.6
```

The alpha sweep evaluated `0.2, 0.4, 0.6, 0.8, 1.0`.

## 5. SVRaster reconstruction and mask lifting

The final method uses **RGB-only SVRaster geometry**. The research script trained Stage 1 from the RGB `images` directory for **20,000 iterations**. Kernel masks did not optimize the final geometry; they were applied later through post-hoc mask lifting.

A verified minimal Stage-1 override is provided in:

```text
configs/svraster_rgb_stage1.yaml
```

Apply that file on top of the official NVlabs/SVRaster defaults for the corresponding SVRaster version, edit `source_path`, and train the RGB geometry.

The research code also contained a later experimental stage that fine-tuned the already-trained representation using mask images. That branch was abandoned and is **not part of the final proposed pipeline**.

Trained SVRaster checkpoints are not included because they are very large. Users are expected to train the RGB SVRaster geometry themselves from the provided settings and the official SVRaster implementation.

Then lift the masks to the RGB-trained reconstructed surface:

```bash
python scripts/04_svraster_mask_lifting.py \
  --svraster-root external/svraster \
  --model-path /path/to/cob1_rgb_svraster_model \
  --mask-dir-name masks_alpha06 \
  --iteration 20000 \
  --output outputs/cob1_alpha06_lifted.ply
```

`--mask-dir-name` is interpreted relative to the source dataset stored in the SVRaster model's `config.yaml`, matching the original experiment.

## 6. Kernel clustering and counting

```bash
python scripts/05_count_kernels.py \
  --input outputs/cob1_alpha06_lifted.ply \
  --output-dir outputs/counts
```

For the no-adaptive-split ablation:

```bash
python scripts/05_count_kernels.py \
  --input outputs/cob1_alpha06_lifted.ply \
  --no-adaptive-split \
  --output-dir outputs/counts_no_split
```

The final clustering implementation preserves the experimental parameters for PCA-based head/center/tail partitioning, local-density bridge removal, region-specific DBSCAN, and adaptive DBSCAN/K-means subdivision of oversized clusters.

## 7. Evaluation

`results/proposed_predictions.csv` contains the final reported nine-ear counts.

```bash
python scripts/06_evaluate_counts.py --csv results/proposed_predictions.csv
```

Expected MAE: **3.00 kernels**.

The no-adaptive-split result in `results/proportional_no_split_predictions.csv` has expected MAE **14.67 kernels**.

## 8. Alpha ablation

After generating and lifting point clouds for each alpha, arrange them as:

```text
outputs/alpha_sweep/
├── alpha_0.2/*.ply
├── alpha_0.4/*.ply
├── alpha_0.6/*.ply
├── alpha_0.8/*.ply
└── alpha_1.0/*.ply
```

Then run:

```bash
python scripts/07_alpha_ablation_counts.py \
  --root outputs/alpha_sweep \
  --ground-truth results/ground_truth_counts.csv \
  --output results/alpha_ablation_results.csv
```

## Baselines and previous study

The previous InvNeRF-Seg maize-kernel pipeline is the main historical baseline for this study and is maintained separately:

https://github.com/ZJiangsan/Automated-3D-Kernel-Instance-Segmentation-and-Counting

The new repository does not duplicate that entire codebase. `baselines/` contains concise provenance/reference scripts for the additional full-mask SVRaster and FruitNeRF comparisons; the notebook-style exploratory drivers were intentionally excluded from the clean public pipeline. See `baselines/README.md`.

## Reproducibility status

The code necessary for the **core proposed method** is organized here, including a verified minimal RGB-only SVRaster Stage-1 configuration. The repository intentionally does not distribute the very large trained SVRaster checkpoints. The RGB image dataset, trained U-Net weights, and SD3 comparison implementation are also not currently bundled. See `docs/REPRODUCIBILITY_NOTES.md` for details.

## Citation

A provisional citation is provided in `CITATION.cff`. Update it when the manuscript receives a DOI or final bibliographic information.
