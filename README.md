# 3D Kernel Counting with SVRaster Mask Lifting

Research code for **Geometry-First Mask Lifting and Topology-Aware Instance Separation for 3D Maize Kernel Counting**.

The repository implements the final pipeline developed as a methodological follow-up to the authors' previous InvNeRF-Seg maize kernel-counting study.

The proposed SVRaster workflow contains **one RGB reconstruction step only**. Kernel masks do not train or fine-tune SVRaster. The two information streams meet only after the RGB geometry has been reconstructed:

```text
Multi-view RGB images ──> RGB-only SVRaster reconstruction ──> fixed 3D surface ──┐
                                                                                  ├─> post-hoc mask lifting
Multi-view RGB images ──> U-Net surface masks ──> proportional masks ─────────────┘
                                                                                          ↓
                                                                            topology-aware clustering
                                                                                          ↓
                                                                                     kernel count
```

The complete pipeline produced the nine-ear predictions stored in `results/proposed_predictions.csv` (MAE = 3.00). The corresponding proportional-mask/no-adaptive-split ablation is stored in `results/proportional_no_split_predictions.csv` (MAE = 14.67).

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
│   └── svraster_rgb_reconstruction.yaml
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

Four manually annotated RGB images from different ears and complementary viewing angles were used in the present study. Only the visible outward-facing kernel surfaces were annotated.

Train/fine-tune the U-Net:

```bash
python scripts/01_train_unet.py \
  --images data/annotations/images \
  --labels data/annotations/labels \
  --output weights/kernel_unet.pth
```

The original development script initialized the present training run from a previous U-Net checkpoint. To reproduce continuation from an earlier checkpoint explicitly, use:

```bash
python scripts/01_train_unet.py \
  --images data/annotations/images \
  --labels data/annotations/labels \
  --resume weights/previous_unet.pth \
  --output weights/kernel_unet.pth
```

Generate incomplete surface masks for all views:

```bash
python scripts/02_infer_unet_masks.py \
  --images data/cob1/images \
  --checkpoint weights/kernel_unet.pth \
  --output data/cob1/masks
```

## 4. Proportional mask transformation

The manuscript uses a distance-transform-based proportional reduction to retain compact central portions of the predicted visible surface regions and improve separation before 3D lifting.

```bash
python scripts/03_generate_proportional_masks.py \
  --masks data/cob1/masks \
  --output data/cob1/masks_alpha06 \
  --alpha 0.6
```

The alpha sweep evaluated `0.2, 0.4, 0.6, 0.8, 1.0`.

## 5. RGB-only SVRaster reconstruction and post-hoc mask lifting

The final method first trains **SVRaster from RGB images only for 20,000 iterations**. Kernel masks are not used during SVRaster training and there is no mask-based SVRaster fine-tuning stage in the proposed workflow.

A minimal reconstruction override is provided in:

```text
configs/svraster_rgb_reconstruction.yaml
```

Apply that file on top of the official NVlabs/SVRaster defaults for the corresponding SVRaster version, edit `source_path`, and train the RGB geometry. Trained SVRaster checkpoints are not included because they are very large; users are expected to train the RGB geometry themselves.

After RGB reconstruction is complete, lift the proportional masks onto the fixed RGB-trained surface:

```bash
python scripts/04_svraster_mask_lifting.py \
  --svraster-root external/svraster \
  --model-path /path/to/cob1_rgb_svraster_model \
  --mask-dir-name masks_alpha06 \
  --iteration 20000 \
  --output outputs/cob1_alpha06_lifted.ply
```

The lifting script uses the trained RGB model for rendered depth/transmittance and TSDF surface extraction, then projects the 2D masks onto those fixed surface points. `--mask-dir-name` is interpreted relative to the source dataset stored in the SVRaster model's `config.yaml`.

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

The new repository does not duplicate that entire codebase. `baselines/` contains concise provenance/reference scripts for the additional unscaled-mask SVRaster and FruitNeRF comparisons; the notebook-style exploratory drivers were intentionally excluded from the clean public pipeline. See `baselines/README.md`.

## Reproducibility status

The code necessary for the **core proposed method** is organized here, including the minimal RGB-only SVRaster reconstruction configuration. The repository intentionally does not distribute the very large trained SVRaster checkpoints. The RGB image dataset and trained U-Net weights are also not currently bundled.

Figure 2 includes the **SA3D post-hoc mask back-projection baseline** as a secondary comparison. The original development script used for that SA3D comparison is not part of this cleaned release. The repository therefore claims reproducibility for the proposed SVRaster pipeline and included ablations, **not end-to-end reproducibility of every comparison method shown in the manuscript**. See `baselines/README.md` and `docs/REPRODUCIBILITY_NOTES.md` for details.

## Citation

A provisional citation is provided in `CITATION.cff`. Update it when the manuscript receives a DOI or final bibliographic information.
