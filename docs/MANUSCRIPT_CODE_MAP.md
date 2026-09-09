# Manuscript-to-code map

| Manuscript component | Repository implementation |
|---|---|
| Four-image U-Net mask generation | `scripts/01_train_unet.py`, `scripts/02_infer_unet_masks.py` |
| Proportional dot-mask transformation | `scripts/03_generate_proportional_masks.py`, `src/kernel3d/masks.py` |
| Alpha values 0.2–1.0 | `scripts/03_generate_proportional_masks.py` + `scripts/07_alpha_ablation_counts.py` |
| RGB-only SVRaster geometry + TSDF surface extraction | official NVlabs/SVRaster + `scripts/04_svraster_mask_lifting.py` |
| Post-hoc 2D-to-3D mask lifting | `scripts/04_svraster_mask_lifting.py` |
| PCA head/center/tail division | `src/kernel3d/clustering.py::split_cob_regions` |
| Local-density bridge removal | `src/kernel3d/clustering.py::process_region` |
| Region-specific DBSCAN | `src/kernel3d/clustering.py::process_region` |
| Adaptive DBSCAN/K-means subdivision | `src/kernel3d/clustering.py::process_region` |
| Final proposed count | `scripts/05_count_kernels.py` |
| No-adaptive-split ablation | `scripts/05_count_kernels.py --no-adaptive-split` |
| Count metrics | `scripts/06_evaluate_counts.py` |
| Unscaled-mask SVRaster reference result | `baselines/svraster_unscaled_mask_experiment.py` + `results/unscaled_no_split_predictions.csv` |
| FruitNeRF comparison configuration provenance | `baselines/fruitnerf_full_mask_experiment.py` |
| Previous InvNeRF-Seg kernel-counting baseline | external previous-study repository |
| SD3 comparison | not available in supplied code; must be added before full-reproduction claim |
