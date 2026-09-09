# SVRaster configuration

`svraster_rgb_stage1.yaml` contains the verified Stage-1 overrides used for the RGB-only SVRaster geometry reconstruction in the final proposed pipeline.

The original research script trained Stage 1 from the RGB `images` directory for 20,000 iterations with `bound_scale = 0.85` and `res_downscale = 1.0`. Kernel masks were not used to optimize the final geometry (`lambda_mask = 0.0`). The later experimental mask-image fine-tuning stage was abandoned and is not part of the final method.

The configuration file is intentionally provided as a **minimal override** rather than a synthetic full configuration. Apply it on top of the official NVlabs/SVRaster defaults corresponding to the SVRaster version used for the experiment.

Before training, change:

```yaml
source_path: /path/to/maize_ear_dataset
```

to the directory containing one maize-ear dataset with the RGB `images` folder and the associated COLMAP/SVRaster inputs.

Trained SVRaster checkpoints are not distributed in this repository because of their large size. Users should train the RGB geometry themselves and then run the provided post-hoc mask-lifting and counting scripts.
