# SVRaster configuration

`svraster_rgb_reconstruction.yaml` contains the minimal overrides for the **final RGB-only SVRaster reconstruction** used in the proposed pipeline.

The final workflow trains SVRaster from the RGB `images` directory for 20,000 iterations with `bound_scale = 0.85` and `res_downscale = 1.0`. Kernel masks are not used to train or fine-tune the SVRaster representation. After RGB reconstruction is complete, the masks are introduced only by the post-hoc mask-lifting script.

The configuration file is intentionally provided as a **minimal override** rather than a synthetic full configuration. Apply it on top of the official NVlabs/SVRaster defaults corresponding to the SVRaster version used for the experiment.

Before training, change:

```yaml
source_path: /path/to/maize_ear_dataset
```

to the directory containing one maize-ear dataset with the RGB `images` folder and the associated COLMAP/SVRaster inputs.

Trained SVRaster checkpoints are not distributed in this repository because of their large size. Users should train the RGB geometry themselves and then run the provided post-hoc mask-lifting and counting scripts.
