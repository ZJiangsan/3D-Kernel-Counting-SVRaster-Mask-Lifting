# SVRaster configuration

The supplied research scripts load `config.yaml` from each trained SVRaster model directory. The actual experiment `config.yaml` was **not included in the supplied files**, so no synthetic replacement is provided here.

Before public release, copy one representative RGB-training configuration used for the manuscript into this directory, for example:

```text
configs/cob_rgb_svraster.yaml
```

The proposed method requires RGB-trained geometry. Kernel masks are applied only during post-hoc mask lifting.
