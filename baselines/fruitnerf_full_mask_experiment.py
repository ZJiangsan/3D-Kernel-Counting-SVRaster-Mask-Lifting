#!/usr/bin/env python3
"""FruitNeRF full-incomplete-mask baseline provenance.

This clean repository does not duplicate the original notebook-style ~1k-line
research driver. The manuscript comparison used a Nerfacto-style semantic field
with binary mask supervision during optimization. The values below are retained
from the supplied research script so the evaluated configuration is explicit.

For the previous InvNeRF-Seg implementation, use the separate public repository
linked from README.md.
"""
from dataclasses import dataclass, asdict
import json

@dataclass(frozen=True)
class FruitNeRFConfig:
    near_plane: float = 0.05
    far_plane: float = 500.0
    num_proposal_samples_0: int = 256
    num_proposal_samples_1: int = 96
    num_nerf_samples_per_ray: int = 48
    num_proposal_iterations: int = 2
    proposal_warmup: int = 5000
    semantic_loss_weight: float = 1.0
    predict_normals: bool = True
    train_rays_per_batch: int = 5120
    eval_rays_per_batch: int = 5120
    epochs_per_stage: int = 2000

if __name__ == "__main__":
    print(json.dumps(asdict(FruitNeRFConfig()), indent=2))
    print("This file documents the evaluated baseline configuration; it is not the proposed method.")
