#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm


def import_svraster(svraster_root: Path):
    sys.path.insert(0, str(svraster_root))
    from src.config import cfg, update_config
    from src.dataloader.data_pack import DataPack
    from src.sparse_voxel_model import SparseVoxelModel
    from src.utils.fuser_utils import Fuser
    return cfg, update_config, DataPack, SparseVoxelModel, Fuser


def tsdf_fusion(Fuser, cam_lst, depth_lst, alpha_lst, grid_pts_xyz, trunc_dist, crop_border, alpha_thres):
    fuser = Fuser(
        xyz=grid_pts_xyz,
        bandwidth=trunc_dist,
        use_trunc=True,
        fuse_tsdf=True,
        feat_dim=0,
        alpha_thres=alpha_thres,
        crop_border=crop_border,
        normal_weight=False,
        depth_weight=False,
        border_weight=False,
        use_half=False,
    )
    for cam, frame_depth, frame_alpha in zip(cam_lst, depth_lst, alpha_lst):
        fuser.integrate(cam, frame_depth.cuda(), alpha=frame_alpha.cuda())
    return fuser.tsdf.squeeze(1).contiguous()


def extract_and_paint(
    voxel_model,
    data_pack,
    SparseVoxelModel,
    Fuser,
    init_level: int,
    final_level: int,
    bbox_scale: float,
    bandwidth_vox: float,
    crop_border: float,
    alpha_thres: float,
    save_gpu: bool,
):
    """TSDF-extract the RGB-trained SVRaster geometry and lift 2D masks to surface points.

    This is a cleaned form of the manuscript experiment script. 2D masks are
    loaded through DataPack and written into the point colors; they are not used
    to optimize SVRaster geometry.
    """
    cam_all = data_pack.get_train_cameras()
    chunk_size = 20
    num_chunks = math.ceil(len(cam_all) / chunk_size)
    inside_min = voxel_model.scene_center - 0.5 * voxel_model.inside_extent * bbox_scale
    inside_max = voxel_model.scene_center + 0.5 * voxel_model.inside_extent * bbox_scale

    vol = SparseVoxelModel(sh_degree=0)
    vol.model_init(
        bounding=torch.stack([inside_min, inside_max]),
        outside_level=0,
        init_n_level=init_level,
    )

    grid_tsdf = None
    for level in range(init_level, final_level + 1):
        bandwidth = bandwidth_vox * vol.vox_size[0].item()
        tsdf_accum = torch.zeros(vol.grid_pts_xyz.shape[0], device="cuda")
        tsdf_count = torch.zeros(vol.grid_pts_xyz.shape[0], device="cuda")
        for i in range(num_chunks):
            cams = cam_all[i * chunk_size:(i + 1) * chunk_size]
            depths, alphas = [], []
            for cam in cams:
                with torch.no_grad():
                    render = voxel_model.render(cam, output_depth=True, output_T=True)
                    depth = render["raw_depth"][[2]]
                    alpha = 1 - render["raw_T"]
                    if save_gpu:
                        depth = depth.cpu()
                        alpha = alpha.cpu()
                    depths.append(depth)
                    alphas.append(alpha)
            tsdf = tsdf_fusion(
                Fuser, cams, depths, alphas, vol.grid_pts_xyz,
                bandwidth, crop_border, alpha_thres,
            )
            valid = ~tsdf.isnan()
            tsdf_accum[valid] += tsdf[valid]
            tsdf_count[valid] += 1
            del tsdf, depths, alphas
            torch.cuda.empty_cache()

        grid_tsdf = tsdf_accum / tsdf_count.clamp(min=1)
        grid_tsdf[tsdf_count == 0] = float("nan")
        if level < final_level:
            vox_tsdf = grid_tsdf[vol.vox_key]
            thickness = min(2 / bandwidth_vox, 0.99)
            prune = (
                vox_tsdf.isnan().any(-1)
                | (vox_tsdf.amax(1) < -thickness)
                | (vox_tsdf.amin(1) > thickness)
            )
            vol.pruning(prune)
            vol.subdividing(torch.ones([vol.num_voxels], dtype=torch.bool))

    surface_mask = torch.isfinite(grid_tsdf) & (grid_tsdf.abs() < 0.5)
    pts = vol.grid_pts_xyz[surface_mask]
    pts_gpu = pts.cuda()
    colors = torch.full((len(pts), 3), 0.5, device="cuda")
    closest_dist = torch.full((len(pts),), 1e9, device="cuda")

    for cam in tqdm(cam_all, desc="Lifting 2D masks"):
        with torch.no_grad():
            render = voxel_model.render(cam, output_depth=True, output_T=True)
            depth_img = render["raw_depth"][[2]]
            alpha_img = 1 - render["raw_T"]
            mask_img = cam.image.cuda()
            h, w = depth_img.shape[-2:]
            uv = cam.project(pts_gpu)
            valid = (uv.abs() <= 1).all(-1)
            idx = torch.where(valid)[0]
            if len(idx) == 0:
                continue
            uv_sel = uv[idx]
            pts_sel = pts_gpu[idx]
            alpha = torch.nn.functional.grid_sample(
                alpha_img.view(1, 1, h, w), uv_sel.view(1, 1, -1, 2), align_corners=False
            ).flatten()
            visible = alpha > alpha_thres
            idx = idx[visible]
            uv_sel = uv_sel[visible]
            pts_sel = pts_sel[visible]
            if len(idx) == 0:
                continue
            depth = torch.nn.functional.grid_sample(
                depth_img.view(1, 1, h, w), uv_sel.view(1, 1, -1, 2), align_corners=False
            ).flatten()
            pts_depth = ((pts_sel - cam.position) @ cam.lookat)
            dist = (depth - pts_depth).abs()
            update = dist < closest_dist[idx]
            idx = idx[update]
            uv_sel = uv_sel[update]
            dist = dist[update]
            mask_val = torch.nn.functional.grid_sample(
                mask_img[None], uv_sel.view(1, 1, -1, 2), align_corners=False
            ).squeeze().T
            colors[idx] = mask_val
            closest_dist[idx] = dist

    return pts.cpu().numpy(), (colors.cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Lift 2D kernel masks onto an RGB-trained SVRaster surface.")
    ap.add_argument("--svraster-root", type=Path, required=True, help="Clone of https://github.com/NVlabs/svraster")
    ap.add_argument("--model-path", type=Path, required=True, help="RGB-trained SVRaster model directory containing config.yaml")
    ap.add_argument("--mask-dir-name", required=True, help="Mask directory name relative to the SVRaster source dataset")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iteration", type=int, default=20000)
    ap.add_argument("--init-level", type=int, default=8)
    ap.add_argument("--final-level", type=int, default=10)
    ap.add_argument("--bandwidth-vox", type=float, default=5.0)
    ap.add_argument("--bbox-scale", type=float, default=0.8)
    ap.add_argument("--crop-border", type=float, default=0.01)
    ap.add_argument("--alpha-thres", type=float, default=0.9)
    ap.add_argument("--keep-on-gpu", action="store_true", help="Keep intermediate depth/alpha tensors on GPU.")
    args = ap.parse_args()

    cfg, update_config, DataPack, SparseVoxelModel, Fuser = import_svraster(args.svraster_root)
    update_config(str(args.model_path / "config.yaml"))
    cfg.data.res_downscale = 1.0
    cfg.data.image_dir_name = args.mask_dir_name
    cfg.data.mask_dir_name = args.mask_dir_name

    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        mask_dir_name=cfg.data.mask_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
        camera_params_only=False,
    )
    model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )
    model.load_iteration(str(args.model_path), args.iteration)
    model.freeze_vox_geo()

    points, colors = extract_and_paint(
        model, data_pack, SparseVoxelModel, Fuser,
        args.init_level, args.final_level, args.bbox_scale,
        args.bandwidth_vox, args.crop_border, args.alpha_thres,
        save_gpu=not args.keep_on_gpu,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trimesh.points.PointCloud(points, colors).export(args.output)
    print(f"Saved {len(points):,} mask-painted surface points: {args.output}")


if __name__ == "__main__":
    main()
