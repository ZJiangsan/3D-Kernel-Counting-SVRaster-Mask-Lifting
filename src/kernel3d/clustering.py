from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class RegionParams:
    density_percentile: float
    eps_base: float
    eps_split: float
    min_size_factor: float
    split_trigger: float
    final_size_factor: float


REGION_PARAMS = {
    "center": RegionParams(10.0, 0.0155, 0.0065, 0.29, 2.40, 0.11),
    "head": RegionParams(1.0, 0.0145, 0.0085, 0.25, 2.35, 0.13),
    "tail": RegionParams(1.0, 0.0145, 0.0085, 0.25, 2.35, 0.13),
}


def _principal_axis_coords(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([])
    t = PCA(n_components=1).fit_transform(points).ravel()
    span = t.max() - t.min()
    return np.zeros_like(t) if span == 0 else (t - t.min()) / span


def _find_min_cut_plane(
    t_coords: np.ndarray,
    search_center: float,
    search_radius: float = 0.09,
    num_candidates: int = 15,
    thickness: float = 0.01,
) -> float:
    start = max(0.0, search_center - search_radius)
    end = min(1.0, search_center + search_radius)
    best_cut = search_center
    min_density = float("inf")
    for cut in np.linspace(start, end, num_candidates):
        density = np.sum((t_coords > cut - thickness / 2) & (t_coords < cut + thickness / 2))
        if density < min_density:
            min_density = density
            best_cut = float(cut)
    return best_cut


def split_cob_regions(points: np.ndarray) -> dict[str, np.ndarray]:
    """Split the cob into head/center/tail using the manuscript's PCA valley search."""
    t = _principal_axis_coords(points)
    if len(t) == 0:
        z = np.zeros(0, dtype=bool)
        return {"head": z, "center": z, "tail": z}
    low = np.percentile(t, 0.01)
    high = np.percentile(t, 99.99)
    valid = (t >= low) & (t <= high)
    t_clean = t[valid]
    if len(t_clean) == 0:
        z = np.zeros(len(points), dtype=bool)
        return {"head": z.copy(), "center": z.copy(), "tail": z.copy()}
    span = t_clean.max() - t_clean.min()
    t_clean = np.zeros_like(t_clean) if span == 0 else (t_clean - t_clean.min()) / span
    cut1 = _find_min_cut_plane(t_clean, 0.27)
    cut2 = _find_min_cut_plane(t_clean, 0.73)
    idx = np.where(valid)[0]
    head = np.zeros(len(points), dtype=bool)
    center = np.zeros(len(points), dtype=bool)
    tail = np.zeros(len(points), dtype=bool)
    head[idx[t_clean < cut1]] = True
    center[idx[(t_clean >= cut1) & (t_clean <= cut2)]] = True
    tail[idx[t_clean > cut2]] = True
    return {"head": head, "center": center, "tail": tail}


def process_region(
    points: np.ndarray,
    region_name: str,
    adaptive_split: bool = True,
) -> tuple[np.ndarray, int]:
    """Cluster one cob region using the final experimental parameters."""
    original_n = len(points)
    if original_n == 0:
        return np.full(0, -1, dtype=int), 0
    p = REGION_PARAMS[region_name]

    n_neighbors = min(5, len(points))
    if n_neighbors < 2:
        return np.full(original_n, -1, dtype=int), 0
    dists, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(points).kneighbors(points)
    local_density = 1.0 / (dists[:, 1:].mean(axis=1) + 1e-6)
    keep = local_density > np.percentile(local_density, p.density_percentile)
    if not np.any(keep):
        return np.full(original_n, -1, dtype=int), 0
    pts = points[keep]

    labels = DBSCAN(eps=p.eps_base, min_samples=6).fit(pts).labels_
    unique = sorted(set(labels) - {-1})
    sizes = [int(np.sum(labels == lab)) for lab in unique]
    if not sizes:
        return np.full(original_n, -1, dtype=int), 0
    non_dust = [s for s in sizes if s > 15] or sizes
    median_size = float(np.median(non_dust))
    adaptive_min = median_size * p.min_size_factor
    split_threshold = median_size * p.split_trigger

    filtered = np.full_like(labels, -1)
    kept_labels: list[int] = []
    for lab in unique:
        if np.sum(labels == lab) > adaptive_min:
            filtered[labels == lab] = len(kept_labels)
            kept_labels.append(lab)

    refined = np.full_like(filtered, -1)
    next_id = 0
    for lab in range(len(kept_labels)):
        mask = filtered == lab
        cluster = pts[mask]
        idx = np.where(mask)[0]
        if len(cluster) <= split_threshold or not adaptive_split:
            refined[mask] = next_id
            next_id += 1
            continue

        split_success = False
        sub_labels = DBSCAN(eps=p.eps_split, min_samples=5).fit(cluster).labels_
        sub_ids = sorted(set(sub_labels) - {-1})
        if len(sub_ids) >= 2:
            child_sizes = [np.sum(sub_labels == s) for s in sub_ids]
            if all(size <= split_threshold for size in child_sizes):
                for s in sub_ids:
                    refined[idx[sub_labels == s]] = next_id
                    next_id += 1
                split_success = True

        if not split_success:
            ratio = len(cluster) / max(median_size * 1.4, 1.0)
            k = int(np.clip(ratio, 2, 4))
            km_labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit(cluster).labels_
            child_sizes = [np.sum(km_labels == i) for i in range(k)]
            if all(size <= split_threshold for size in child_sizes):
                for i in range(k):
                    refined[idx[km_labels == i]] = next_id
                    next_id += 1
                split_success = True

        if not split_success:
            refined[mask] = next_id
            next_id += 1

    cleaned = np.full_like(refined, -1)
    final_id = 0
    final_threshold = median_size * p.final_size_factor
    for lab in range(next_id):
        if np.sum(refined == lab) >= final_threshold:
            cleaned[refined == lab] = final_id
            final_id += 1

    out = np.full(original_n, -1, dtype=int)
    out[keep] = cleaned
    return out, final_id


def count_kernel_points(
    points: np.ndarray,
    colors: np.ndarray,
    adaptive_split: bool = True,
    mask_brightness_threshold: int = 220,
    preclean_eps: float = 0.08,
    preclean_min_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Count kernels from a mask-painted SVRaster point cloud."""
    rgb = colors[:, :3]
    brightness = rgb.min(axis=1)
    kernel_points = points[brightness > mask_brightness_threshold]
    if len(kernel_points) == 0:
        return kernel_points, np.full(0, -1, dtype=int), 0

    if len(kernel_points) > 50:
        labels = DBSCAN(eps=preclean_eps, min_samples=preclean_min_samples).fit(kernel_points).labels_
        valid_ids = set(labels) - {-1}
        if valid_ids:
            sizes = {lab: np.sum(labels == lab) for lab in valid_ids}
            kernel_points = kernel_points[labels == max(sizes, key=sizes.get)]

    regions = split_cob_regions(kernel_points)
    all_labels = np.full(len(kernel_points), -1, dtype=int)
    next_global_id = 0
    for region_name, region_mask in regions.items():
        region_points = kernel_points[region_mask]
        region_labels, count = process_region(region_points, region_name, adaptive_split=adaptive_split)
        idx = np.where(region_mask)[0]
        for local_id in range(count):
            all_labels[idx[region_labels == local_id]] = next_global_id
            next_global_id += 1
    return kernel_points, all_labels, next_global_id


def load_mask_painted_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    cloud = trimesh.load(path, process=False)
    if not hasattr(cloud, "vertices"):
        raise ValueError(f"Expected a point cloud/mesh with vertices: {path}")
    points = np.asarray(cloud.vertices)
    colors = np.asarray(cloud.visual.vertex_colors)
    if len(colors) != len(points):
        raise ValueError(f"PLY has no per-point colors: {path}")
    return points, colors


def save_labeled_ply(path: str | Path, points: np.ndarray, labels: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.zeros((len(points), 4), dtype=np.uint8)
    colors[:, 3] = 255
    cmap = plt.get_cmap("nipy_spectral")
    for lab in sorted(set(labels) - {-1}):
        rng = np.random.default_rng(int(lab))
        rgb = 0.3 + 0.7 * np.asarray(cmap(rng.random())[:3])
        colors[labels == lab, :3] = (rgb * 255).astype(np.uint8)
    trimesh.points.PointCloud(points, colors).export(path)
