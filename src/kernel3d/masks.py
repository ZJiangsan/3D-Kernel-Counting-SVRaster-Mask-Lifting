from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN


def labelme_json_to_binary_mask(json_path: str | Path, height: int, width: int) -> np.ndarray:
    """Convert polygon annotations from a LabelMe/AnyLabeling JSON file to a binary mask."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data.get("shapes", []):
        pts = np.asarray(shape["points"], dtype=np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask


def pad_to_multiple_reflect(image: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflect-pad the bottom/right edges so H and W are divisible by ``multiple``."""
    h, w = image.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    pad_bottom = new_h - h
    pad_right = new_w - w
    if pad_bottom == 0 and pad_right == 0:
        return image, (0, 0)
    padded = cv2.copyMakeBorder(
        image, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_REFLECT_101
    )
    return padded, (pad_bottom, pad_right)


def resize_keep_aspect_max(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale == 1.0:
        return image
    return cv2.resize(
        image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )


def proportional_dot_mask(
    mask: np.ndarray,
    alpha: float = 0.6,
    min_distance: int = 7,
    threshold_abs: float = 3.0,
    isolate_main_cluster: bool = True,
    cluster_eps: float = 80.0,
    cluster_min_samples: int = 3,
    min_radius: int = 2,
) -> np.ndarray:
    """Generate proportional dot masks used in the manuscript.

    The implementation follows the experimental scripts: distance transform ->
    Gaussian smoothing -> local maxima -> optional DBSCAN selection of the main
    cob cluster -> circles whose radii are ``alpha`` times the local distance-
    transform radius.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    _, mask_bin = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (3, 3), 0)
    coordinates = peak_local_max(
        dist,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False,
    )
    if len(coordinates) == 0:
        return np.zeros_like(mask_bin)

    if isolate_main_cluster and len(coordinates) >= cluster_min_samples:
        labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples).fit(coordinates).labels_
        valid = labels >= 0
        if np.any(valid):
            ids, counts = np.unique(labels[valid], return_counts=True)
            main_id = ids[np.argmax(counts)]
            coordinates = coordinates[labels == main_id]

    out = np.zeros_like(mask_bin)
    for row, col in coordinates:
        radius = max(min_radius, int(dist[row, col] * alpha))
        cv2.circle(out, (int(col), int(row)), radius, 255, -1)
    return out


def generate_mask_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    alpha: float,
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp"),
) -> int:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in extensions:
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        out = proportional_dot_mask(mask, alpha=alpha)
        cv2.imwrite(str(output_dir / path.name), out)
        n += 1
    return n
