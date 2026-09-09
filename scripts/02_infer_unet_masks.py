#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel3d.masks import pad_to_multiple_reflect, resize_keep_aspect_max
from kernel3d.unet import build_unet


def predict(model, image_bgr: np.ndarray, device, threshold: float) -> np.ndarray:
    original_h, original_w = image_bgr.shape[:2]
    h, w = image_bgr.shape[:2]
    image = cv2.resize(image_bgr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    image = resize_keep_aspect_max(image, 1024)
    prepad_h, prepad_w = image.shape[:2]
    image, _ = pad_to_multiple_reflect(image, 32)
    # Match the original batch_inference.py: Albumentations/ImageNet normalization.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    x = torch.from_numpy(image).permute(2, 0, 1)[None].float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    prob = prob[:prepad_h, :prepad_w]
    prob = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser(description="Apply the trained U-Net to all RGB frames.")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(encoder_weights=None).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    n = 0
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        mask = predict(model, image, device, args.threshold)
        cv2.imwrite(str(args.output / f"{path.stem}.png"), mask)
        n += 1
    print(f"Saved {n} masks to {args.output}")


if __name__ == "__main__":
    main()
