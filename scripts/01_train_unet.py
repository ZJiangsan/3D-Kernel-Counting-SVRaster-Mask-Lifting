#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel3d.masks import labelme_json_to_binary_mask, pad_to_multiple_reflect, resize_keep_aspect_max
from kernel3d.unet import BCEDiceLoss, build_unet


def augment_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.CLAHE(clip_limit=3.0, p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
        A.ChannelDropout(p=0.1),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.2),
    ])


def prepare_dataset(images_dir: Path, labels_dir: Path, work_dir: Path, n_aug: int, seed: int):
    processed_images = work_dir / "processed/images"
    processed_masks = work_dir / "processed/masks"
    train_root = work_dir / "split/train"
    val_root = work_dir / "split/val"
    for d in [processed_images, processed_masks, train_root / "images", train_root / "masks",
              val_root / "images", val_root / "masks"]:
        d.mkdir(parents=True, exist_ok=True)
    # Avoid stale duplicates when rerunning.
    for d in [processed_images, processed_masks, train_root / "images", train_root / "masks",
              val_root / "images", val_root / "masks"]:
        for p in d.glob("*.png"):
            p.unlink()

    aug = augment_pipeline()
    for path in tqdm(sorted(images_dir.iterdir()), desc="Preparing U-Net data"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = labels_dir / f"{path.stem}.json"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing annotation: {label_path}")
        image = cv2.imread(str(path))
        h, w = image.shape[:2]
        mask = labelme_json_to_binary_mask(label_path, h, w)
        image = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        image = resize_keep_aspect_max(image, 1024)
        mask = resize_keep_aspect_max(mask, 1024)
        image, _ = pad_to_multiple_reflect(image, 32)
        mask, _ = pad_to_multiple_reflect(mask, 32)
        base = path.stem
        cv2.imwrite(str(processed_images / f"{base}_orig.png"), image)
        cv2.imwrite(str(processed_masks / f"{base}_orig.png"), mask)
        for i in range(n_aug):
            out = aug(image=image, mask=mask)
            ai, _ = pad_to_multiple_reflect(out["image"], 32)
            am, _ = pad_to_multiple_reflect(out["mask"], 32)
            cv2.imwrite(str(processed_images / f"{base}_aug{i:03d}.png"), ai)
            cv2.imwrite(str(processed_masks / f"{base}_aug{i:03d}.png"), am)

    # This preserves the experimental script: split after augmentation.
    files = sorted(processed_images.glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(len(files) * 0.2)
    val_set, train_set = files[:n_val], files[n_val:]
    for subset, dst in [(train_set, train_root), (val_set, val_root)]:
        for img_path in subset:
            shutil.copy2(img_path, dst / "images" / img_path.name)
            shutil.copy2(processed_masks / img_path.name, dst / "masks" / img_path.name)
    return train_root, val_root


class SegDataset(Dataset):
    def __init__(self, root: Path):
        self.images = sorted((root / "images").glob("*.png"))
        self.masks = sorted((root / "masks").glob("*.png"))
        if len(self.images) != len(self.masks):
            raise ValueError("Image/mask file count mismatch")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mask = (mask > 0).astype(np.float32)
        # Match the manuscript training script: Albumentations ImageNet normalization.
        image = A.Normalize()(image=image)["image"]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask)
        return image, mask


def main():
    ap = argparse.ArgumentParser(description="Train the binary U-Net used to generate incomplete kernel-surface masks.")
    ap.add_argument("--images", type=Path, required=True, help="Four manually annotated RGB images.")
    ap.add_argument("--labels", type=Path, required=True, help="AnyLabeling/LabelMe JSON polygon annotations.")
    ap.add_argument("--work-dir", type=Path, default=Path("unet_work"))
    ap.add_argument("--output", type=Path, default=Path("weights/kernel_unet.pth"))
    ap.add_argument("--resume", type=Path, default=None, help="Optional previous 3-image checkpoint used as initialization.")
    ap.add_argument("--augmentations", type=int, default=60, help="Experimental script used 60 augmented variants per annotated image.")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--early-stop", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_root, val_root = prepare_dataset(args.images, args.labels, args.work_dir, args.augmentations, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Initialized from: {args.resume}")

    train_dl = DataLoader(SegDataset(train_root), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(SegDataset(val_root), batch_size=args.batch_size, shuffle=False)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    patience = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for image, mask in train_dl:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(image)[:, 0]
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(len(train_dl), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, mask in val_dl:
                image, mask = image.to(device), mask.to(device)
                val_loss += float(criterion(model(image)[:, 0], mask).item())
        val_loss /= max(len(val_dl), 1)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:04d}: train={train_loss:.5f} val={val_loss:.5f}")

        if val_loss < best:
            best = val_loss
            patience = 0
            torch.save(model.state_dict(), args.output)
        else:
            patience += 1
        if patience >= args.early_stop:
            print("Early stopping")
            break

    with open(args.output.with_suffix(".history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
