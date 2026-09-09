#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel3d.clustering import count_kernel_points, load_mask_painted_ply, save_labeled_ply


def count_one(path: Path, adaptive_split: bool, output_dir: Path | None):
    points, colors = load_mask_painted_ply(path)
    kernel_points, labels, count = count_kernel_points(points, colors, adaptive_split=adaptive_split)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_labeled_ply(output_dir / f"{path.stem}_labeled.ply", kernel_points, labels)
    return count


def main():
    ap = argparse.ArgumentParser(description="Topology-aware kernel clustering and counting.")
    ap.add_argument("--input", type=Path, required=True, help="A mask-painted PLY or a directory of PLY files.")
    ap.add_argument("--output-dir", type=Path, default=Path("count_outputs"))
    ap.add_argument("--no-adaptive-split", action="store_true", help="Ablation corresponding to the proportional-mask/no-split experiment.")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    files = [args.input] if args.input.is_file() else sorted(args.input.glob("*.ply"))
    if not files:
        raise FileNotFoundError(f"No PLY files found: {args.input}")
    rows = []
    for path in files:
        count = count_one(path, adaptive_split=not args.no_adaptive_split, output_dir=args.output_dir)
        rows.append({"sample": path.stem, "pred": count})
        print(f"{path.name}: {count}")
    csv_path = args.csv or (args.output_dir / "predictions.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "pred"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
