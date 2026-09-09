#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel3d.clustering import count_kernel_points, load_mask_painted_ply
from kernel3d.metrics import count_metrics


def load_gt(path: Path) -> dict[str, int]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["sample"]: int(r["gt"]) for r in rows}


def main():
    ap = argparse.ArgumentParser(
        description="Count pre-generated mask-lifted PLYs for an alpha sweep. "
                    "Expected layout: ROOT/alpha_0.2/*.ply, ROOT/alpha_0.4/*.ply, ..."
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--output", type=Path, default=Path("alpha_ablation_results.csv"))
    args = ap.parse_args()

    gt = load_gt(args.ground_truth)
    rows = []
    for alpha in args.alphas:
        d = args.root / f"alpha_{alpha:.1f}"
        files = sorted(d.glob("*.ply"))
        if not files:
            print(f"WARNING: no PLY files in {d}")
            continue
        alpha_gt, alpha_pred = [], []
        for path in files:
            sample = path.stem.split("_")[0]
            if sample not in gt:
                print(f"WARNING: no GT for {sample}; skipped")
                continue
            points, colors = load_mask_painted_ply(path)
            _, _, pred = count_kernel_points(points, colors, adaptive_split=True)
            alpha_gt.append(gt[sample]); alpha_pred.append(pred)
            rows.append({"alpha": alpha, "sample": sample, "gt": gt[sample], "pred": pred,
                         "error": gt[sample] - pred, "abs_error": abs(gt[sample] - pred)})
        if alpha_gt:
            m = count_metrics(alpha_gt, alpha_pred)
            print(f"alpha={alpha:.1f}: MAE={m['MAE']:.2f}, R2={m['R2']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "sample", "gt", "pred", "error", "abs_error"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
