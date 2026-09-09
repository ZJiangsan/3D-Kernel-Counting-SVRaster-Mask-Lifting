#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel3d.metrics import count_metrics


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "gt" not in rows[0] or "pred" not in rows[0]:
        raise ValueError("CSV must contain columns: sample, gt, pred")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate kernel-count predictions.")
    ap.add_argument("--csv", type=Path, required=True, help="CSV with columns sample,gt,pred")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    rows = read_rows(args.csv)
    metrics = count_metrics([float(r["gt"]) for r in rows], [float(r["pred"]) for r in rows])
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
