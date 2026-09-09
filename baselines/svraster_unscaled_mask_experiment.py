#!/usr/bin/env python3
"""Reference evaluator for the supplied unscaled-mask SVRaster ablation.

The original development file mixed TSDF extraction, mask painting, multiple
clustering experiments, plotting, and historical results. The clean repository
keeps the reported per-ear predictions in results/unscaled_no_split_predictions.csv
and evaluates them here.

The supplied source records MAE = 80.2222 for this no-split run, and that value
is reported directly without reinterpretation.
"""
from pathlib import Path
import csv, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel3d.metrics import count_metrics

REFERENCE = ROOT / "results" / "unscaled_no_split_predictions.csv"

def main():
    with REFERENCE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    metrics = count_metrics([int(r["gt"]) for r in rows], [int(r["pred"]) for r in rows])
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
