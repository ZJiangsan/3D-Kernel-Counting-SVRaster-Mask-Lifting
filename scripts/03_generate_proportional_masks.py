#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel3d.masks import generate_mask_directory


def main():
    ap = argparse.ArgumentParser(description="Transform incomplete surface masks into proportional dot masks.")
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.6)
    args = ap.parse_args()
    n = generate_mask_directory(args.masks, args.output, args.alpha)
    print(f"Generated {n} masks at alpha={args.alpha:.2f}: {args.output}")


if __name__ == "__main__":
    main()
