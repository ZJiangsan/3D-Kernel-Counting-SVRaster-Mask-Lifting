from __future__ import annotations

import numpy as np


def count_metrics(gt, pred) -> dict[str, float]:
    gt = np.asarray(gt, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if gt.shape != pred.shape:
        raise ValueError("gt and pred must have the same shape")
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    rel = float(np.mean(np.abs(err) / np.maximum(gt, 1.0)) * 100.0)
    if len(gt) > 1 and np.var(gt) > 0:
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = float("nan")
    return {"MAE": mae, "RMSE": rmse, "mean_relative_error_percent": rel, "R2": r2}
