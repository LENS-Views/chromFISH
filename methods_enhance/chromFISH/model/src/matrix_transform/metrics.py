from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity


def safe_correlations(prediction: np.ndarray, target: np.ndarray):
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if prediction.std() <= 1e-12 or target.std() <= 1e-12:
        return 0.0, 0.0
    pcc, _ = pearsonr(target, prediction)
    spearman, _ = spearmanr(target, prediction)
    return (
        float(pcc) if np.isfinite(pcc) else 0.0,
        float(spearman) if np.isfinite(spearman) else 0.0,
    )


def compute_metrics(
    prediction: np.ndarray, target: np.ndarray, data_range: float
) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError(f"预测和目标形状不一致: {prediction.shape} vs {target.shape}")
    data_range = max(float(data_range), 1e-8)
    pcc, spearman = safe_correlations(prediction, target)
    mse = float(np.mean((prediction - target) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(data_range / np.sqrt(mse)))
    ssim = float(
        structural_similarity(
            prediction,
            target,
            data_range=data_range,
            win_size=3,
        )
    )
    return {
        "pcc": pcc,
        "spearman": spearman,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "ssim": ssim,
        "psnr": psnr,
    }


def mean_metrics(values: Iterable[Dict[str, float]]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {}
    keys = values[0].keys()
    return {
        key: float(np.nanmean([item[key] for item in values]))
        for key in keys
    }
