from __future__ import annotations

import numpy as np
from scipy import ndimage


def psnr_region(pred: np.ndarray, gt: np.ndarray, region: np.ndarray, data_range: float = 1.0) -> float:
    m = np.asarray(region, bool)
    if not m.any():
        return float("nan")
    mse = float(np.mean((np.asarray(pred, np.float64)[m] - np.asarray(gt, np.float64)[m]) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(data_range ** 2 / mse))


def mae_region(pred: np.ndarray, gt: np.ndarray, region: np.ndarray) -> float:
    m = np.asarray(region, bool)
    return float(np.mean(np.abs(np.asarray(pred, np.float64)[m] - np.asarray(gt, np.float64)[m]))) if m.any() \
        else float("nan")


def ssim_map(pred: np.ndarray, gt: np.ndarray, sigma: float = 1.5, data_range: float = 1.0) -> np.ndarray:
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda a: ndimage.gaussian_filter(a, sigma, truncate=3.5)
    mp, mg = f(p), f(g)
    vp, vg, cpg = f(p * p) - mp * mp, f(g * g) - mg * mg, f(p * g) - mp * mg
    return ((2 * mp * mg + c1) * (2 * cpg + c2)) / ((mp * mp + mg * mg + c1) * (vp + vg + c2))


def ssim_region(pred: np.ndarray, gt: np.ndarray, region: np.ndarray, **kw) -> float:
    m = np.asarray(region, bool)
    return float(ssim_map(pred, gt, **kw)[m].mean()) if m.any() else float("nan")


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * float((a & b).sum()) / s


def image_metrics(pred: np.ndarray, gt: np.ndarray, brain: np.ndarray, seg: np.ndarray = None) -> dict:
    out = {"psnr_brain": psnr_region(pred, gt, brain), "ssim_brain": ssim_region(pred, gt, brain),
           "mae_brain": mae_region(pred, gt, brain)}
    if seg is not None:
        et = (np.asarray(seg) == 3) & brain
        out["psnr_ET"] = psnr_region(pred, gt, et)
    return out
