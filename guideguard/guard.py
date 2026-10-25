from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import ndimage
from scipy.special import xlogy
from scipy.stats import binom

ET_LABEL = 3
UNLIKELY, CHECK, SUPPORTED = 0, 1, 2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit_grid(n: int = 200, lo: float = -12.0, hi: float = 12.0) -> np.ndarray:
    return np.concatenate([[np.inf], sigmoid(np.linspace(hi, lo, n)), [0.0]]).astype(np.float64)


def as_score(s, brain) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    brain = np.asarray(brain, bool)
    if not np.all(np.isfinite(s[brain])):
        raise ValueError("score map has non-finite values inside the brain")
    if s[brain].size and (s[brain].min() < -1e-6 or s[brain].max() > 1 + 1e-6):
        raise ValueError("score map must hold probabilities in [0,1]")
    return np.where(brain, np.clip(np.nan_to_num(s), 0.0, 1.0), 0.0)


def averaged_score(g_et: np.ndarray, m_et: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(g_et, np.float64) + np.asarray(m_et, np.float64))


def ehat_from_prob(p_et, brain, thr: float = 0.5) -> np.ndarray:
    return (as_score(p_et, brain) >= thr) & np.asarray(brain, bool)


def dilate_mm(mask, radius_mm: float = 2.0) -> np.ndarray:
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    if not mask.any():
        return out
    marg = int(np.ceil(radius_mm)) + 1
    nz = np.nonzero(mask)
    sl = tuple(slice(max(int(a.min()) - marg, 0), min(int(a.max()) + marg + 1, n)) for a, n in zip(nz, mask.shape))
    out[sl] = ndimage.distance_transform_edt(~mask[sl]) <= radius_mm + 1e-9
    return out


def flag_set(s, ehat, brain, lam: float) -> np.ndarray:
    brain = np.asarray(brain, bool)
    return brain & ~np.asarray(ehat, bool) & (as_score(s, brain) >= lam)


def supported_set(s, ehat, brain, mu: Optional[float]) -> np.ndarray:
    ehat = np.asarray(ehat, bool) & np.asarray(brain, bool)
    if mu is None:
        return np.zeros_like(ehat)
    return ehat & (as_score(s, brain) >= mu)


def three_level_map(s, ehat, brain, lam: float, mu: Optional[float]) -> np.ndarray:
    brain = np.asarray(brain, bool)
    ehat = np.asarray(ehat, bool) & brain
    D = supported_set(s, ehat, brain, mu)
    C = flag_set(s, ehat, brain, lam)
    out = np.zeros(brain.shape, np.uint8)
    out[(ehat & ~D) | C] = CHECK
    out[D] = SUPPORTED
    return out


def _count_ge(values, grid):
    v = np.sort(np.asarray(values, np.float64).ravel())
    return (v.size - np.searchsorted(v, grid, side="left")).astype(np.int64)


def _count_lt(values, grid):
    v = np.sort(np.asarray(values, np.float64).ravel())
    return np.searchsorted(v, grid, side="left").astype(np.int64)


def lesion_labels(et, min_lesion_vox: int = 10):
    lab, n = ndimage.label(et, structure=np.ones((3, 3, 3), bool))
    if n == 0:
        return lab, []
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    return lab, [i for i in range(1, n + 1) if sizes[i] >= min_lesion_vox]


def patient_curves(s, ehat, seg, brain, lambdas, mus, dil_mm: float = 2.0, min_lesion_vox: int = 10,
                   lesion_cover: float = 0.5) -> Dict[str, np.ndarray]:
    brain = np.asarray(brain, bool)
    s = as_score(s, brain)
    ehat = np.asarray(ehat, bool) & brain
    et = (np.asarray(seg) == ET_LABEL) & brain
    not_e = brain & ~ehat
    miss = et & ~ehat
    dil = dilate_mm(et, dil_mm)
    out = {"et": int(et.sum()), "n_brain": int(brain.sum()),
           "miss_unacc": _count_lt(s[miss], lambdas),
           "flag": _count_ge(s[not_e], lambdas), "flag_fa": _count_ge(s[not_e & ~et], lambdas),
           "shown_fa": int((ehat & ~et).sum()),
           "d_size": _count_ge(s[ehat], mus), "d_fp": _count_ge(s[ehat & ~dil], mus)}
    lab, keep = lesion_labels(et, min_lesion_vox)
    les = np.zeros(len(lambdas), np.int64)
    for li in keep:
        l = lab == li
        covered = int((l & ehat).sum()) + _count_ge(s[l & ~ehat], lambdas)
        les += (covered < lesion_cover * int(l.sum())).astype(np.int64)
    out["n_lesions"] = len(keep)
    out["lesion_unacc"] = les
    return out


def stack_curves(curves: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack([np.asarray(c[k]) for c in curves], 0) for k in curves[0]}


def miss_losses(cur: Dict[str, np.ndarray], min_et: int = 50, variant: str = "voxel"):
    et = cur["et"].astype(np.float64)
    pop = et >= min_et
    if variant == "voxel":
        loss = np.where(pop[:, None], cur["miss_unacc"] / np.where(pop, et, 1.0)[:, None], 0.0)
    elif variant == "lesion":
        nl = cur["n_lesions"].astype(np.float64)
        pop = pop & (nl > 0)
        loss = np.where(pop[:, None], cur["lesion_unacc"] / np.where(pop, nl, 1.0)[:, None], 0.0)
    elif variant == "tail":
        r1 = np.where(pop[:, None], cur["miss_unacc"] / np.where(pop, et, 1.0)[:, None], 0.0)
        loss = np.where(pop[:, None], (r1 > 0.5).astype(np.float64), 0.0)
    else:
        raise ValueError(variant)
    return loss, pop


def fdp_losses(cur: Dict[str, np.ndarray]) -> np.ndarray:
    return (cur["d_fp"] / np.maximum(cur["d_size"], 1)).astype(np.float64)


def crc_index(loss: np.ndarray, alpha: float, B: float = 1.0) -> int:
    loss = np.asarray(loss, np.float64)
    if not np.all(loss[:, -1] == 0):
        raise ValueError("grid must end at lambda = 0 with loss 0")
    if np.any(np.diff(loss, axis=1) > 1e-12):
        raise ValueError("loss must be non-increasing along the descending grid")
    n = loss.shape[0]
    ok = (loss.sum(0) + B) / (n + 1.0) <= alpha
    tail_ok = np.flip(np.logical_and.accumulate(np.flip(ok)))
    return int(np.argmax(tail_ok)) if tail_ok[-1] else loss.shape[1] - 1


def hb_pvalue(loss_sum: float, n: int, beta: float) -> float:
    if n <= 0:
        return 1.0
    rhat = loss_sum / n
    a = min(rhat, beta)
    h1 = xlogy(a, a / beta) + xlogy(1.0 - a, (1.0 - a) / (1.0 - beta))
    p_h = np.exp(-n * h1)
    p_b = np.e * binom.cdf(np.ceil(loss_sum - 1e-9), n, beta)
    return float(min(p_h, p_b, 1.0))


def ltt_index(loss: np.ndarray, beta: float, delta: float) -> int:
    loss = np.asarray(loss, np.float64)
    j = -1
    for k in range(loss.shape[1]):
        if hb_pvalue(float(loss[:, k].sum()), loss.shape[0], beta) <= delta:
            j = k
        else:
            break
    return j


def calibrate(curves: Sequence[Dict[str, np.ndarray]], lambdas: np.ndarray, mus: np.ndarray, alpha: float = 0.1,
              beta: float = 0.2, delta: float = 0.1, min_et: int = 50, variant: str = "voxel") -> dict:
    cur = stack_curves(curves)
    loss, pop = miss_losses(cur, min_et, variant)
    i = crc_index(loss[pop], alpha)
    j = ltt_index(fdp_losses(cur), beta, delta)
    lam = float(lambdas[i])
    mu = float(mus[j]) if j >= 0 else None
    return {"lambda_hat": None if not np.isfinite(lam) else lam, "mu_hat": None if mu is None or not np.isfinite(mu)
            else mu, "alpha": alpha, "beta": beta, "delta": delta, "variant": variant, "min_et": min_et,
            "n_cal": int(len(curves)), "n_cal_et_positive": int(pop.sum())}


def apply_thresholds(s, ehat, brain, thr: dict) -> np.ndarray:
    lam = thr["lambda_hat"]
    return three_level_map(s, ehat, brain, np.inf if lam is None else lam, thr["mu_hat"])


def evaluate_map(m: np.ndarray, ehat, seg, brain, min_et: int = 50) -> dict:
    brain = np.asarray(brain, bool)
    et = (np.asarray(seg) == ET_LABEL) & brain
    marked = (m > 0) & brain
    out = {"et_vox": int(et.sum()), "review_burden_ml": float((marked & ~et).sum()) / 1000.0,
           "check_ml": float((m == CHECK).sum()) / 1000.0, "supported_ml": float((m == SUPPORTED).sum()) / 1000.0}
    if et.sum() >= min_et:
        out["miss"] = float((et & ~marked).sum()) / float(et.sum())
    sup = m == SUPPORTED
    out["supported_fdp"] = float((sup & ~dilate_mm(et)).sum()) / max(int(sup.sum()), 1)
    return out
