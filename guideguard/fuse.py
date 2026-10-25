from __future__ import annotations

from typing import Sequence

import numpy as np

from .volumes import VOL_SHAPE


def fuse(arrays: Sequence[np.ndarray], weights: Sequence[float], brain: np.ndarray) -> np.ndarray:
    if len(arrays) != len(weights) or abs(sum(weights) - 1.0) > 1e-9 or min(weights) < 0:
        raise ValueError("weights must be >= 0, one per member, and sum to 1")
    acc = np.zeros(VOL_SHAPE, np.float32)
    for a, w in zip(arrays, weights):
        a = np.asarray(a).astype(np.float32)
        if a.shape != VOL_SHAPE or not np.isfinite(a).all():
            raise ValueError(f"member of shape {a.shape} or with non-finite values")
        acc += np.float32(w) * a
    return (np.clip(acc, 0.0, 1.0) * brain).astype(np.float16)
