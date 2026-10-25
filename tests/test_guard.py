import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard import guard as G
from guideguard.fuse import fuse


def synthetic_patient(rng, shape=(24, 24, 16)):
    brain = np.zeros(shape, bool)
    brain[2:-2, 2:-2, 2:-2] = True
    seg = np.zeros(shape, np.uint8)
    c = rng.integers(6, 14, size=3)
    r = rng.integers(2, 4)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    et = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r ** 2) & brain
    seg[et] = 3
    s = np.clip(0.6 * et + rng.normal(0.2, 0.15, shape), 0, 1) * brain
    ehat = et & (rng.random(shape) < 0.5)
    return s, ehat, seg, brain


def test_crc_controls_miss():
    rng = np.random.default_rng(0)
    lam, mus = G.logit_grid(60), G.logit_grid(60)
    pts = [synthetic_patient(rng) for _ in range(300)]
    curves = [G.patient_curves(s, e, seg, b, lam, mus) for s, e, seg, b in pts]
    miss = []
    for k in range(20):
        idx = rng.permutation(len(pts))
        cal, test = idx[:100], idx[100:]
        thr = G.calibrate([curves[i] for i in cal], lam, mus, alpha=0.1, min_et=10)
        for i in test:
            s, e, seg, b = pts[i]
            m = G.apply_thresholds(s, e, b, thr)
            r = G.evaluate_map(m, e, seg, b, min_et=10)
            if "miss" in r:
                miss.append(r["miss"])
    assert np.mean(miss) <= 0.1 + 0.02


def test_lambda_zero_flags_whole_brain():
    rng = np.random.default_rng(1)
    s, e, seg, b = synthetic_patient(rng)
    m = G.three_level_map(s, e, b, 0.0, None)
    assert np.all(m[b] > 0) and np.all(m[~b] == 0)


def test_fuse_mean():
    brain = np.ones((240, 240, 155), bool)
    a, c = np.full(brain.shape, 0.2, np.float16), np.full(brain.shape, 0.6, np.float16)
    assert np.allclose(fuse([a, c], [0.5, 0.5], brain).astype(np.float32), 0.4, atol=1e-3)


def test_image_free_threshold_not_above_image_tied():
    rng = np.random.default_rng(2)
    lam, mus = G.logit_grid(60), G.logit_grid(60)
    pts = [synthetic_patient(rng) for _ in range(100)]
    tied = G.calibrate([G.patient_curves(s, e, seg, b, lam, mus) for s, e, seg, b in pts], lam, mus, min_et=10)
    free = G.calibrate([G.patient_curves(s, np.zeros_like(b), seg, b, lam, mus) for s, e, seg, b in pts], lam, mus,
                       min_et=10)
    lam_hat = lambda t: np.inf if t["lambda_hat"] is None else t["lambda_hat"]
    assert lam_hat(free) <= lam_hat(tied) < np.inf
