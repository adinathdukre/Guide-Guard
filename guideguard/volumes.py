from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

VOL_SHAPE = (240, 240, 155)
NC_MODS = ("t1n", "t2w", "t2f")
TARGET_MOD = "t1c"
ALL_MODS = NC_MODS + (TARGET_MOD,)
NC_PCT = (0.5, 99.5)
T1C_PCT = (0.5, 99.9)
BRATS_AFFINE = np.array([[-1.0, 0.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0, 239.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])


def brain_mask(raw: Dict[str, np.ndarray]) -> np.ndarray:
    b = np.zeros(raw["t1n"].shape, dtype=bool)
    for m in NC_MODS:
        b |= raw[m] != 0
    return b


def normalise(x: np.ndarray, brain: np.ndarray, pct) -> np.ndarray:
    v = x[brain].astype(np.float64)
    out = np.zeros(x.shape, dtype=np.float64)
    if v.size == 0:
        return out.astype(np.float32)
    lo, hi = np.percentile(v, pct)
    if hi > lo:
        out[brain] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


def map_seg(seg_raw: np.ndarray) -> np.ndarray:
    s = np.rint(np.asarray(seg_raw)).astype(np.int64)
    if not set(np.unique(s).tolist()) <= {0, 1, 2, 3, 4}:
        raise ValueError(f"unexpected labels {np.unique(s)}")
    return np.where(s == 4, 3, s).astype(np.uint8)


def load_raw_case(case_dir: Path, case: str, with_target: bool = True, with_seg: bool = True):
    import nibabel as nib

    raw, affine = {}, None
    names = NC_MODS + ((TARGET_MOD,) if with_target else ()) + (("seg",) if with_seg else ())
    for m in names:
        img = nib.load(str(Path(case_dir) / f"{case}-{m}.nii.gz"))
        if tuple(img.shape) != VOL_SHAPE:
            raise ValueError(f"{case} {m}: shape {img.shape} != {VOL_SHAPE} (SRI24, 1 mm, BraTS layout expected)")
        affine = img.affine if affine is None else affine
        raw[m] = np.asarray(img.get_fdata(dtype=np.float64))
    return raw, affine


def preprocess_case(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    brain = brain_mask(raw)
    out = {"brain": brain.astype(np.uint8)}
    for m in ALL_MODS:
        if m in raw:
            out[m] = normalise(raw[m], brain, T1C_PCT if m == TARGET_MOD else NC_PCT).astype(np.float16)
    if "seg" in raw:
        out["seg"] = map_seg(raw["seg"])
    return out


def save_case(out_dir: Path, arrays: Dict[str, np.ndarray], meta: Optional[dict] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, a in arrays.items():
        tmp = out_dir / f".{k}.{os.getpid()}.npy"
        np.save(tmp, a)
        os.replace(tmp, out_dir / f"{k}.npy")
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return out_dir


def load_case(case_dir: Path, mods: Sequence[str] = ALL_MODS, with_seg: bool = False) -> Dict[str, np.ndarray]:
    d = Path(case_dir)
    out: Dict[str, np.ndarray] = {}
    for m in mods:
        p = d / f"{m}.npy"
        if p.exists():
            out[m] = np.load(p).astype(np.float32)
    out["brain"] = np.load(d / "brain.npy").astype(bool)
    if with_seg and (d / "seg.npy").exists():
        out["seg"] = np.load(d / "seg.npy").astype(np.uint8)
    return out


def to_model(v: np.ndarray, brain: np.ndarray) -> np.ndarray:
    img = 2.0 * np.asarray(v, dtype=np.float32) - 1.0
    img[~brain] = -1.0
    return img


def from_model(img: np.ndarray, brain: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(img, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0) * brain.astype(np.float32)


def finalize_prediction(pred01: np.ndarray, brain: np.ndarray) -> np.ndarray:
    p = np.nan_to_num(np.asarray(pred01, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if p.shape != VOL_SHAPE:
        raise ValueError(f"prediction shape {p.shape} != {VOL_SHAPE}")
    return (np.clip(p, 0.0, 1.0) * brain.astype(np.float32)).astype(np.float16)


def save_volume(path: Path, vol: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.npy")
    np.save(tmp, vol)
    os.replace(tmp, path)
    return path


def write_nifti(vol: np.ndarray, path: Path, dtype=np.float32, affine: np.ndarray = BRATS_AFFINE) -> Path:
    import nibabel as nib

    img = nib.Nifti1Image(np.asarray(vol).astype(dtype, copy=False), affine)
    img.header.set_data_dtype(dtype)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def brain_bbox(brain: np.ndarray, margin: Sequence[int] = (0, 0, 0)):
    idx = np.nonzero(brain)
    if len(idx[0]) == 0:
        return tuple(slice(0, s) for s in brain.shape)
    out = []
    for ax in range(brain.ndim):
        lo = max(int(idx[ax].min()) - int(margin[ax]), 0)
        hi = min(int(idx[ax].max()) + 1 + int(margin[ax]), brain.shape[ax])
        out.append(slice(lo, hi))
    return tuple(out)


def list_cases(root: Path, cases: Optional[Sequence[str]] = None) -> list:
    root = Path(root)
    found = sorted(p.name for p in root.iterdir() if (p / "brain.npy").exists())
    if cases:
        missing = sorted(set(cases) - set(found))
        if missing:
            raise FileNotFoundError(f"cases not in {root}: {missing[:5]}")
        return [c for c in found if c in set(cases)]
    return found
