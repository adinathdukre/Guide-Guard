from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .volumes import NC_MODS, TARGET_MOD, load_case

PRIOR_KEYS = ("p_wt", "p_tc", "p_et")
PRIOR_REGIONS = ("WT", "TC", "ET")
CHANNELS = NC_MODS + PRIOR_KEYS
CHANNEL_NAMES = {"0": "T1n", "1": "T2w", "2": "T2f", "3": "noNorm", "4": "noNorm", "5": "noNorm"}
EXPECTED_NORMALIZATION = ("ZScoreNormalization",) * 3 + ("NoNormalization",) * 3
LR_MIRROR_AXES = (2,)
PLANS = "nnResUNetPlans"
CONFIG = "3d_fullres"
TRAINER = "nnUNetTrainerGuide3D"
FOLD = 0
SELECTION_FILE = "selection.json"


def to_nn(vol_xyz: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(vol_xyz).transpose(2, 1, 0))


from_nn = to_nn


def add_prior(v: Dict[str, np.ndarray], probs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    brain = v["brain"]
    for r, k in zip(PRIOR_REGIONS, PRIOR_KEYS):
        p = np.asarray(probs[r], dtype=np.float32)
        v[k] = np.where(brain, np.clip(p, 0.0, 1.0), np.float32(0.0)).astype(np.float32)
    return v


def stack_inputs(v: Dict[str, np.ndarray], n_in: int = len(CHANNELS)) -> np.ndarray:
    return np.stack([to_nn(v[k]).astype(np.float32) for k in CHANNELS[:n_in]])


def read_selection(model_dir: Path, fold: int = FOLD) -> dict:
    p = Path(model_dir) / f"fold_{fold}" / SELECTION_FILE
    return json.loads(p.read_text()) if p.exists() else {"checkpoint": "checkpoint_final.pth", "step_size": 0.5}


class Guide3D:
    def __init__(self, model_dir: Path, checkpoint: str = "auto", device: str = "cuda", step_size: float = None,
                 tta: bool = True, fold: int = FOLD):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        sel = read_selection(model_dir, fold)
        checkpoint = sel["checkpoint"] if checkpoint == "auto" else checkpoint
        step_size = float(sel["step_size"]) if step_size is None else float(step_size)
        self.torch = torch
        self.pred = nnUNetPredictor(tile_step_size=step_size, use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=device != "cpu", device=torch.device(device),
                                    verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        self.pred.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint)
        self.pm, self.cm = self.pred.plans_manager, self.pred.configuration_manager
        self.pre = self.cm.preprocessor_class(verbose=False)
        self.n_in = len(self.pred.dataset_json["channel_names"])
        if tuple(self.cm.normalization_schemes) != EXPECTED_NORMALIZATION[: self.n_in]:
            raise RuntimeError(f"{model_dir}: normalisation {self.cm.normalization_schemes}")
        self.pred.use_mirroring = bool(tta)
        self.pred.allowed_mirroring_axes = LR_MIRROR_AXES if tta else None
        self.checkpoint, self.step_size = checkpoint, step_size

    @property
    def network(self):
        return self.pred.network

    def __call__(self, v: Dict[str, np.ndarray]) -> np.ndarray:
        x = stack_inputs(v, self.n_in)
        props = {"spacing": [1.0, 1.0, 1.0]}
        data, _ = self.pre.run_case_npy(x, None, props, self.pm, self.cm, self.pred.dataset_json)
        out = self.pred.predict_logits_from_preprocessed_data(self.torch.from_numpy(data).float())
        out = out[0].float().cpu().numpy()
        full = np.zeros(props["shape_before_cropping"], np.float32)
        full[tuple(slice(int(a), int(b)) for a, b in props["bbox_used_for_cropping"])] = out
        return from_nn(full.transpose(list(self.pm.transpose_backward)))


def write_nii(arr_zyx: np.ndarray, path: Path) -> None:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path), useCompression=True)


def write_raw_case(raw_ds: Path, case: str, v: Dict[str, np.ndarray]) -> None:
    for i, k in enumerate(CHANNELS):
        write_nii(to_nn(v[k]).astype(np.float32), raw_ds / "imagesTr" / f"{case}_{i:04d}.nii.gz")
    write_nii(to_nn(v["brain"]).astype(np.uint8), raw_ds / "labelsTr" / f"{case}.nii.gz")


def write_raw_dataset_json(raw_ds: Path, n: int) -> None:
    dj = {"channel_names": CHANNEL_NAMES, "labels": {"background": 0, "brain": 1}, "numTraining": n,
          "file_ending": ".nii.gz"}
    (raw_ds / "dataset.json").write_text(json.dumps(dj, indent=2))


def encode_code(brain: np.ndarray, seg: np.ndarray) -> np.ndarray:
    s = np.asarray(seg)
    code = (np.asarray(brain, bool).astype(np.int32) + 2 * (s > 0) + 4 * np.isin(s, (1, 3)) + 8 * (s == 3))
    return code.astype(np.float32)


def crop_to_grid(vol_xyz: np.ndarray, props: dict, transpose_forward) -> np.ndarray:
    t = to_nn(vol_xyz).transpose(list(transpose_forward))
    sl = tuple(slice(int(lo), int(hi)) for lo, hi in props["bbox_used_for_cropping"])
    return np.ascontiguousarray(t[sl], dtype=np.float32)


def inject_targets(pre_ds: Path, vol_root: Path) -> int:
    plans = json.loads((pre_ds / f"{PLANS}.json").read_text())
    tf = plans["transpose_forward"]
    folder = pre_ds / plans["configurations"][CONFIG]["data_identifier"]
    n = 0
    for npz in sorted(folder.glob("*.npz")):
        case = npz.name[:-4]
        props = pickle.load(open(npz.with_suffix(".pkl"), "rb"))
        with np.load(npz) as f:
            data = f["data"]
        c = load_case(Path(vol_root) / case, (TARGET_MOD,), with_seg=True)
        t1c = crop_to_grid(c[TARGET_MOD], props, tf)
        code = crop_to_grid(encode_code(c["brain"], c["seg"]), props, tf)
        if t1c.shape != data.shape[1:]:
            raise RuntimeError(f"{case}: target {t1c.shape} vs data {data.shape}")
        tmp = npz.with_name(case + ".part.npz")
        np.savez_compressed(tmp, data=data, seg=np.stack([t1c, code], 0))
        tmp.rename(npz)
        for stale in (npz.with_suffix(".npy"), npz.with_name(case + "_seg.npy")):
            stale.unlink(missing_ok=True)
        n += 1
    return n


def snapshot_candidates(model_dir: Path, fold: int = FOLD) -> Sequence[str]:
    d = Path(model_dir) / f"fold_{fold}"
    return sorted(p.name for p in d.glob("checkpoint_*.pth") if p.name != "checkpoint_latest.pth")
