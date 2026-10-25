from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .volumes import VOL_SHAPE, load_case, write_nifti

REGION_NAMES = ("WT", "TC", "ET")
LABELS = {"background": 0, "WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}
REGIONS_CLASS_ORDER = [2, 1, 3]
CHANNEL_NAMES = {"t1n": "T1n", "t1c": "T1c", "t2w": "T2w", "t2f": "T2f"}
CHANNELS_NC = ("t1n", "t2w", "t2f")
CHANNELS_S = ("t1n", "t1c", "t2w", "t2f")
FILE_ENDING = ".nii.gz"
TRAINER = "nnUNetTrainer_250epochs"
PLANS = "nnUNetPlans"
N_FOLDS_H = 5
FOLD_G = 5
FOLD_S = 0
MODELS = {
    "h": {"config": "2d", "folds": tuple(range(N_FOLDS_H)), "channels": CHANNELS_NC, "mirroring": True},
    "g": {"config": "3d_fullres", "folds": (FOLD_G,), "channels": CHANNELS_NC, "mirroring": False},
    "S": {"config": "3d_fullres", "folds": (FOLD_S,), "channels": CHANNELS_S, "mirroring": False},
}


def vol_to_nnunet(vol: np.ndarray) -> np.ndarray:
    if vol.ndim == 3:
        return np.ascontiguousarray(vol.transpose(2, 1, 0))
    if vol.ndim == 4:
        return np.ascontiguousarray(vol.transpose(0, 3, 2, 1))
    raise ValueError(vol.shape)


def nnunet_to_vol(arr: np.ndarray) -> np.ndarray:
    return vol_to_nnunet(arr)


def prepare_input(vols: Dict[str, np.ndarray], brain: np.ndarray, channels: Sequence[str]) -> np.ndarray:
    br = np.asarray(brain).astype(bool)
    out = []
    for m in channels:
        a = np.asarray(vols[m], dtype=np.float32)
        a = np.where(br, np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0), 0.0).astype(np.float32)
        out.append(vol_to_nnunet(a))
    return np.stack(out)


class VolumePredictor:
    def __init__(self, predictor, channels: Sequence[str]):
        self.predictor = predictor
        self.channels = tuple(channels)
        dj = predictor.dataset_json
        if tuple(k for k in dj["labels"] if k != "background") != REGION_NAMES:
            raise AssertionError(f"unexpected labels {dj['labels']}")
        got = [dj["channel_names"][str(i)] for i in range(len(dj["channel_names"]))]
        if got != [CHANNEL_NAMES[c] for c in self.channels]:
            raise AssertionError(f"channel order {got}")

    @classmethod
    def from_trained(cls, model_dir, folds: Sequence, channels: Sequence[str], checkpoint: str = "checkpoint_final.pth",
                     device: Optional[str] = None, use_mirroring: bool = False, tile_step_size: float = 0.5):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        pred = nnUNetPredictor(tile_step_size=tile_step_size, use_gaussian=True, use_mirroring=use_mirroring,
                               perform_everything_on_device=dev.type == "cuda", device=dev, verbose=False,
                               verbose_preprocessing=False, allow_tqdm=False)
        pred.initialize_from_trained_model_folder(str(model_dir), use_folds=tuple(folds), checkpoint_name=checkpoint)
        return cls(pred, channels)

    def predict(self, vols: Dict[str, np.ndarray], brain: np.ndarray) -> Dict[str, np.ndarray]:
        import torch
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

        p = self.predictor
        arr = prepare_input(vols, brain, self.channels)
        props = {"spacing": [1.0, 1.0, 1.0]}
        pp = p.configuration_manager.preprocessor_class(verbose=False)
        data, _, props = pp.run_case_npy(arr, None, props, p.plans_manager, p.configuration_manager, p.dataset_json)
        logits = p.predict_logits_from_preprocessed_data(torch.from_numpy(np.ascontiguousarray(data)).float()).cpu()
        _, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, p.plans_manager, p.configuration_manager, p.label_manager, props, return_probabilities=True)
        out = nnunet_to_vol(np.asarray(probs, dtype=np.float32))
        return {r: out[i] for i, r in enumerate(REGION_NAMES)}


def load_model(kind: str, model_dir, device: Optional[str] = None, folds: Optional[Sequence[int]] = None,
               use_mirroring: Optional[bool] = None) -> VolumePredictor:
    m = MODELS[kind]
    return VolumePredictor.from_trained(model_dir, folds if folds is not None else m["folds"], m["channels"],
                                        device=device,
                                        use_mirroring=m["mirroring"] if use_mirroring is None else use_mirroring)


def save_probs(path: Path, probs: Dict[str, np.ndarray], source: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **{r: np.asarray(probs[r]).astype(np.float16) for r in REGION_NAMES})
    os.replace(tmp, path)
    path.with_suffix(".json").write_text(json.dumps({"case": path.stem, "source": source}))
    return path


def load_probs(path: Path, brain: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {r: np.clip(np.asarray(z[r], dtype=np.float32), 0.0, 1.0) for r in REGION_NAMES}
    if brain is not None:
        for r in out:
            out[r] *= brain.astype(np.float32)
    return out


def load_cv_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        p = f["probabilities"]
    if p.ndim != 4 or p.shape[0] != len(REGION_NAMES):
        raise ValueError(f"{path}: expected (3,Z,Y,X), got {p.shape}")
    v = nnunet_to_vol(p.astype(np.float32, copy=False))
    if tuple(v.shape[1:]) != VOL_SHAPE:
        raise ValueError(f"{path}: shape {v.shape[1:]}")
    return {r: v[i] for i, r in enumerate(REGION_NAMES)}


def dataset_json(name: str, channels: Sequence[str], num_training: int) -> dict:
    return {"channel_names": {str(i): CHANNEL_NAMES[c] for i, c in enumerate(channels)}, "labels": dict(LABELS),
            "regions_class_order": list(REGIONS_CLASS_ORDER), "numTraining": int(num_training),
            "file_ending": FILE_ENDING, "overwrite_image_reader_writer": "SimpleITKIO", "name": name}


def write_raw_dataset(raw_root: Path, name: str, channels: Sequence[str], vol_root: Path,
                      train_cases: Sequence[str], test_cases: Sequence[str] = ()) -> Path:
    d = Path(raw_root) / name
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for case, tr in [(c, True) for c in train_cases] + [(c, False) for c in test_cases]:
        v = load_case(Path(vol_root) / case, tuple(channels), with_seg=tr)
        for i, m in enumerate(channels):
            write_nifti(np.where(v["brain"], v[m], 0.0), d / ("imagesTr" if tr else "imagesTs") /
                        f"{case}_{i:04d}{FILE_ENDING}", np.float32)
        if tr:
            write_nifti(v["seg"], d / "labelsTr" / f"{case}{FILE_ENDING}", np.uint8)
    (d / "dataset.json").write_text(json.dumps(dataset_json(name, channels, len(train_cases)), indent=1))
    return d


def cv_splits(folds: Dict[str, int], train: Sequence[str], val: Sequence[str], n_folds: int = N_FOLDS_H) -> List[dict]:
    out = [{"train": sorted(c for c in train if folds[c] != f), "val": sorted(c for c in train if folds[c] == f)}
           for f in range(n_folds)]
    out.append({"train": sorted(train), "val": sorted(val)})
    return out


def write_splits(preprocessed_root: Path, name: str, splits: List[dict]) -> Path:
    p = Path(preprocessed_root) / name / "splits_final.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(splits, indent=1))
    return p
