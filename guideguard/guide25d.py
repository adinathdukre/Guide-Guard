from __future__ import annotations

import copy
import math
import os
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .volumes import TARGET_MOD, brain_bbox, from_model, to_model

N_PRIOR = 3
N_SEG = 3
SEG_INIT_STD = 0.01
REG_T = 999
SLICE_THICKNESS = 5
VIEWS = ("axial", "coronal", "sagittal")
VIEW_PERM = {"axial": (0, 1, 2), "coronal": (1, 0, 2), "sagittal": (2, 0, 1)}
PERM = {"axial": (0, 1, 2, 3), "coronal": (0, 2, 1, 3), "sagittal": (0, 3, 1, 2)}
INV = {"axial": (0, 1, 2, 3), "coronal": (0, 2, 1, 3), "sagittal": (0, 2, 3, 1)}
PRIOR_KEYS = ("WT", "TC", "ET")
LR_AXIS = 0

_M: Dict[str, object] = {}
_ORIG: Dict[str, object] = {}


def yoda_root() -> Path:
    root = os.environ.get("YODA_ROOT")
    if not root:
        raise RuntimeError("set YODA_ROOT to a clone of https://github.com/Deep-MI/YODA (commit 86e8dc8)")
    return Path(root)


def upstream():
    if "ns" in _M:
        return _M["ns"]
    root = str(yoda_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from generative.networks.nets import DiffusionModelUNet
        from lib.utils.monai_helper import scheduler_factory
    from omegaconf import OmegaConf

    _M["ns"] = SimpleNamespace(DiffusionModelUNet=DiffusionModelUNet, scheduler_factory=scheduler_factory,
                               OmegaConf=OmegaConf, root=Path(root))
    return _M["ns"]


def yoda_config(n_sources: int = 3):
    ns = upstream()
    OC = ns.OmegaConf
    cfg = OC.merge(OC.load(ns.root / "configs" / "defaults.yml"),
                   OC.load(ns.root / "configs" / "datasets" / "brats_train.yml"))
    guid = ["t1n", "t2w", "t2f"][:n_sources]
    cfg.data.target_sequence = "t1c"
    cfg.data.guidance_sequences = guid
    cfg.data.dataset = None
    cfg.data.data_dir = None
    cfg.model.unet.in_channels = cfg.data.slice_thickness * (1 + len(guid))
    cfg.model.unet.use_flash_attention = False
    return cfg


def build_unet(cfg) -> torch.nn.Module:
    ns = upstream()
    return ns.DiffusionModelUNet(**ns.OmegaConf.to_container(cfg.model.unet, resolve=True))


def train_scheduler(cfg):
    ns = upstream()
    sc = copy.deepcopy(ns.OmegaConf.to_container(cfg.model.train_noise_sched, resolve=True))
    sched = ns.scheduler_factory(ns.OmegaConf.create(sc))
    sched.set_timesteps(int(cfg.model.train_noise_sched.num_train_timesteps))
    return sched


def _sdpa_attention_block(self, query, key, value):
    out = F.scaled_dot_product_attention(query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1), scale=self.scale)
    return out.squeeze(1)


def _sdpa_cross_attention(self, query, key, value):
    if self.upcast_attention:
        return _ORIG["CrossAttention"](self, query, key, value)
    out = F.scaled_dot_product_attention(query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1), scale=self.scale)
    return out.squeeze(1)


def set_sdpa(enabled: bool) -> None:
    upstream()
    from generative.networks.nets import diffusion_model_unet as dmu

    if "AttentionBlock" not in _ORIG:
        _ORIG["AttentionBlock"] = dmu.AttentionBlock._attention
        _ORIG["CrossAttention"] = dmu.CrossAttention._attention
    dmu.AttentionBlock._attention = _sdpa_attention_block if enabled else _ORIG["AttentionBlock"]
    dmu.CrossAttention._attention = _sdpa_cross_attention if enabled else _ORIG["CrossAttention"]


def alphas_999(cfg) -> Tuple[float, float]:
    sched = train_scheduler(cfg)
    ab = float(sched.alphas_cumprod[REG_T])
    return math.sqrt(ab), math.sqrt(1.0 - ab)


def guide_config(base_cfg, prior: bool):
    ns = upstream()
    cfg = ns.OmegaConf.create(ns.OmegaConf.to_container(base_cfg, resolve=True))
    cfg.model.unet.in_channels = int(cfg.model.unet.in_channels) + (N_PRIOR * int(cfg.data.slice_thickness)
                                                                    if prior else 0)
    cfg.model.unet.out_channels = 1 + N_SEG
    return cfg


def expand_state_dict(base_sd, target_sd, seed: int = 0):
    g = torch.Generator().manual_seed(int(seed))
    out = {}
    for k, tgt in target_sd.items():
        src = base_sd[k]
        if src.shape == tgt.shape:
            out[k] = src.clone()
        elif k == "conv_in.conv.weight":
            w = torch.zeros_like(tgt)
            w[:, : src.shape[1]] = src
            out[k] = w
        elif k == "out.2.conv.weight":
            w = torch.randn(tgt.shape, generator=g, dtype=torch.float32).to(tgt.dtype) * SEG_INIT_STD
            w[: src.shape[0]] = src
            out[k] = w
        elif k == "out.2.conv.bias":
            b = torch.zeros_like(tgt)
            b[: src.shape[0]] = src
            out[k] = b
        else:
            raise ValueError(f"unexpected shape change for {k}: {tuple(src.shape)} -> {tuple(tgt.shape)}")
    extra = sorted(set(base_sd) - set(target_sd))
    if extra:
        raise KeyError(f"pretrained keys without a target: {extra[:5]}")
    return out


def build_from_base(base_ckpt: str, prior: bool = True, seed: int = 0):
    ns = upstream()
    s = torch.load(base_ckpt, map_location="cpu")
    base_cfg = ns.OmegaConf.create(s["config"])
    cfg = guide_config(base_cfg, prior)
    net = build_unet(cfg)
    net.load_state_dict(expand_state_dict(s["ema"], net.state_dict(), seed), strict=True)
    return net, cfg


def load_guide25d(path: str, map_location="cpu"):
    ns = upstream()
    s = torch.load(path, map_location=map_location)
    cfg = ns.OmegaConf.create(s["config"])
    net = build_unet(cfg)
    net.load_state_dict(s["ema"], strict=True)
    return net, cfg, s


def has_prior(cfg) -> bool:
    thick = int(cfg.data.slice_thickness)
    extra = int(cfg.model.unet.in_channels) - thick * (1 + len(cfg.data.guidance_sequences))
    if extra not in (0, N_PRIOR * thick):
        raise ValueError(f"in_channels {cfg.model.unet.in_channels} is neither the plain nor the prior network")
    return extra > 0


def to_dhw(v: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(v).transpose(2, 1, 0))


from_dhw = to_dhw


def pad_or_crop_2d(x: np.ndarray, size: int, value: float) -> np.ndarray:
    out = x
    for ax in (-2, -1):
        n = out.shape[ax]
        if n < size:
            d = size - n
            pad = [(0, 0)] * out.ndim
            pad[ax] = (d // 2, d - d // 2)
            out = np.pad(out, pad, constant_values=value)
        elif n > size:
            s = (n - size) // 2
            sl = [slice(None)] * out.ndim
            sl[ax] = slice(s, s + size)
            out = out[tuple(sl)]
    return out


def flip_lr(c: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in c.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            out[k] = np.flip(v, LR_AXIS)
        elif isinstance(v, np.ndarray) and v.ndim == 4:
            out[k] = np.flip(v, LR_AXIS + 1)
        else:
            out[k] = v
    return out


def stack_prior(probs: Dict[str, np.ndarray], brain: np.ndarray) -> np.ndarray:
    p = np.stack([np.asarray(probs[k], dtype=np.float32) for k in PRIOR_KEYS], 0)
    np.clip(p, 0.0, 1.0, out=p)
    p *= brain.astype(np.float32)[None]
    return p


def prepare_case(c: Dict[str, np.ndarray], guidance: Sequence[str], thickness: int = SLICE_THICKNESS,
                 with_target: bool = True) -> Dict[str, np.ndarray]:
    brain = c["brain"]
    bd = to_dhw(brain)
    box = brain_bbox(bd, margin=(thickness // 2, 0, 0))
    out = {"guid": np.stack([to_dhw(to_model(c[m], brain))[box] for m in guidance], 0), "brain": bd[box], "box": box}
    if with_target:
        out["tgt"] = to_dhw(to_model(c[TARGET_MOD], brain))[box]
    if "prior" in c:
        out["prior"] = np.stack([to_dhw(c["prior"][k])[box] for k in range(c["prior"].shape[0])], 0).astype(np.float32)
    if "seg" in c:
        s = to_dhw(c["seg"])[box]
        out["lab"] = np.stack([(s >= 1) & (s <= 3), (s == 1) | (s == 3), s == 3], 0).astype(np.uint8)
    return out


def make_slab_sampler(guidance: Sequence[str], img_size: int = 224, thickness: int = SLICE_THICKNESS,
                      flip_p: float = 0.0, with_prior: bool = False, with_labels: bool = False):
    a = thickness // 2

    def sampler(c, rng, n) -> List[Dict[str, np.ndarray]]:
        flip = bool(flip_p > 0 and rng.random() < flip_p)
        p = prepare_case(flip_lr(c) if flip else c, guidance, thickness)
        out = []
        for _ in range(n):
            view = VIEWS[int(rng.choice(3, p=np.full(3, 1 / 3)))]
            perm = VIEW_PERM[view]
            cperm = (0,) + tuple(x + 1 for x in perm)
            tgt, gd, br = p["tgt"].transpose(perm), p["guid"].transpose(cperm), p["brain"].transpose(perm)
            s = int(rng.integers(0, max(tgt.shape[0] - thickness, 0) + 1))
            sl = slice(s, s + thickness)
            t, g, b = tgt[sl], gd[:, sl], br[sl]
            smp = {"tgt": pad_or_crop_2d(t, img_size, -1.0).astype(np.float32),
                   "guid": pad_or_crop_2d(g.reshape((-1,) + g.shape[2:]), img_size, -1.0).astype(np.float32),
                   "brain": pad_or_crop_2d(b[a][None].astype(np.float32), img_size, 0.0)}
            if with_prior:
                pr = p["prior"].transpose(cperm)[:, sl]
                smp["prior"] = pad_or_crop_2d(pr.reshape((-1,) + pr.shape[2:]), img_size, 0.0).astype(np.float32)
            if with_labels:
                smp["lab"] = pad_or_crop_2d(p["lab"].transpose(cperm)[:, sl][:, a].astype(np.float32), img_size, 0.0)
            out.append(smp)
        return out

    return sampler


def case_seed(case: str, seed: int = 0) -> int:
    return int(seed) * 1000003 + zlib.crc32(case.encode())


@torch.no_grad()
def regress_view(model, guid: torch.Tensor, prior: Optional[torch.Tensor], sa: float, sb: float, img_size: int,
                 thickness: int = SLICE_THICKNESS, batch_size: int = 16, dtype=torch.float16) -> torch.Tensor:
    dev = guid.device
    C, D, H, W = guid.shape
    S, half = int(img_size), thickness // 2
    oh, ow = (S - H) // 2, (S - W) // 2
    noise = torch.randn(D, 1, S, S, device=dev).to(dtype)
    parts = [torch.full((D + 2 * half, 1, S, S), -1.0, dtype=dtype, device=dev)]
    parts[0][half:half + D] = noise
    gp = torch.full((D + 2 * half, C, S, S), -1.0, dtype=dtype, device=dev)
    gp[half:half + D, :, oh:oh + H, ow:ow + W] = guid.transpose(0, 1).to(dtype)
    parts.append(gp)
    if prior is not None:
        pp = torch.zeros((D + 2 * half, prior.shape[0], S, S), dtype=dtype, device=dev)
        pp[half:half + D, :, oh:oh + H, ow:ow + W] = prior.transpose(0, 1).to(dtype)
        parts.append(pp)
    X = torch.cat(parts, 1)
    out = torch.empty((D, S, S), dtype=torch.float32, device=dev)
    off = torch.arange(thickness, device=dev)
    t = torch.full((1,), REG_T, device=dev, dtype=torch.long)
    use_amp = dev.type == "cuda" or dtype == torch.bfloat16
    for i in range(0, D, batch_size):
        idx = torch.arange(i, min(i + batch_size, D), device=dev)
        slab = X[idx[:, None] + off[None]].transpose(1, 2).reshape(idx.numel(), -1, S, S)
        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=use_amp):
            o = model(slab, t)
        out[idx] = torch.clamp(sa * noise[idx, 0].float() - sb * o[:, 0].float(), -1.0, 1.0)
    return out[:, oh:oh + H, ow:ow + W]


def combine_passes(preds: Sequence[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(p, dtype=np.float32) for p in preds], 0).mean(0)


@torch.no_grad()
def predict_passes(model, cfg, c: Dict[str, np.ndarray], dev: torch.device, tta: bool = False,
                   views: Sequence[str] = VIEWS, batch_size: int = 16, seed: int = 0, dtype=None) -> List[np.ndarray]:
    guid_names = list(cfg.data.guidance_sequences)
    thick = int(cfg.data.slice_thickness)
    S = int(cfg.data.img_size[0])
    use_prior = has_prior(cfg)
    if use_prior and "prior" not in c:
        raise KeyError("prior network but the case has no 'prior'")
    dtype = dtype or (torch.float16 if dev.type == "cuda" else torch.float32)
    sa, sb = alphas_999(cfg)
    model.eval()
    prepared = []
    for flip in ((False, True) if tta else (False,)):
        cc = flip_lr(c) if flip else c
        if not use_prior:
            cc = {k: v for k, v in cc.items() if k != "prior"}
        p = prepare_case({k: v for k, v in cc.items() if k != "seg"}, guid_names, thick, with_target=False)
        p["flip"] = flip
        prepared.append(p)
    torch.manual_seed(seed)
    shape_dhw = tuple(c["brain"].shape[::-1])
    out: List[np.ndarray] = []
    for p in prepared:
        g = torch.from_numpy(p["guid"]).float().to(dev)
        pr = torch.from_numpy(p["prior"]).float().to(dev) if use_prior else None
        for v in views:
            x0 = regress_view(model, g.permute(*PERM[v]).contiguous(),
                              pr.permute(*PERM[v]).contiguous() if pr is not None else None,
                              sa, sb, S, thick, batch_size, dtype)
            x0 = x0.unsqueeze(0).permute(*INV[v])[0].cpu().numpy()
            full = -np.ones(shape_dhw, np.float32)
            full[p["box"]] = x0
            full = from_dhw(full)
            if p["flip"]:
                full = np.ascontiguousarray(np.flip(full, 0))
            out.append(full)
    return out


def predict_volume(model, cfg, c: Dict[str, np.ndarray], dev: torch.device, case: str, tta: bool = True,
                   seed: int = 0, batch_size: int = 16) -> np.ndarray:
    passes = predict_passes(model, cfg, c, dev, tta=tta, batch_size=batch_size, seed=case_seed(case, seed))
    return from_model(combine_passes(passes), c["brain"])
