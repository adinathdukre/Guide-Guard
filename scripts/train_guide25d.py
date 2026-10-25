import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.guide25d import REG_T, alphas_999, build_from_base, make_slab_sampler, set_sdpa, stack_prior, upstream
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import ALL_MODS, load_case


class SlabStream(IterableDataset):
    def __init__(self, cases, vol_root, maps, sampler, samples_per_volume, buffer, seed, prior):
        super().__init__()
        self.cases, self.vol_root, self.maps = list(cases), Path(vol_root), Path(maps)
        self.sampler, self.spv, self.buffer, self.seed, self.prior = sampler, samples_per_volume, buffer, seed, prior

    def load(self, case):
        c = load_case(self.vol_root / case, ALL_MODS, with_seg=True)
        if self.prior:
            c["prior"] = stack_prior(load_probs(self.maps / f"{case}.npz"), c["brain"])
        return c

    def __iter__(self):
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        rng = np.random.default_rng([self.seed, 1_000_003, wid])
        buf, rnd = [], 0
        while True:
            perm = np.random.default_rng([self.seed, rnd]).permutation(len(self.cases))
            for ci in perm[wid::nw]:
                for s in self.sampler(self.load(self.cases[ci]), rng, self.spv):
                    buf.append(s)
                    if len(buf) >= self.buffer:
                        yield buf.pop(int(rng.integers(len(buf))))
            rnd += 1


def collate(samples):
    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in samples[0]}


def lr_at(step, max_steps, warmup, lr, lr_min):
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    prog = min(max(step - warmup, 0) / max(max_steps - warmup, 1), 1.0)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * prog))


def seg_loss(logits, lab, smooth=1.0):
    p = torch.sigmoid(logits.float())
    inter = (p * lab).sum((0, 2, 3))
    dice = (2.0 * inter + smooth) / (p.sum((0, 2, 3)) + lab.sum((0, 2, 3)) + smooth)
    return (1.0 - dice).mean() + F.binary_cross_entropy_with_logits(logits.float(), lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--maps", required=True, type=Path, help="h's maps: out-of-fold for every training case")
    ap.add_argument("--split_csv", required=True, type=Path)
    ap.add_argument("--base_ckpt", required=True, help="YODA regression checkpoint (20 inputs, 1 output, EMA)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no_prior", action="store_true", help="control arm without the lesion maps")
    ap.add_argument("--prior_dropout", type=float, default=0.15)
    ap.add_argument("--max_steps", type=int, default=25_000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_min", type=float, default=1e-6)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--flip_p", type=float, default=0.5)
    ap.add_argument("--seg_weight", type=float, default=0.005)
    ap.add_argument("--ema_beta", type=float, default=0.999)
    ap.add_argument("--snap_every", type=int, default=5_000)
    ap.add_argument("--samples_per_volume", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    prior = not a.no_prior
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device)
    set_sdpa(True)
    a.out.mkdir(parents=True, exist_ok=True)
    model, cfg = build_from_base(a.base_ckpt, prior, seed=a.seed)
    model = model.to(dev).train()
    thick, img = int(cfg.data.slice_thickness), int(cfg.data.img_size[0])
    sa, sb = alphas_999(cfg)
    from ema_pytorch import EMA

    ema = EMA(model, beta=a.ema_beta, update_after_step=1000, update_every=1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr_at(0, a.max_steps, a.warmup, a.lr, a.lr_min))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    train = [r["case"] for r in csv.DictReader(open(a.split_csv)) if r["split"] == "train"]
    sampler = make_slab_sampler(list(cfg.data.guidance_sequences), img, thick, flip_p=a.flip_p, with_prior=prior,
                                with_labels=True)
    ds = SlabStream(train, a.vol_root, a.maps, sampler, a.samples_per_volume, 4 * a.batch_size, a.seed, prior)
    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, collate_fn=collate,
                    pin_memory=dev.type == "cuda")
    cfg_dict = upstream().OmegaConf.to_container(cfg, resolve=True)
    (a.out / "config.json").write_text(json.dumps(cfg_dict, indent=1))
    half = thick // 2
    step, micro, t0 = 0, 0, time.time()
    for b in dl:
        tgt, guid = b["tgt"].to(dev), b["guid"].to(dev)
        lab = b["lab"].to(dev)
        parts = [torch.randn_like(tgt), guid]
        if prior:
            p = b["prior"].to(dev)
            keep = (torch.rand(p.shape[0], device=dev) >= a.prior_dropout).to(p.dtype)
            parts.append(p * keep[:, None, None, None])
        noise = parts[0]
        t = torch.full((tgt.shape[0],), REG_T, device=dev, dtype=torch.long)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = model(torch.cat(parts, 1), t)
        out = out.float()
        x0 = sa * noise[:, half:half + 1] - sb * out[:, :1]
        loss = F.mse_loss(x0, tgt[:, half:half + 1]) + a.seg_weight * seg_loss(out[:, 1:4], lab)
        scaler.scale(loss / a.grad_accum).backward()
        micro += 1
        if micro % a.grad_accum:
            continue
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        ema.update()
        step += 1
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.max_steps, a.warmup, a.lr, a.lr_min)
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.5f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{time.time() - t0:.0f}s", flush=True)
        if step % a.snap_every == 0 or step >= a.max_steps:
            torch.save({"ema": ema.ema_model.state_dict(), "step": step, "config": cfg_dict,
                        "args": {"sdpa": 1, "prior": prior}}, a.out / f"snap_{step:06d}.pt")
        if step >= a.max_steps:
            break


if __name__ == "__main__":
    main()
