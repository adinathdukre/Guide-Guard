import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.guide25d import has_prior, load_guide25d, predict_volume, set_sdpa, stack_prior
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import NC_MODS, finalize_prediction, list_cases, load_case, save_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--maps", required=True, type=Path, help="h's lesion maps (<maps>/<case>.npz)")
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--no_tta", action="store_true", help="Guide-fast: no left-right flip test-time augmentation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    s = torch.load(a.ckpt, map_location="cpu")
    set_sdpa(bool(s.get("args", {}).get("sdpa", 1)))
    net, cfg, _ = load_guide25d(str(a.ckpt))
    net = net.to(dev).eval()
    use_prior = has_prior(cfg)
    for case in list_cases(a.vol_root, a.cases):
        out = a.out / f"{case}.npy"
        if out.exists() and not a.overwrite:
            continue
        t0 = time.time()
        c = load_case(a.vol_root / case, NC_MODS)
        if use_prior:
            c["prior"] = stack_prior(load_probs(a.maps / f"{case}.npz"), c["brain"])
        pred = predict_volume(net, cfg, c, dev, case, tta=not a.no_tta, seed=a.seed,
                              batch_size=a.batch_size)
        save_volume(out, finalize_prediction(pred, c["brain"]))
        print(f"{case} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
