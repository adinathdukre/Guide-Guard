import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.nnunet_helpers import CHANNELS_S, load_model, save_probs
from guideguard.volumes import NC_MODS, list_cases, load_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path, help="synthetic T1c folder (<case>.npy, [0,1])")
    ap.add_argument("--ckpt", required=True, type=Path, help="checkpoint root with S/")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    S = load_model("S", a.ckpt / "S", a.device)
    for case in list_cases(a.vol_root, a.cases):
        out = a.out / f"{case}.npz"
        if out.exists() and not a.overwrite:
            continue
        v = load_case(a.vol_root / case, NC_MODS)
        v["t1c"] = np.load(a.images / f"{case}.npy").astype(np.float32)
        save_probs(out, S.predict({m: v[m] for m in CHANNELS_S}, v["brain"]), f"S_on_{a.images.name}")
        print(case, flush=True)


if __name__ == "__main__":
    main()
