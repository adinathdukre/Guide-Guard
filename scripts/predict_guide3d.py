import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.guide3d import Guide3D, add_prior
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import NC_MODS, finalize_prediction, list_cases, load_case, save_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--maps", required=True, type=Path, help="h's lesion maps (<maps>/<case>.npz)")
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--checkpoint", default="auto", help="auto = fold_0/selection.json")
    ap.add_argument("--step_size", type=float, default=None)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    net = Guide3D(a.model_dir, a.checkpoint, a.device, a.step_size, tta=not a.no_tta)
    for case in list_cases(a.vol_root, a.cases):
        out = a.out / f"{case}.npy"
        if out.exists() and not a.overwrite:
            continue
        t0 = time.time()
        v = load_case(a.vol_root / case, NC_MODS)
        if net.n_in > len(NC_MODS):
            v = add_prior(v, load_probs(a.maps / f"{case}.npz"))
        save_volume(out, finalize_prediction(net(v), v["brain"]))
        print(f"{case} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
