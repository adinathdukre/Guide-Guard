import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.fuse import fuse
from guideguard.volumes import list_cases, save_volume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--member", nargs="+", required=True, type=Path, help="prediction folders")
    ap.add_argument("--weights", nargs="+", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    for case in list_cases(a.vol_root, a.cases):
        brain = np.load(a.vol_root / case / "brain.npy").astype(bool)
        save_volume(a.out / f"{case}.npy", fuse([np.load(d / f"{case}.npy") for d in a.member], a.weights, brain))
    print(f"fused -> {a.out}")


if __name__ == "__main__":
    main()
