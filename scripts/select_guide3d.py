import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.guide3d import FOLD, SELECTION_FILE, Guide3D, add_prior, snapshot_candidates
from guideguard.metrics import psnr_region
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import ALL_MODS, finalize_prediction, load_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path)
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--maps", required=True, type=Path)
    ap.add_argument("--split_csv", required=True, type=Path)
    ap.add_argument("--step_sizes", nargs="+", type=float, default=[0.5, 0.3])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    val = [r["case"] for r in csv.DictReader(open(a.split_csv)) if r["split"] == "val"]
    data = {}
    for case in val:
        v = add_prior(load_case(a.vol_root / case, ALL_MODS), load_probs(a.maps / f"{case}.npz"))
        data[case] = (v.pop("t1c"), v)

    def run(ck, step):
        net = Guide3D(a.model_dir, ck, a.device, step, tta=False)
        return float(np.mean([psnr_region(finalize_prediction(net(v), v["brain"]).astype(np.float32), y, v["brain"])
                              for y, v in data.values()]))

    rows = [{"checkpoint": ck, "step_size": a.step_sizes[0], "psnr_brain": run(ck, a.step_sizes[0])}
            for ck in snapshot_candidates(a.model_dir)]
    best = max(rows, key=lambda r: r["psnr_brain"])
    rows += [{"checkpoint": best["checkpoint"], "step_size": s, "psnr_brain": run(best["checkpoint"], s)}
             for s in a.step_sizes[1:]]
    best = max(rows, key=lambda r: r["psnr_brain"])
    p = a.model_dir / f"fold_{FOLD}" / SELECTION_FILE
    p.write_text(json.dumps({**best, "cells": rows}, indent=1))
    print("selected", best)


if __name__ == "__main__":
    main()
