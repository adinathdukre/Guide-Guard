import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.guide25d import (case_seed, combine_passes, has_prior, load_guide25d, predict_passes, set_sdpa,
                                 stack_prior)
from guideguard.metrics import psnr_region
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import ALL_MODS, from_model, load_case


def score(net, cfg, cases, a, dev, tta):
    use_prior = has_prior(cfg)
    out = []
    for case in cases:
        c = load_case(a.vol_root / case, ALL_MODS)
        if use_prior:
            c["prior"] = stack_prior(load_probs(a.maps / f"{case}.npz"), c["brain"])
        y = c.pop("t1c")
        passes = predict_passes(net, cfg, c, dev, tta=tta, seed=case_seed(case, 0))
        out.append(psnr_region(from_model(combine_passes(passes), c["brain"]), y, c["brain"]))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--maps", required=True, type=Path)
    ap.add_argument("--split_csv", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    set_sdpa(True)
    val = [r["case"] for r in csv.DictReader(open(a.split_csv)) if r["split"] == "val"]
    snaps = sorted(a.run_dir.glob("snap_*.pt"))
    cells = []
    for p in snaps:
        net, cfg, _ = load_guide25d(str(p))
        r = score(net.to(dev), cfg, val, a, dev, tta=False)
        cells.append({"snapshot": p.name, "tta": 0, "psnr_brain": r})
        print(p.name, r, flush=True)
    best = max(cells, key=lambda r: r["psnr_brain"])
    net, cfg, _ = load_guide25d(str(a.run_dir / best["snapshot"]))
    r = score(net.to(dev), cfg, val, a, dev, tta=True)
    cells.append({"snapshot": best["snapshot"], "tta": 1, "psnr_brain": r})
    final = max([c for c in cells if c["snapshot"] == best["snapshot"]], key=lambda r: r["psnr_brain"])
    (a.run_dir / "selection.json").write_text(json.dumps({"final": final, "cells": cells}, indent=1))
    print("selected", final)


if __name__ == "__main__":
    main()
