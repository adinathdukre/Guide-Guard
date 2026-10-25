import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard import guard as G
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import list_cases, save_volume


def read_cases(p):
    return [l.strip() for l in open(p) if l.strip()] if p else None


def inputs(a, case):
    d = a.vol_root / case
    brain = np.load(d / "brain.npy").astype(bool)
    g = load_probs(a.maps / "g" / f"{case}.npz", brain)["ET"]
    if a.score == "avg":
        s = G.averaged_score(g, load_probs(a.maps / "h" / f"{case}.npz", brain)["ET"])
    else:
        s = g
    if a.reader is None:
        return brain, np.zeros_like(brain), s, d
    p_et = load_probs(a.reader / f"{case}.npz", brain)["ET"]
    return brain, G.ehat_from_prob(p_et, brain), s, d


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "apply", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--vol_root", required=True, type=Path)
        r = p.add_mutually_exclusive_group(required=True)
        r.add_argument("--reader", type=Path, help="S on the synthetic images (<case>.npz)")
        r.add_argument("--no_image", action="store_true", help="NC-only CRC: the image-free map (E_hat empty)")
        p.add_argument("--maps", required=True, type=Path, help="folder with g/ and h/ maps")
        p.add_argument("--score", choices=("g", "avg"), default="g",
                       help="g = the NC ET score g (primary), avg = the averaged score (g + m_ET)/2")
        p.add_argument("--cases", type=Path, default=None, help="text file, one case per line")
        if name == "calibrate":
            p.add_argument("--out", required=True, type=Path)
            p.add_argument("--alpha", type=float, default=0.1)
            p.add_argument("--beta", type=float, default=0.2)
            p.add_argument("--delta", type=float, default=0.1)
            p.add_argument("--variant", choices=("voxel", "lesion", "tail"), default="voxel")
            p.add_argument("--min_et", type=int, default=50)
        else:
            p.add_argument("--thresholds", required=True, type=Path)
            p.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cases = list_cases(a.vol_root, read_cases(a.cases))
    if a.cmd == "calibrate":
        lambdas, mus = G.logit_grid(), G.logit_grid()
        curves = []
        for case in cases:
            brain, ehat, s, d = inputs(a, case)
            curves.append(G.patient_curves(s, ehat, np.load(d / "seg.npy"), brain, lambdas, mus))
        thr = G.calibrate(curves, lambdas, mus, a.alpha, a.beta, a.delta, a.min_et, a.variant)
        thr.update(score=a.score, image=not a.no_image, calibration_cases=cases)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(thr, indent=1))
        print(json.dumps({k: v for k, v in thr.items() if k != "calibration_cases"}, indent=1))
        return
    thr = json.loads(a.thresholds.read_text())
    if thr.get("score", a.score) != a.score:
        raise SystemExit(f"thresholds were calibrated with score {thr['score']}, not {a.score}")
    if thr.get("image", True) == a.no_image:
        raise SystemExit("thresholds were calibrated " + ("with" if thr.get("image", True) else "without")
                         + " an image; pass " + ("--reader" if thr.get("image", True) else "--no_image"))
    overlap = sorted(set(thr.get("calibration_cases", [])) & set(cases))
    if overlap:
        print(f"warning: {len(overlap)} cases were used for calibration", file=sys.stderr)
    rows = []
    for case in cases:
        brain, ehat, s, d = inputs(a, case)
        m = G.apply_thresholds(s, ehat, brain, thr)
        if a.cmd == "apply":
            save_volume(a.out / f"{case}.npy", m)
        else:
            rows.append({"case": case, **G.evaluate_map(m, ehat, np.load(d / "seg.npy"), brain, thr["min_et"])})
    if a.cmd == "evaluate":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        miss = [r["miss"] for r in rows if "miss" in r]
        print(f"{len(rows)} cases; mean miss {np.mean(miss):.3f} over {len(miss)} ET-positive; mean review burden "
              f"{np.mean([r['review_burden_ml'] for r in rows]):.1f} mL")


if __name__ == "__main__":
    main()
