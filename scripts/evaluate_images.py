import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.guard import ET_LABEL, ehat_from_prob
from guideguard.metrics import dice, image_metrics
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import TARGET_MOD, list_cases, load_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path, help="synthetic T1c folder (<case>.npy, [0,1])")
    ap.add_argument("--reader", type=Path, default=None, help="S on the same images (<case>.npz), for ET Dice")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for case in list_cases(a.vol_root, a.cases):
        c = load_case(a.vol_root / case, (TARGET_MOD,), with_seg=True)
        pred = np.load(a.images / f"{case}.npy").astype(np.float32)
        r = {"case": case, **image_metrics(pred, c[TARGET_MOD], c["brain"], c.get("seg"))}
        if a.reader is not None:
            et_hat = ehat_from_prob(load_probs(a.reader / f"{case}.npz", c["brain"])["ET"], c["brain"])
            r["dice_ET"] = dice(et_hat, (c["seg"] == ET_LABEL) & c["brain"])
        rows.append(r)
        print(case, {k: round(v, 4) for k, v in r.items() if k != "case"}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases; " + ", ".join(f"mean {k} {np.nanmean([r[k] for r in rows]):.4f}" for k in keys[1:]))


if __name__ == "__main__":
    main()
