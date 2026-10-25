import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.nnunet_helpers import MODELS, load_cv_npz, load_model, save_probs
from guideguard.volumes import list_cases, load_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path, help="checkpoint root with h/ and g/")
    ap.add_argument("--out", required=True, type=Path, help="writes <out>/h/<case>.npz and <out>/g/<case>.npz")
    ap.add_argument("--models", nargs="+", choices=("h", "g"), default=["h", "g"])
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--h_folds", nargs="+", type=int, default=None, help="subset of h's folds (Guide-fast: 0)")
    ap.add_argument("--h_no_mirroring", action="store_true", help="Guide-fast: h without mirroring")
    ap.add_argument("--from_cv", type=Path, default=None,
                    help="training cases: h's out-of-fold maps from nnU-Net CV outputs (<dir>/fold_k/validation)")
    ap.add_argument("--folds_csv", type=Path, default=None, help="case,fold table of h's CV (with --from_cv)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cases = list_cases(a.vol_root, a.cases)
    if a.from_cv is not None:
        import csv

        fold = {r["case"]: int(r["fold"]) for r in csv.DictReader(open(a.folds_csv))}
        for case in cases:
            out = a.out / "h" / f"{case}.npz"
            if out.exists() and not a.overwrite:
                continue
            k = fold[case]
            save_probs(out, load_cv_npz(a.from_cv / f"fold_{k}" / "validation" / f"{case}.npz"), f"oof_fold{k}")
            print(case, f"oof_fold{k}", flush=True)
        return
    for kind in a.models:
        kw = {}
        if kind == "h":
            kw = {"folds": a.h_folds, "use_mirroring": False if a.h_no_mirroring else None}
        net = load_model(kind, a.ckpt / kind, a.device, **kw)
        folds = a.h_folds if kind == "h" and a.h_folds else MODELS[kind]["folds"]
        src = f"{kind}_f{''.join(map(str, folds))}" + ("_nomirror" if kind == "h" and a.h_no_mirroring else "")
        for case in cases:
            out = a.out / kind / f"{case}.npz"
            if out.exists() and not a.overwrite:
                continue
            v = load_case(a.vol_root / case, MODELS[kind]["channels"])
            save_probs(out, net.predict({m: v[m] for m in MODELS[kind]["channels"]}, v["brain"]), src)
            print(kind, case, flush=True)


if __name__ == "__main__":
    main()
