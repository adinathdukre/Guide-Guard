import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.volumes import load_raw_case, preprocess_case, save_case


def run(args):
    case_dir, out_root, with_target, with_seg = args
    case = case_dir.name
    raw, affine = load_raw_case(case_dir, case, with_target, with_seg)
    save_case(Path(out_root) / case, preprocess_case(raw), {"case": case, "affine": affine.tolist()})
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no_target", action="store_true")
    ap.add_argument("--no_seg", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dirs = sorted(p for p in a.raw_root.iterdir() if p.is_dir() and (p / f"{p.name}-t1n.nii.gz").exists())
    jobs = [(d, a.out, not a.no_target, not a.no_seg) for d in dirs]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, case in enumerate(ex.map(run, jobs)):
            print(f"{i + 1}/{len(jobs)} {case}", flush=True)


if __name__ == "__main__":
    main()
