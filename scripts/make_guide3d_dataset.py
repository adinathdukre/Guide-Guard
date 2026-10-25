import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.guide3d import add_prior, inject_targets, write_raw_case, write_raw_dataset_json
from guideguard.nnunet_helpers import load_probs
from guideguard.volumes import NC_MODS, load_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("raw", "splits", "inject"))
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--maps", type=Path, help="h's maps: out-of-fold for training cases, ensemble for val")
    ap.add_argument("--split_csv", required=True, type=Path)
    ap.add_argument("--dataset", default="Dataset622_Guide3D")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.split_csv)))
    train = [r["case"] for r in rows if r["split"] == "train"]
    val = [r["case"] for r in rows if r["split"] == "val"]
    if a.step == "raw":
        raw_ds = Path(os.environ["nnUNet_raw"]) / a.dataset
        for i, case in enumerate(train + val):
            v = add_prior(load_case(a.vol_root / case, NC_MODS), load_probs(a.maps / f"{case}.npz"))
            write_raw_case(raw_ds, case, v)
            print(f"{i + 1}/{len(train) + len(val)} {case}", flush=True)
        write_raw_dataset_json(raw_ds, len(train) + len(val))
    elif a.step == "splits":
        p = Path(os.environ["nnUNet_preprocessed"]) / a.dataset / "splits_final.json"
        p.write_text(json.dumps([{"train": train, "val": val}], indent=1))
        print(p)
    else:
        print(inject_targets(Path(os.environ["nnUNet_preprocessed"]) / a.dataset, a.vol_root), "cases injected")


if __name__ == "__main__":
    main()
