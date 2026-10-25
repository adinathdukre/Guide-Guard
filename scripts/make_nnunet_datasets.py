import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guideguard.nnunet_helpers import CHANNELS_NC, CHANNELS_S, cv_splits, write_raw_dataset, write_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol_root", required=True, type=Path)
    ap.add_argument("--split_csv", required=True, type=Path, help="case,patient,split")
    ap.add_argument("--folds_csv", required=True, type=Path, help="case,fold (h's 5-fold CV over train)")
    ap.add_argument("--nc_dataset", default="Dataset511_GuideNC")
    ap.add_argument("--s_dataset", default="Dataset512_GuideS")
    ap.add_argument("--splits_only", action="store_true", help="after nnUNetv2_plan_and_preprocess")
    a = ap.parse_args()
    raw, pre = Path(os.environ["nnUNet_raw"]), Path(os.environ["nnUNet_preprocessed"])
    rows = list(csv.DictReader(open(a.split_csv)))
    train = [r["case"] for r in rows if r["split"] == "train"]
    val = [r["case"] for r in rows if r["split"] == "val"]
    folds = {r["case"]: int(r["fold"]) for r in csv.DictReader(open(a.folds_csv))}
    if not a.splits_only:
        write_raw_dataset(raw, a.nc_dataset, CHANNELS_NC, a.vol_root, train + val)
        write_raw_dataset(raw, a.s_dataset, CHANNELS_S, a.vol_root, train + val)
        return
    print(write_splits(pre, a.nc_dataset, cv_splits(folds, train, val)))
    print(write_splits(pre, a.s_dataset, [{"train": sorted(train), "val": sorted(val)}]))


if __name__ == "__main__":
    main()
