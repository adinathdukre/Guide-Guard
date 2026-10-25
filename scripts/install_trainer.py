import argparse
import shutil
from pathlib import Path

import nnunetv2

SRC = Path(__file__).resolve().parents[1] / "guideguard" / "nnunet_ext"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    a = ap.parse_args()
    dst = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants" / "guide_guard"
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    if a.copy:
        shutil.copytree(SRC, dst)
    else:
        dst.symlink_to(SRC, target_is_directory=True)
    print(f"{dst} -> {SRC}")


if __name__ == "__main__":
    main()
