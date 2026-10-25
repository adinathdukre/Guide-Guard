#!/usr/bin/env bash
set -euo pipefail

usage() { echo "usage: $0 VOL_ROOT CKPT_ROOT WORK_DIR [--fast] [--cases C1 C2 ...]"; exit 1; }
[[ $# -ge 3 ]] || usage
VOL=$1 CKPT=$2 WORK=$3; shift 3
FAST=0 CASES=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --fast) FAST=1; shift ;;
    --cases) shift; while [[ $# -gt 0 && $1 != --* ]]; do CASES+=("$1"); shift; done ;;
    *) usage ;;
  esac
done
CA=(); [[ ${#CASES[@]} -gt 0 ]] && CA=(--cases "${CASES[@]}")

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY_MAIN=${PY_MAIN:-python}
PY_YODA=${PY_YODA:-$PY_MAIN}
PY_NNT=${PY_NNT:-$PY_MAIN}

if [[ $FAST == 1 ]]; then
  $PY_MAIN "$HERE/predict_maps.py" --vol_root "$VOL" --ckpt "$CKPT" --out "$WORK/maps_fast" --models h \
    --h_folds 0 --h_no_mirroring "${CA[@]}"
  $PY_YODA "$HERE/predict_guide25d.py" --vol_root "$VOL" --maps "$WORK/maps_fast/h" \
    --ckpt "$CKPT/guide25d/guide25d.pt" --out "$WORK/guide_fast" --no_tta "${CA[@]}"
  echo "Guide-fast images: $WORK/guide_fast"
  exit 0
fi

$PY_MAIN "$HERE/predict_maps.py" --vol_root "$VOL" --ckpt "$CKPT" --out "$WORK/maps" --models h g "${CA[@]}"
$PY_YODA "$HERE/predict_guide25d.py" --vol_root "$VOL" --maps "$WORK/maps/h" --ckpt "$CKPT/guide25d/guide25d.pt" \
  --out "$WORK/members/guide25d" "${CA[@]}"
for r in r1 r2; do
  $PY_NNT "$HERE/predict_guide3d.py" --vol_root "$VOL" --maps "$WORK/maps/h" --model_dir "$CKPT/guide3d_$r" \
    --out "$WORK/members/guide3d_$r" "${CA[@]}"
done
$PY_MAIN "$HERE/fuse_guide.py" --vol_root "$VOL" --member "$WORK/members/guide3d_r1" "$WORK/members/guide3d_r2" \
  --weights 0.5 0.5 --out "$WORK/members/guide3d" "${CA[@]}"
$PY_MAIN "$HERE/fuse_guide.py" --vol_root "$VOL" --member "$WORK/members/guide25d" "$WORK/members/guide3d" \
  --weights 0.5 0.5 --out "$WORK/guide" "${CA[@]}"
$PY_MAIN "$HERE/predict_reader.py" --vol_root "$VOL" --images "$WORK/guide" --ckpt "$CKPT" --out "$WORK/reader/guide" \
  "${CA[@]}"
echo "Guide images: $WORK/guide; lesion maps and NC score: $WORK/maps; S on Guide: $WORK/reader/guide"
