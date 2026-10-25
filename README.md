# Guide and Guard

Code for **"Guide and Guard: Prior-Guided Virtual-Contrast MRI Synthesis with Conformal Flags for Missed
Enhancement"** (A. A. Rege, A. M. Dukre, I. Razzak; IEEE ISBI 2027 submission).

- **Guard** attaches a three-level map (Supported / Check / Unlikely) to the T1c of *any* synthesiser.
  - Conformal risk control (CRC) on an NC-only ET score bounds the expected fraction of enhancing tumour (ET)
    that the image neither shows nor flags (the *miss*). ET that the reader proxy *S* finds on the image counts
    as covered, so the bound holds for any synthesiser.
  - Learn-then-Test (LTT) controls false discoveries in the Supported level.
  - Its cost, the *review burden*, is the non-ET volume marked Supported or Check. The same score calibrated
    without an image (*NC-only CRC*) gives the burden without any image, so the difference measures what the
    image adds.
- **Guide** synthesises post-contrast T1 (T1c) from non-contrast (NC) brain MRI (T1n, T2w, FLAIR). Out-of-fold
  lesion maps from an NC segmenter *h* are fed as extra input channels to two regression synthesisers: a 2.5D
  YODA regressor and a 3D nnU-Net with L1 loss (two runs). Guide's image is the mean of the two, with flip
  test-time augmentation.

Four segmentation networks are used, all nnU-Nets trained on our training split and then frozen:
*h* (2D, 5 folds; NC lesion maps m = WT/TC/ET), *g* (3D; the NC ET score), *S* (3D, on all four sequences; the
reader proxy that marks the ET an image shows). *S₂*, used in the paper only as a robustness check, is not
released.

Held-out BraTS 2023 GLI results (249 cases; burden: score *g*, α = 0.1):

| Method | PSNR | SSIM | ET Dice (by S) | Review burden |
|---|---|---|---|---|
| Guide | 23.71 dB | 0.766 | 0.629 | 27.6 mL |
| Guide-fast | 23.61 dB | 0.765 | 0.632 | 26.3 mL |
| NC-only CRC (no image) | | | 0.646 (of *g*) | 30.7 mL |
| Real T1c | ∞ | 1.000 | 0.871 | 3.9 mL |

Code: <https://github.com/adinathdukre/Guide-Guard>. Weights:
<https://huggingface.co/Atharva-Rege/guide-guard>.

## Installation

Three Python environments are needed, because the two Guide members depend on different upstream code bases.

| Env | Used for | Install |
|---|---|---|
| `main` | h, g, S, fusion, Guard | `pip install -r requirements.txt` |
| `yoda` | the 2.5D member | `pip install -r requirements-yoda.txt`, plus a clone of [YODA](https://github.com/Deep-MI/YODA) (commit `86e8dc8`), `export YODA_ROOT=/path/to/YODA` |
| `nnt` | the 3D member | `pip install -e` of [nnUNet_translation](https://github.com/Phyrise/nnUNet_translation) (commit `98f2406`), then `python scripts/install_trainer.py` |

`ema-pytorch` in `requirements-yoda.txt` is only needed for training.

Checkpoints: download `guide-guard-checkpoints.zip` from
[Hugging Face](https://huggingface.co/Atharva-Rege/guide-guard) and unzip it:

```bash
huggingface-cli download Atharva-Rege/guide-guard guide-guard-checkpoints.zip --local-dir .
unzip guide-guard-checkpoints.zip          # -> guide-guard-checkpoints/, used as CKPT below
cd guide-guard-checkpoints && sha256sum -c MANIFEST.txt
```

| Folder | Network | Checkpoint |
|---|---|---|
| `h/` | NC lesion prior *h* (nnU-Net 2d, folds 0–4) | `fold_k/checkpoint_final.pth` |
| `g/` | NC ET score *g* (nnU-Net 3d_fullres, trained on all training cases: fold 5) | `fold_5/checkpoint_final.pth` |
| `S/` | reader proxy *S* (nnU-Net 3d_fullres) | `fold_0/checkpoint_final.pth` |
| `guide25d/` | Guide's 2.5D member (YODA regression + prior, EMA) | `guide25d.pt` |
| `guide3d_r1/`, `guide3d_r2/` | Guide's 3D member, two runs (`nnUNetTrainerGuide3D`) | chosen by `fold_0/selection.json` |

## Data

Inputs are BraTS-layout volumes: SRI24 space, 1 mm, 240×240×155, skull-stripped. Each case folder holds
`<case>-{t1n,t2w,t2f}.nii.gz`, plus `t1c` and `seg` for training or evaluation.

```bash
python scripts/preprocess.py --raw_root /data/BraTS2023-GLI --out VOL     # add --no_target --no_seg for inference-only data
```

Preprocessing does the following:
- The brain mask is the union of the nonzero voxels of the NC inputs.
- NC inputs are clipped to the brain [p0.5, p99.5]; T1c is clipped to the brain [p0.5, p99.9].
- Both are rescaled to [0, 1].

`splits/` holds the patient-level split used in the paper (876 train / 126 val / 249 test cases) and the 5-fold CV
of the training cases.

## Inference

```bash
PY_MAIN=/envs/main/bin/python PY_YODA=/envs/yoda/bin/python PY_NNT=/envs/nnt/bin/python \
  bash scripts/run_guide.sh VOL CKPT WORK [--cases BraTS-GLI-00000-000 ...]
```

| Output | Contents |
|---|---|
| `WORK/guide/<case>.npy` | Guide T1c (float16, [0, 1], 0 outside the brain) |
| `WORK/maps/{h,g}/<case>.npz` | NC lesion maps (WT, TC, ET) and the NC ET score g |
| `WORK/reader/guide/<case>.npz` | Segmenter S on the synthetic image |

`--fast` runs Guide-fast instead: the 2.5D member only, without test-time augmentation, with h's fold 0 only. Its
output goes to `WORK/guide_fast`.

The steps can also be run one at a time: `predict_maps.py`, `predict_guide25d.py`, `predict_guide3d.py`,
`fuse_guide.py` and `predict_reader.py`.

Image quality and the ET Dice of S, per case (brain PSNR, SSIM and MAE; ET PSNR; ET Dice with `--reader`; LPIPS and
lesion-wise Dice of Table 1 are not computed here):

```bash
python scripts/evaluate_images.py --vol_root VOL --images WORK/guide --reader WORK/reader/guide --out images_eval.csv
```

## Guard

Guard works with any synthesiser. For the images of that synthesiser:

1. Run `predict_reader.py` on its images (and `predict_maps.py --models g`, plus `h` for the averaged score).
2. Calibrate on labelled patients who are not used for testing, one timepoint per patient.
3. Apply the calibrated thresholds, or evaluate them.

```bash
python scripts/guard.py calibrate --vol_root VOL --reader WORK/reader/guide --maps WORK/maps --cases cal.txt \
    --alpha 0.1 --out thresholds.json
python scripts/guard.py apply    --vol_root VOL --reader WORK/reader/guide --maps WORK/maps --cases test.txt \
    --thresholds thresholds.json --out WORK/guard_maps          # uint8: 0 Unlikely, 1 Check, 2 Supported
python scripts/guard.py evaluate --vol_root VOL --reader WORK/reader/guide --maps WORK/maps --cases test.txt \
    --thresholds thresholds.json --out guard_eval.csv          # miss, review burden, Supported FDP
```

Options:
- `--score g` (default) uses the NC ET score g, the paper's primary score. `--score avg` uses the averaged score
  s = (g + m_ET)/2.
- `--no_image` in place of `--reader` gives NC-only CRC, the image-free baseline map (calibrate, apply and
  evaluate it the same way). Its burden minus the image's burden is what the image saves.
- `--variant lesion|tail` selects the lesion-level or the patient-tail miss loss.
- Defaults follow the paper: α = 0.1, β = 0.2, δ = 0.1, ET-positive patients have ≥ 50 ET voxels, false
  discoveries lie over 2 mm from the reference ET.

The guarantee holds marginally, in expectation, for ET-positive test patients exchangeable with the calibration
patients; a single patient, or small lesions, can miss more. It did not transfer across sites without
recalibration, and Guard does not detect out-of-scope cohorts (e.g. meningiomas). Recalibrate at a new site.
Labels are algorithmic evidence categories, not diagnoses.

## Training

The released checkpoints are enough for inference and for Guard. Retraining Guide needs the baseline
synthesisers retrained on our split, which both members are fine-tuned from. These base weights are not in the
Hugging Face repo; a folder `BASE` with them is assumed below:
- `yoda.pt`: YODA in regression mode (20 input channels, 1 output, EMA weights);
- `nnunet_l1.pth`: nnU-Net-L1 (nnUNet_translation `nnUNetTrainerMRCT_mae`, `nnResUNetPlans`).

1. **h, g, S (nnU-Net v2, region-based WT/TC/ET).**
   1. `python scripts/make_nnunet_datasets.py --vol_root VOL --split_csv splits/brats23gli_split_v1.csv --folds_csv splits/brats23gli_train_folds_v1.csv`
   2. `nnUNetv2_plan_and_preprocess -d 511 512`
   3. Rerun step 1 with `--splits_only`.
   4. Train h: `nnUNetv2_train 511 2d {0..4} -tr nnUNetTrainer_250epochs --npz`
   5. Train g: `nnUNetv2_train 511 3d_fullres 5 -tr nnUNetTrainer_250epochs`
   6. Train S: `nnUNetv2_train 512 3d_fullres 0 -tr nnUNetTrainer_250epochs`
   7. Write the lesion maps. Training cases get their out-of-fold maps:
      `predict_maps.py --from_cv $nnUNet_results/Dataset511_GuideNC/nnUNetTrainer_250epochs__nnUNetPlans__2d --folds_csv ...`
      All other cases get the 5-fold ensemble: `predict_maps.py --models h`.
2. **2.5D member.** Start from `BASE/yoda.pt`.
   ```bash
   python scripts/train_guide25d.py --vol_root VOL --maps MAPS/h --split_csv splits/brats23gli_split_v1.csv --base_ckpt BASE/yoda.pt --out RUN
   python scripts/select_guide25d.py --run_dir RUN --vol_root VOL --maps MAPS/h --split_csv splits/brats23gli_split_v1.csv
   ```
   The recipe: 25k steps, Adam with a cosine schedule from 1e-4 to 1e-6, prior dropout 0.15, left–right flips and
   an auxiliary WT/TC/ET head (weight 0.005); the snapshot of best validation brain PSNR is kept (released:
   step 20k).
3. **3D member.** Start from `BASE/nnunet_l1.pth`.
   ```bash
   python scripts/make_guide3d_dataset.py raw --vol_root VOL --maps MAPS/h --split_csv splits/brats23gli_split_v1.csv
   nnUNetv2_extract_fingerprint -d 622 && nnUNetv2_plan_experiment -d 622 -pl nnUNetPlannerResUNet
   nnUNetv2_preprocess -d 622 -plans_name nnResUNetPlans -c 3d_fullres
   python scripts/make_guide3d_dataset.py splits --vol_root VOL --split_csv ...
   python scripts/make_guide3d_dataset.py inject --vol_root VOL --split_csv ...
   GUIDE3D_INIT=BASE/nnunet_l1.pth nnUNetv2_train 622 3d_fullres 0 -tr nnUNetTrainerGuide3D -p nnResUNetPlans
   python scripts/select_guide3d.py --model_dir RESULTS/... --vol_root VOL --maps MAPS/h --split_csv ...
   ```
   The recipe: 400 epochs, L1 + 5×MSE, prior dropout 0.15, left–right flips and the auxiliary head (Dice + BCE);
   the snapshot and tile step of best validation brain PSNR are kept in `selection.json`. The paper averages two
   independent runs.

## Licence

The code and model weights are released under Apache-2.0 (`LICENSE`). The code builds on:
- [YODA](https://github.com/Deep-MI/YODA) (Apache-2.0);
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [nnUNet_translation](https://github.com/Phyrise/nnUNet_translation) (Apache-2.0).

The models were trained on BraTS 2023 GLI (CC BY 4.0). They are for research use only and are not validated for
clinical use.
