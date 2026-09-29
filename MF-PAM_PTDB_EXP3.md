# MF-PAM-PTDB-003 — clean+noisy training (matches the authors' 90/10 mix)

Spec: `specs/06_experiment3.md`. Config: `configs/mfpam_ptdb_003_noisy.yaml`. Seed 42. Supersedes Q2 of
`specs/03_experiment.md` for this run only; MF-PAM-PTDB-001 (clean-only) is kept as the comparison baseline.

## What was run
Unmodified MF-PAM (362,479 params), 16 kHz / 8 ms hop, DIO pseudo-labels (as in MF-PAM-PTDB-001). The only change
from Experiment 1: each training crop's MODEL INPUT is drawn 90% noisy / 10% clean (one of the dataset's SNR
{20,10,5,0} dB files, chosen uniformly at random), matching the official `dataset.py` recipe; the DIO label is
always computed from the CLEAN crop, exactly as in the official code. Deviations D18-D20 (fixed discrete SNR set
instead of the paper's NOISEX-92/continuous SNR, no RIR, seeded RNG) are in `specs/06_experiment3.md`.

## Training
Empirical noisy/clean mix verified at ~90/10 before launch (E3-1..E3-3, all passed). Training was watched by
`scripts/watchdog_stop_on_plateau.py` and stopped automatically at **epoch 900** when two consecutive 100-epoch
blocks each gained <=0.10 percentage points of validation RPA (block 701-800: +0.062 pt, block 801-900: +0.068 pt).
Best-validation checkpoint: **epoch 892** (val_RPA_50c 94.34%). No non-finite gradients across the run.

## Test (best-validation checkpoint, run once; TEST speakers F09/F10/M09/M10, 944 utterances, percent)
| Condition | DIO paper RPA | RCA | VRR | VFA | OA | RAPT paper RPA | RCA | VRR | VFA | OA | **RAPT RMVPE RPA** | RCA | OA | VRR | VFA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 93.71 | 94.00 | 80.41 | 0.16 | 94.98 | 84.32 | 87.07 | 87.56 | 1.56 | 94.38 | 78.88 | 80.83 | 94.50 | 87.15 | 1.51 |
| snr20 | 92.95 | 93.24 | 79.82 | 0.16 | 94.84 | 84.32 | 87.07 | 87.25 | 1.46 | 94.44 | 78.78 | 80.71 | 94.56 | 86.84 | 1.41 |
| snr10 | 91.48 | 91.78 | 76.42 | 0.12 | 94.03 | 84.66 | 87.47 | 84.64 | 1.05 | 94.53 | 77.76 | 79.62 | 94.66 | 84.32 | 1.01 |
| snr05 | 90.11 | 90.43 | 72.84 | 0.08 | 93.16 | 84.99 | 87.90 | 81.27 | 0.82 | 94.28 | 75.89 | 77.61 | 94.43 | 81.13 | 0.78 |
| snr00 | 88.25 | 88.60 | 67.00 | 0.06 | 91.73 | 85.31 | 88.28 | 75.31 | 0.58 | 93.50 | 71.74 | 73.24 | 93.70 | 75.50 | 0.55 |

`paper` = official-code protocol with the tolerance bug removed (true 50 cents). `RMVPE` = RMVPE-RRCGD protocol
(predicted-unvoiced set to 0 Hz). Full JSON incl. official-code (buggy) values: `results/MF-PAM-PTDB-003/test_metrics.json`.

## Paper-style DIO replication check: buggy official code vs. the corrected 50-cent metric
This is the same check as `reports/RPA_tolerance_analysis.md`, repeated for the noisy-trained model. "Buggy" = the
official `train.py` metric call as shipped (effective tolerance ~80-220 cents, pitch-dependent, see that report for
the derivation); "fixed" = a true 50-cent tolerance (`raw_pitch_accuracy`/`raw_chroma_accuracy` called correctly).
Per-file mean is the authors' own aggregation; pooled (frame-weighted) is shown for reference.

| Condition | Buggy RPA (per-file) | Buggy RCA (per-file) | Buggy RPA (pooled) | **Fixed 50c RPA (per-file)** | Fixed 50c RCA (per-file) | Fixed 50c RPA (pooled) |
|---|---|---|---|---|---|---|
| clean | 96.79 | 96.80 | 97.03 | **93.49** | 93.81 | 93.71 |
| snr20 | 96.31 | 96.31 | 96.59 | **92.71** | 93.04 | 92.95 |
| snr10 | 95.73 | 95.73 | 96.02 | **91.23** | 91.56 | 91.48 |
| snr05 | 95.16 | 95.17 | 95.46 | **89.87** | 90.23 | 90.11 |
| snr00 | 94.46 | 94.46 | 94.79 | **87.99** | 88.38 | 88.25 |

For comparison, Experiment 1 (clean-only training) on the same clean test set: buggy RPA (per-file) 97.14 vs. fixed
50-cent RPA 94.38 (paper: 97.12). Clean+noisy training (003) is marginally *below* clean-only (001) on the clean
test condition under both metrics (96.79 vs 97.14 buggy; 93.49 vs 94.38 fixed), and both are within ~0.3-0.9 points
of the paper's reported 97.12/97.13 under the buggy metric, not under the stated 50-cent one. See
`reports/RPA_tolerance_analysis.md` for the full derivation of why the buggy tolerance is ~80-220 (pitch-dependent,
not 50) cents and why that -- not a difference in our setup -- most plausibly explains the paper's number.

## Noise robustness (the point of this experiment)
Comparing SNR 0 dB clean-vs-noisy training, RAPT/RMVPE-style RPA (the number used against your RRCGD comparisons):
Experiment 3 (noisy-trained) reaches 71.74 at SNR 0 dB vs. Experiment 1 (clean-only) which was never exposed to
noise during training. Experiment 1's SNR 0 dB numbers are reported in `MF-PAM_PTDB_REPRODUCTION.md` for the DIO
reference only; a like-for-like RAPT/RMVPE SNR-0 comparison is in both experiments' `test_metrics.json` files.

## Reproducibility
```
conda activate mfpam
python scripts/prepare_16k.py --config configs/mfpam_ptdb_003_noisy.yaml --splits train --noisy
python scripts/train_ptdb.py --config configs/mfpam_ptdb_003_noisy.yaml       # auto-stopped at epoch 900 by the watchdog
python scripts/watchdog_stop_on_plateau.py --pid <pid> --log results/MF-PAM-PTDB-003/training_log.csv \
       --eval-config configs/mfpam_ptdb_003_noisy.yaml   # plateau stop + final evaluation, one-shot
```
Seed 42; manifest in `results/MF-PAM-PTDB-003/`. Not bit-exact (cudnn.benchmark on, D13 of specs/03).
