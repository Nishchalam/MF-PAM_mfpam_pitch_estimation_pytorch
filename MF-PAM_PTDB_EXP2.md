# MF-PAM-PTDB-002 — MF-PAM under the RMVPE-RRCGD PTDB protocol

Spec: `specs/05_experiment2.md` (approved 2026-09-25). Config: `configs/mfpam_ptdb_exp2.yaml`. Seed 42.

## What was run
Unmodified MF-PAM (362,479 params) with everything else taken from the RMVPE-RRCGD PTDB pipeline: RAPT `.npy` labels (frame k at k*10 ms),
one-hot 360 classes at 20-cent bins, plain BCE, Adam 1e-3, StepLR(5 epochs, 0.98), grad-clip 3, batch size 8, 256-frame (2.55 s) chunks, clean-only training,
validation every epoch, best-validation-RPA checkpoint, early stopping (patience 4 epochs), local-average decoding, threshold 0.5, RMVPE-style metrics (per-file mean, 50 cents).
Audio 48 kHz -> 12.8 kHz so that MF-PAM's fixed 128-sample hop = 10 ms.

## Training
Stopped early at epoch 21 (iteration 25,579); best validation RPA 64.53 at epoch 17 (iteration 20706); no skipped/non-finite steps. Log: `results/MF-PAM-PTDB-002/training_log.csv`.

## Test (best-validation checkpoint, run once; TEST speakers F09, F10, M09, M10; 944 utterances; RMVPE-RRCGD metrics vs RAPT, percent)
| Condition | RPA | RCA | OA | VR | VFA |
|---|---|---|---|---|---|
| clean | 66.30 | 67.13 | 92.50 | 67.42 | 0.47 |
| snr20 | 66.01 | 66.83 | 92.44 | 67.12 | 0.46 |
| snr10 | 64.66 | 65.44 | 92.17 | 65.72 | 0.42 |
| snr05 | 62.66 | 63.40 | 91.72 | 63.66 | 0.39 |
| snr00 | 57.98 | 58.67 | 90.64 | 58.90 | 0.35 |

Frame-level predictions for the clean test set: `results/MF-PAM-PTDB-002/predictions/`.

## Deviations from the RMVPE-RRCGD pipeline
D-E2-1 12.8 kHz audio instead of 8 kHz (MF-PAM hop constraint); D-E2-2 n/a (targets identical, parity-tested: 60/60 chunks identical to RMVPE-RRCGD cache);
D-E2-3 no augmentation (as RMVPE-RRCGD); D-E2-4 fp32 instead of AMP; MF-PAM outputs 2-3 extra frames, truncated to label length.
Parity tests (`tests/test_exp2_parity.py`): decoding and metric code produce identical results to the RMVPE-RRCGD source code on realistic erroneous predictions.
Not tuned: the RMVPE-RRCGD lr/epochs/patience were used unchanged for MF-PAM; test data was read only for the final evaluation.
