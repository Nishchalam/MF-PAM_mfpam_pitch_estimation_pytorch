# MF-PAM-PTDB-004 — baseline retrained on the NEW 8 kHz PTDB data (supersedes 001–003)

Spec: `specs/07_experiment4.md`. Config: `configs/mfpam_ptdb_004_8k_noisy.yaml`. Seed 42. Supersedes `MF-PAM_PTDB_EXP3.md` (003) as the baseline;
001–003 were computed on the old PTDB copy (noise added at 48 kHz, then resampled) and are kept for history only.

## What was run
Exactly the 003 recipe: unmodified MF-PAM (362,479 params), 16 kHz / 8 ms hop, DIO pseudo-labels on the clean crop, 4.5 s crops / 1 s stride /
Shift(8000), model input 90% noisy / 10% clean (SNR {20,10,5,0} dB files, uniform), batch 64, Adam 3e-4, ExpLR 0.999, BCE, whole-utterance inference,
validation-RPA checkpoint selection, plateau watchdog, corrected 50-cent metrics, test run once.
**Only change (D21):** data = `/media/nishchala/Seagate_2/dataset/PTDB_data_8k_10ms_hop` (8 kHz; noisy files made after resampling), resampled 8 -> 16 kHz
(soxr VHQ); DIO labels recomputed on this audio. Same speaker-disjoint split (12/4/4 speakers; 2832/942/944 utterances) and byte-identical `.f0` files.
Measured SNR of the new noisy files at 16 kHz: 20.22 / 10.22 / 5.22 / 0.22 dB (targets 20/10/5/0).

## Training
Stopped automatically at **epoch 900** by `scripts/watchdog_stop_on_plateau.py` (blocks 701–800: +0.089 pt, 801–900: +0.042 pt, each <=0.10). Best-validation
checkpoint **epoch 898** (val_RPA_50c 93.66%; 003: 94.34% at epoch 892). About 55 s/epoch on an RTX 5080. No non-finite gradients. Noisy share of training input ~90%.

## Test (best-validation checkpoint, run once; TEST speakers F09/F10/M09/M10, 944 utterances, percent)
| Condition | DIO paper RPA | RCA | VRR | VFA | OA | RAPT paper RPA | RCA | VRR | VFA | OA | **RAPT RMVPE RPA** | RCA | OA | VRR | VFA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 93.04 | 93.31 | 78.39 | 0.18 | 94.47 | 84.49 | 87.33 | 85.83 | 1.41 | 94.31 | 77.98 | 79.88 | 94.44 | 85.34 | 1.36 |
| snr20 | 91.54 | 91.83 | 76.21 | 0.13 | 93.96 | 84.73 | 87.60 | 84.29 | 1.10 | 94.42 | 77.35 | 79.19 | 94.55 | 83.84 | 1.06 |
| snr10 | 89.22 | 89.53 | 70.29 | 0.08 | 92.53 | 85.36 | 88.28 | 78.81 | 0.68 | 94.04 | 74.18 | 75.73 | 94.19 | 78.50 | 0.65 |
| snr05 | 87.33 | 87.69 | 65.75 | 0.06 | 91.41 | 85.63 | 88.59 | 74.16 | 0.50 | 93.42 | 70.80 | 72.11 | 93.60 | 74.01 | 0.47 |
| snr00 | 84.51 | 84.94 | 58.29 | 0.04 | 89.57 | 85.37 | 88.40 | 66.02 | 0.36 | 91.99 | 64.25 | 65.25 | 92.26 | 66.42 | 0.33 |

`paper` = official-code protocol with the tolerance bug removed (true 50 cents), frame-pooled; `RMVPE` = RMVPE-RRCGD protocol (predicted-unvoiced set to 0 Hz),
per-file mean. Full JSON incl. per-file values and the doubly-converted official-code metric (`official_style`, NOT used for selection or reporting):
`results/MF-PAM-PTDB-004/test_metrics.json`.
Per-file mean 50-cent DIO RPA (clean): 92.78. Official-code (doubly-converted) clean RPA, shown only for comparison with the paper's 97.12: 96.78 pooled / 96.55 per-file.

## Comparison with the superseded 003 (old data)
| Condition | DIO pooled RPA 003 | 004 | Δ | RAPT-RMVPE RPA 003 | 004 | Δ |
|---|---|---|---|---|---|---|
| clean | 93.71 | 93.04 | −0.67 | 78.88 | 77.98 | −0.90 |
| snr20 | 92.95 | 91.54 | −1.41 | 78.78 | 77.35 | −1.43 |
| snr10 | 91.48 | 89.22 | −2.26 | 77.76 | 74.18 | −3.58 |
| snr05 | 90.11 | 87.33 | −2.78 | 75.89 | 70.80 | −5.09 |
| snr00 | 88.25 | 84.51 | −3.74 | 71.74 | 64.25 | −7.49 |

The gap grows with noise level. Hypothesis (NOT tested): if the old noise was broadband and added at 48 kHz, resampling to 16 kHz removed much of its power, so
003's noisy conditions were easier than their nominal SNR; in the new data the noise is added at 8 kHz, so the nominal SNR is the in-band SNR (verified: 0.2 dB above target at 16 kHz).
Part of the clean-condition drop (−0.67) may come from the new audio being band-limited to 4 kHz; also untested.

## Reproducibility
```
conda activate mfpam_env        # + soxr==1.1.0, PyYAML==6.0.3 (pinned in requirements-lock-mfpam-env.txt)
python scripts/prepare_16k.py --config configs/mfpam_ptdb_004_8k_noisy.yaml --splits train validation --noisy
python scripts/train_ptdb.py --config configs/mfpam_ptdb_004_8k_noisy.yaml        # auto-stopped at epoch 900 by the watchdog
python scripts/watchdog_stop_on_plateau.py --pid <pid> --log results/MF-PAM-PTDB-004/training_log.csv \
       --eval-config configs/mfpam_ptdb_004_8k_noisy.yaml     # plateau stop + one-shot test evaluation
```
Seed 42; manifest in `results/MF-PAM-PTDB-004/`. Not bit-exact (cudnn.benchmark on). `scripts/run_004_when_gpu_free.sh` launches training + watchdog once the GPU is idle.

## Not part of this run
`MF-PAM-PTDB-005` (128 ms input window on the same data; `specs/08_experiment5.md`, `configs/mfpam_ptdb_005_8k_128ms.yaml`) is implemented behind `training.window_ms` but was put on hold and never trained. The 004 evaluation ran with `window_samples: null` (whole utterances).
