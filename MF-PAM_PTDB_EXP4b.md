# MF-PAM-PTDB-004b — seed replicate of the 004 baseline (seed 43)

Config: `configs/mfpam_ptdb_004b_8k_noisy_seed43.yaml`. Identical to 004 (`MF-PAM_PTDB_EXP4.md`) except `experiment.seed` 42 -> 43 and the output paths.
Same new 8 kHz data, same speaker-disjoint split (train F01-F06/M01-M06, valid F07/F08/M07/M08, test F09/F10/M09/M10), same recipe, same plateau watchdog.
Purpose: put a seed-variation scale on the 003 -> 004 drop.

## Training
Stopped by the watchdog at **epoch 900** (blocks 701-800: +0.085 pt, 801-900: +0.072 pt, each <=0.10). Best-validation checkpoint = epoch 900,
val_RPA_50c **93.57%** (004: 93.66% at epoch 898). Test set evaluated once on that checkpoint (944 utterances).

## Test (percent, 50-cent tolerance)
| Condition | DIO pooled RPA 004 (seed 42) | 004b (seed 43) | Δ | RAPT-RMVPE per-file RPA 004 | 004b | Δ |
|---|---|---|---|---|---|---|
| clean | 93.04 | 92.97 | -0.07 | 77.98 | 78.53 | +0.55 |
| snr20 | 91.54 | 91.50 | -0.04 | 77.35 | 77.86 | +0.51 |
| snr10 | 89.22 | 89.12 | -0.10 | 74.18 | 74.70 | +0.51 |
| snr05 | 87.33 | 87.28 | -0.05 | 70.80 | 71.30 | +0.50 |
| snr00 | 84.51 | 84.45 | -0.06 | 64.25 | 64.73 | +0.49 |

Clean, 004b: per-file DIO RPA 92.72; official-code (doubly converted, comparison only) 96.69 pooled / 96.45 per-file.

## Reading
- Against the DIO labels the two seeds agree within 0.04-0.10 pt in every condition, so seed variation (two seeds only) is about 0.1 pt there.
  The 003 -> 004 drops (-0.67 clean, -1.41 to -3.74 noisy) are 7-37 times that, so they are not seed noise; they come from the data change.
- Against RAPT the seed-43 model is a constant ~+0.5 pt higher in every condition. With two seeds this is not explained; it is a reminder that
  RAPT-referenced numbers carry larger seed variation than DIO-referenced ones. The 003 -> 004 RAPT drops (-0.90 to -7.49) still exceed it.
- Both runs stopped at epoch 900 with validation RPA still rising slowly (~+0.07 pt per 100 epochs), as in 004.
- Two seeds give a range, not a confidence interval.
