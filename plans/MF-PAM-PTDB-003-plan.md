# Plan — MF-PAM-PTDB-003 (clean+noisy training; approved 2026-09-27)

Reuses MF-PAM-PTDB-001's infrastructure (`src/ptdb_common.py`, `src/metrics.py`, `src/evaluation.py`, `src/manifest.py`,
`scripts/evaluate_ptdb.py`) unchanged. Only the TRAIN dataset and the training script change.

| ID | Task | Spec | Acceptance |
|---|---|---|---|
| E3-T1 | `scripts/prepare_16k.py --splits train --noisy`: resample the 4 SNR train files to 16 kHz (validation/test noisy already prepared for eval) | 06 | E3-4 |
| E3-T2 | `src/ptdb_dataset.py`: add `TrainChunksNoisy` (shares chunk indexing with `TrainChunks`; per-crop 90/10 noisy/clean draw, shared shift) | 06 | E3-1, E3-2, E3-3 |
| E3-T3 | `src/train_ptdb.py` / `configs/mfpam_ptdb_003_noisy.yaml`: select `TrainChunksNoisy` when `augmentation.noisy_input: true`; new `paths` (results/checkpoints `MF-PAM-PTDB-003`) | 06 | - |
| E3-T4 | Quick verification: empirical 90/10 mix ratio, label-source check, alignment check (no full smoke_test.py rerun needed — model/quantiser/chunking already verified for 001) | 06 | E3-1..E3-3 |
| E3-T5 | Launch training in background (GPU 0 free since 001 stopped); auto-commit only at the end (per prior user instruction pattern) unless told otherwise | 06 | - |
| E3-T6 | After training: evaluate best-validation checkpoint on TEST (clean + noisy SNR), report alongside MF-PAM-PTDB-001 | 06, 04 | - |

## Status (2026-09-29)
All tasks done. Trained to epoch 900 (auto-stopped by `scripts/watchdog_stop_on_plateau.py` on a validation-RPA
plateau: two consecutive 100-epoch blocks each gained <=0.10 pt). Best-validation checkpoint epoch 892
(val_RPA_50c 94.34%). Final test evaluation run once (clean + SNR 20/10/5/0 dB). Report: `MF-PAM_PTDB_EXP3.md`;
paper-style buggy-vs-fixed replication check added to `reports/RPA_tolerance_analysis.md` section 6.
