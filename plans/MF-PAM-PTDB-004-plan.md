# Plan — MF-PAM-PTDB-004 (baseline on the new 8 kHz data; approved 2026-10-26 by user instruction)

| ID | Task | Acceptance |
|---|---|---|
| E4-T1 | `configs/mfpam_ptdb_004_8k_noisy.yaml` = 003 config with new root / cache / native_sample_rate / paths | - |
| E4-T2 | `scripts/prepare_16k.py --splits train validation --noisy` (done); test files prepared by `evaluate_ptdb.py` at the end | - |
| E4-T3 | `scripts/verify_ptdb.py` on the new data; quick checks (90/10 mix, SNR of noisy files) | E4-1..3 |
| E4-T4 | wait for a free GPU (`scripts/run_004_when_gpu_free.sh`), train in background + plateau watchdog (stop + one-shot test eval) | - |
| E4-T5 | report `MF-PAM_PTDB_EXP4.md`; mark 001/003 reports as superseded by 004 | - |

## Status (2026-10-27)
All tasks done. Trained to epoch 900 (watchdog plateau stop), best-validation checkpoint epoch 898 (val_RPA_50c 93.66%). Test evaluation run once (clean + SNR 20/10/5/0).
Report: `MF-PAM_PTDB_EXP4.md`.
