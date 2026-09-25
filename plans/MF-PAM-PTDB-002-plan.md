# Plan — MF-PAM-PTDB-002 (APPROVED 2026-09-25; ALL TASKS DONE)

| ID | Task | Spec/AC |
|---|---|---|
| E2-T1 | `scripts/prepare_12k8.py`: resample clean train/valid/test (and, if approved, noisy test SNR 20/10/5/0) 48→12.8 kHz (soxr VHQ) into the cache dir; read-only use of `.npy` labels | 05, A3 |
| E2-T2 | `src/rmvpe_protocol.py`: verbatim copies (with source path/commit comment) of label one-hot, `to_local_average_cents`, metric evaluation | 05, A4, A5, A7 |
| E2-T3 | `src/ptdb_exp2_dataset.py`: chunking 256 frames ↔ 32768 samples (+ remainder chunk), full-file eval sets, label/audio frame check | 05, A3, A6 |
| E2-T4 | `configs/mfpam_ptdb_exp2.yaml` (all values incl. seed 42, bs, lr 1e-3, StepLR, clip 3, epochs 30, patience 4) | 05 |
| E2-T5 | `src/train_exp2.py` + `scripts/train_exp2.py`: fp32 training loop, per-epoch validation (RMVPE-style), best/last checkpoints, early stopping, CSV log | 05, A1, A8 |
| E2-T6 | Parity tests: labels vs RMVPE cache (A4), decoding (A5), metrics (A7); `scripts/smoke_test_exp2.py` | A4–A7 |
| E2-T7 | `scripts/evaluate_exp2.py`: best-val checkpoint on TEST (clean [+ noisy]); RMVPE-style metrics + frame CSVs | 05, A8 |
| E2-T8 | Manifest + `MF-PAM_PTDB_EXP2.md` report (comparison table vs RMVPE-RRCGD) | A9 |
Order: T1 → T2 → T3 → T6(parity) → T4 → T5 → smoke → train (GPU 1) → T7 → T8.
