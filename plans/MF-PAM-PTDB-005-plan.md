# Plan — MF-PAM-PTDB-005 (8 kHz data, 128 ms window; (ON HOLD))

| ID | Task | Acceptance |
|---|---|---|
| E4-T1 | `configs/mfpam_ptdb_005_8k_128ms.yaml` (new root/cache, `training.window_ms: 128`) | - |
| E4-T2 | `scripts/prepare_16k.py --splits train validation --noisy` (8k->16k; train DIO cached in window mode); test prepared by `evaluate_ptdb.py` at the end | E4-1 |
| E4-T3 | `src/ptdb_dataset.py::TrainWindowsNoisy`, `src/evaluation.py::forward_windowed`, window plumbing in `train_ptdb.py` / `evaluate_ptdb.py` (all gated on `training.window_ms`; 001/003 paths unchanged) | E4-1..3 |
| E4-T4 | `tests/test_window005.py` | E4-1..3 |
| E4-T5 | `scripts/verify_ptdb.py` on the new data (split, alignment, RAPT offset refit on train files) | - |
| E4-T6 | wait for a free GPU, train (background) + `watchdog_stop_on_plateau.py` (plateau stop + one-shot test eval) | - |
| E4-T7 | report `MF-PAM_PTDB_EXP4.md` (test best-validation checkpoint, run once) | - |
