# Plan — MF-PAM-PTDB-001 (SPEC APPROVED 2026-09-25; implementation status below)

Principles: keep `model.py`, `module.py`, `augment.py`, `utils.py` untouched; put new code in `src/`, `scripts/`, `configs/`; upstream files only patched where unavoidable and recorded in the deviation register. Blocking questions Q1–Q7 (see report) must be answered first.

| ID | Task | Specs | Acceptance |
|---|---|---|---|
| T0 | Git: fork URL → `origin`=fork, `upstream`=official, branch `ptdb-reproduction`; `.gitignore` (data, checkpoints, results, logs) ; add `pyworld`-free env (only needed if DIO labels chosen) + tensorboard/matplotlib | 00 | R-2 |
| T1 | `src/ptdb_dataset.py`: PTDB adapter — read split dirs, filter clean files, load 16 kHz wav + label, chunk 4.5 s/stride 1 s, Shift(8000) via `augment.Shift`, targets via `dataset.hz_to_onehot` (import, no copy); no random splitting | 01, 02, 03 | D-6, M-4, T-1, T-2 |
| T2 | `configs/mfpam_ptdb.yaml`: all experiment values (seed, paths, batch, lr, scheduler, epochs, chunk, augmentation, thresholds, metrics) | 03 | R-3 |
| T3 | `scripts/verify_ptdb.py`: speaker disjointness, counts, SR/channels, NaN/Inf, F0 stats, col-1 vs F0>0 agreement, audio/F0 alignment; writes `dataset_summary.json` | 02 | D-1…D-5, N-1, N-2 |
| T3b | `scripts/prepare_16k.py`: resample 48→16 kHz once to a cache outside the repo (fixed resampler, logged) ; labels → 8 ms grid | 02, 03 (D2, D8) | D-3, D-5 |
| T4 | `src/train_ptdb.py`: validation pipeline (validation speakers only), per-epoch BCE + RPA/RCA/VRR/VFA/OA, best-checkpoint on validation | 01, 03 | T-2, T-4, T-5 |
| T5 | `scripts/smoke_test.py` (batch → fwd → target → loss → bwd → step → val → save → reload; NaN/Inf/shape checks; prints shapes) | 04 | S-1, M-2, M-3, N-1…N-4 |
| T6 | Training logging (CSV + TensorBoard, lr, elapsed, grad-finite counter) | 04 | T-5, N-3, N-4 |
| T7 | Checkpoint management: last + best-val, resume, top-k optional | 04 | T-4 |
| T8 | `src/metrics.py` + `scripts/evaluate_ptdb.py`: official RPA/RCA (per-file + pooled) plus OA/VRR/VFA via mir_eval; test split loaded only here, best checkpoint only; unit tests for metrics and quantiser round-trip | 01, 03, 04 | E-1, E-2, M-5, T-3 |
| T9 | Frame-level export `results/MF-PAM-PTDB-001/predictions/<utt>.csv` (frame,time,reference_f0,predicted_f0,reference_voiced,predicted_voiced) at the model's 8 ms grid (+ optional 10 ms resample for other estimators) | 03 | E-3 |
| T10 | Reproducibility manifest: config.yaml, environment.txt, git_commit.txt, dataset_summary.json, training_log.csv, validation_metrics.csv, test_metrics.json | 04 | R-1…R-6 |
| T11 | `M-1` check (git diff on model files) in CI-like script; single entry `python train.py --config configs/mfpam_ptdb.yaml` | 03 | M-1 |
| T12 | Full training (only after smoke passes) → evaluation → `MF-PAM_PTDB_REPRODUCTION.md` | all | all |

Order: T0 → T2 → T3 → T3b → T1 → T5 → (GATE: smoke pass) → T4/T6/T7 → T12 training → T8/T9 → T10 → report.

## Status (2026-09-25)
Done: T0 (remote/branch; **fork pending token**), T1 `src/ptdb_dataset.py`, T2 `configs/mfpam_ptdb.yaml`, T3 `scripts/verify_ptdb.py` (passed), T3b `scripts/prepare_16k.py`, T4 `src/train_ptdb.py`, T5 `scripts/smoke_test.py` (**PASSED**), T6, T7 (last/best checkpoints, resume), T8 `src/metrics.py` + `scripts/evaluate_ptdb.py` + `tests/test_metrics.py` (passed; pipeline exercised on VALIDATION only with a 3-epoch trial checkpoint), T9 frame CSV export, T10 manifest writer (`src/manifest.py`, written at train start).
Entry point: `python scripts/train_ptdb.py --config configs/mfpam_ptdb.yaml`.
Not done: T12 full training (≈20 s/epoch measured → ≈17 h for 3100 epochs), final test evaluation, `MF-PAM_PTDB_REPRODUCTION.md`. Test split has not been opened by any training/selection code.
