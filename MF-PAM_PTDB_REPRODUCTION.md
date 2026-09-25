# MF-PAM on PTDB-TUG — reproduction report (Experiment 1: MF-PAM-PTDB-001)
Branch `ptdb-reproduction`. Specs: `specs/`. Config: `configs/mfpam_ptdb.yaml`. Experiment 2 (RMVPE-RRCGD protocol) is in `MF-PAM_PTDB_EXP2.md`.

## A. Original MF-PAM (as implemented, upstream 9303df5)
Raw waveform, 16 kHz, x4 sinc upsampling, 2 PNP-Conv + 3 P-Conv blocks (Snake), 2-layer LSTM(96), Light-BiFPN (48 ch), Linear(240->360)+sigmoid; 362,479 parameters (unchanged; `git diff` against upstream is empty).
Quantised F0 (360 bins, 25 cent, 32.7 Hz), BCE, Adam 3e-4, ExponentialLR 0.999/epoch, batch 64, 4.5 s crops then shift(8000) to 4 s, DIO pseudo-labels (8 ms hop). Details: `specs/01_original_mfpam.md`.

## B. Dataset
PTDB-TUG (user copy `PTDB_data_10ms_hop`), speaker-disjoint 60/20/20: train F01-F06,M01-M06 (2832 utts, 6.0 h); validation F07,F08,M07,M08 (942, 1.74 h); test F09,F10,M09,M10 (944, 1.85 h). 48 kHz mono; RAPT `.f0` labels (10 ms). Split never modified. `specs/02_ptdb_dataset.md`.

## C. Adaptations
16 kHz resampling (soxr VHQ); labels = official DIO recipe on the PTDB audio; clean-only training; seeded shuffle (seed 42); validation-based checkpoint selection; RAPT time offset 20 ms fitted on train files only; extra metrics; test monitoring (D16); manual early stop (D17). Register: `specs/03_experiment.md` (D1-D17).

## D. Training
Official hyper-parameters (batch 64, Adam 3e-4, exponential decay 0.999, BCE, official chunking and shift augmentation, no grad-clip/AMP), fp32, cudnn.benchmark on. Planned 3100 epochs; **stopped manually after epoch 577** (D17): validation RPA gains per 100 epochs had shrunk to 0.35, 0.20, 0.10, 0.07 points; the decision used validation curves only. No non-finite gradients.

## E. Validation
Every epoch on validation speakers; checkpoint = highest pooled strict 50-cent RPA against DIO (`val_RPA_50c`), ties -> lower BCE. Selected epoch 549 (val RPA_50c 95.02).

## F. Testing
One evaluation of the selected checkpoint on TEST speakers, clean and noisy (SNR 20/10/5/0; noisy input, labels from the clean file). Test data was additionally *monitored* every 20 epochs (`test_monitor.csv`) for the best/last checkpoints, never used for selection, stopping or tuning: describe the result as "test-monitored, validation-selected".

## G. Results (percent; 50 cents; voicing threshold 0.5; per-file means)
`paper` = official-code protocol with the tolerance bug removed (RPA/RCA on reference-voiced frames, voicing-independent). `RMVPE` = RMVPE-RRCGD protocol (predicted-unvoiced set to 0 Hz).

| Cond | DIO paper RPA | RCA | VRR | VFA | OA | RAPT paper RPA | RCA | VRR | VFA | OA | **RAPT RMVPE RPA** | RCA | OA | VRR | VFA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clean | 94.38 | 94.72 | 81.31 | 0.20 | 95.43 | 83.72 | 86.92 | 87.96 | 1.84 | 94.30 | 79.12 | 81.04 | 94.30 | 87.96 | 1.84 |
| snr20 | 92.82 | 93.19 | 80.83 | 0.19 | 95.30 | 83.64 | 86.83 | 87.78 | 1.72 | 94.35 | 78.97 | 80.87 | 94.35 | 87.78 | 1.72 |
| snr10 | 89.96 | 90.34 | 78.36 | 0.16 | 94.62 | 82.84 | 86.02 | 86.11 | 1.38 | 94.25 | 77.36 | 79.20 | 94.25 | 86.11 | 1.38 |
| snr05 | 86.50 | 86.93 | 72.94 | 0.12 | 92.99 | 80.64 | 83.84 | 81.09 | 1.05 | 93.23 | 72.04 | 73.71 | 93.23 | 81.09 | 1.05 |
| snr00 | 77.63 | 78.14 | 57.75 | 0.09 | 88.55 | 73.99 | 77.16 | 65.46 | 0.70 | 89.74 | 55.97 | 57.37 | 89.74 | 65.46 | 0.70 |

Full JSON (pooled, official-style, RMVPE-style, DIO / RAPT / RAPT-naive): `results/MF-PAM-PTDB-001/test_metrics.json`; clean-test frame CSVs: `results/MF-PAM-PTDB-001/predictions/` (local, not committed).
Official-code RPA has an effective ~170-cent tolerance (mir_eval misuse, D11); it appears only in the JSON as `official_style`.

## H. Reproducibility
```
conda activate mfpam        # requirements-lock-mfpam-env.txt
python scripts/verify_ptdb.py && python scripts/prepare_16k.py --splits train validation
python scripts/train_ptdb.py --config configs/mfpam_ptdb.yaml     # stopped at epoch 577 (D17)
python scripts/evaluate_ptdb.py                                    # best checkpoint, test
```
Seed 42; manifest in `results/MF-PAM-PTDB-001/` (config, git commit, environment, dataset summary, logs). Not bit-exact (cudnn.benchmark).

## I. Deviations from an exact reproduction
Paper split / noise / RIR recipe and epoch count unavailable; DIO labels not RAPT; 16 kHz resampling; shuffling; clean-only training; validation selection; manual early stop (D17); test monitoring (D16); tolerance-bug fix in metrics (D11). Model, loss, optimiser, chunking and augmentation are official.
