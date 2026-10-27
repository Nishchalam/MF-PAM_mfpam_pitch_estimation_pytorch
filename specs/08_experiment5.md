# 08 — Experiment MF-PAM-PTDB-005: new 8 kHz PTDB data, 128 ms window

STATUS: ON HOLD (not approved to run; user said the 128 ms window was a side idea, the baseline is MF-PAM-PTDB-004, specs/07). Original request: 2026-10-25 "do MF-PAM on PTDB data over 128ms window" + "retrain on the new PTDB_8k data".
replication gave 93.71 RPA" and "retrain the MF-PAM on the new PTDB_8k data, the current results are on faulty old
data"). The 93.71 figure = MF-PAM-PTDB-003 clean-test DIO RPA with the corrected 50-cent tolerance (`MF-PAM_PTDB_EXP3.md`).
The buggy official-code metric (97.03 pooled / 96.79 per-file for the same model) is NOT used.

## Data
`/media/nishchala/Seagate_2/dataset/PTDB_data_8k_10ms_hop/{train,valid,test}`: 8 kHz float32 wavs (clean + `snr{20,10,5,00}_`
noisy), same speaker-disjoint 60/20/20 split and byte-identical `.f0` files as the old 48 kHz copy (md5-checked for all
three splits). The old `PTDB_data_10ms_hop` results (001-003) are superseded and kept only for history.

## What changes vs. 003 (everything else is 003 unchanged: architecture, BCE, Adam 3e-4, ExpLR 0.999, batch 64, seed 42,
## 90/10 noisy/clean train input, DIO labels, validation checkpoint selection, plateau watchdog, corrected metrics)
- **D21 audio:** 8 kHz -> 16 kHz with soxr VHQ (MF-PAM's 128-sample hop = 8 ms needs 16 kHz). The content is band-limited
  to 4 kHz by the data; DIO labels are recomputed on this 16 kHz audio (`labels_dio`, new cache `MF-PAM_data_8k`).
- **128 ms window:** the model input in training, validation and test is a 128 ms segment (2048 samples = 16 frames).
  * Train: one window per crop start (every 1 s, as official), random start offset 0..8000 samples in whole hops (D23, replaces
    `augment.Shift`); model input 90% noisy / 10% clean, shared start.
  * Validation / test: each utterance is cut into consecutive non-overlapping 128 ms windows (zero-padded tail), every window is
    run independently, frames are concatenated (`src/evaluation.py::forward_windowed`).
- **D22 labels:** training labels = slice of the cached DIO of the CLEAN full utterance (same definition as the val/test
  references) instead of DIO recomputed on a 128 ms crop (DIO has edge artefacts on such short signals).
- Epoch/plateau rule unchanged (watchdog: two consecutive 100-epoch blocks gaining <= 0.10 pt val RPA).
- UNKNOWN / not claimed: what exactly was faulty in the old data (labels are identical; the audio/noisy files differ in rate and
  generation, see the new `noise_generation_log.csv`).

## Acceptance (in addition to specs/04, 06)
- [ ] E4-1 windows feed exactly 16 frames, shapes (64,16,360); labels equal the cached clean-utterance DIO slice at the window grid
- [ ] E4-2 90/10 noisy/clean mix within +/-3 pts; window start identical for noisy/clean; train speakers only
- [ ] E4-3 windowed inference of an utterance == per-chunk model output (no cross-window leakage), length == ceil(T/2048)*16
