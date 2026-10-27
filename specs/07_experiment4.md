# 07 — Experiment MF-PAM-PTDB-004: baseline retrained on the NEW 8 kHz PTDB data

STATUS: APPROVED via explicit user instruction 2026-10-26 ("this is for baseline only ... rerun on proper data").
Supersedes MF-PAM-PTDB-003 as the baseline (003 and 001 stay in the repo for history, marked superseded).

## Why
MF-PAM-PTDB-003 (clean+noisy, 93.71 clean-test DIO RPA with the corrected 50-cent metric) was trained and tested on the old
PTDB copy whose noisy files were made by adding noise at 48 kHz and THEN resampling. The new `PTDB_data_8k_10ms_hop` adds noise
after resampling to 8 kHz. Clean-test and noisy-test numbers of 003 are therefore not a valid baseline. The metric itself in 003 is
correct (single cent conversion); the doubly-converted official-code metric exists only as `official_style` in the JSON.

## What is run
Exactly the 003 recipe (specs/03 + specs/06, all unchanged): MF-PAM unmodified (362,479 params), 16 kHz, 8 ms hop, DIO labels on
the clean crop, 4.5 s crops / 1 s stride / Shift(8000) -> 4.0 s, 90% noisy / 10% clean model input (SNR {20,10,5,0} dB files,
uniform), batch 64, Adam 3e-4, ExpLR 0.999, BCE, seed 42, validation-RPA checkpoint selection, whole-utterance inference,
plateau watchdog (two consecutive 100-epoch blocks gaining <= 0.10 pt), corrected 50-cent metrics, test run once at the end.
Config: `configs/mfpam_ptdb_004_8k_noisy.yaml`.

## Only change vs. 003
- **D21 data:** `/media/nishchala/Seagate_2/dataset/PTDB_data_8k_10ms_hop` (8 kHz float32 wavs, clean + `snr{20,10,5,00}_` noisy;
  same speaker-disjoint 60/20/20 split; `.f0` files byte-identical to the old copy, md5-checked). MF-PAM needs 16 kHz, so audio is
  resampled 8 -> 16 kHz (soxr VHQ); content is band-limited to 4 kHz by the data. DIO labels are recomputed on this audio.
  Cache: `/media/nishchala/Seagate_3/MF-PAM_data_8k` (the old cache is untouched).
- UNKNOWN: any other fault in the old data besides the noise/resampling order.

## Acceptance
- [x] E4-1 noisy/clean mix ~90/10 (+/-3 pts), labels from the clean crop, train speakers only (as E3-1..E3-4)
- [x] E4-2 measured SNR of the new noisy files at 16 kHz is close to the target for a sample of files
- [x] E4-3 RAPT offset refit on TRAIN files recorded (`rapt_alignment.json`)
