# 02 — PTDB-TUG dataset (user's copy; authoritative split)

Root: `/home/batch_2024/ee24s004/KT/RMVPE-RRCGD-spot-computation/dataset/PTDB_data_10ms_hop/`
(read-only for this project; do not reorganise). Statistics below were measured on 2026-09-25.

## Directory structure
`{train,valid,test}/` flat directories. Per clean utterance `SPK_utt.{wav,f0,npy}`; noisy copies
`snr{00,05,10,20}_SPK_utt.{wav,f0,npy}`. Also non-audio artefacts to IGNORE: `roots_*` dirs,
`_frame_cache_*` dirs, `0_NCLASS*`, `noise_generation_log.csv`, `validate_extracted_roots.m`, `file_check.py`, `.npy` (aligned labels for the user's other pipeline, not MF-PAM).
Clean file identification rule: filename starts with `[FM]\d\d_` (no `snr` prefix).
Sibling `PTDB_data/` is a 20 ms-hop variant — NOT used.

## Speakers (speaker-disjoint, verified)
| Split | Speakers | # spk | Clean utts | Clean duration | Speaker → files |
|---|---|---|---|---|---|
| TRAIN | F01–F06, M01–M06 | 12 | 2832 | 6.005 h | 236 each |
| VALIDATION (`valid/`) | F07, F08, M07, M08 | 4 | 942 | 1.744 h | F08 has 234 (others 236) |
| TEST | F09, F10, M09, M10 | 4 | 944 | 1.854 h | 236 each |
```
train ∩ validation = ∅   train ∩ test = ∅   validation ∩ test = ∅   (verified from filenames)
```
Ratio 12:4:4 = 60/20/20 by speakers (utterances 60.0/20.0/20.0 %: 2832/942/944 of 4718). Paper cites 4,720 PTDB utts; test+valid+train clean = 4,718 (2 short vs 4,720: F08 has 2 fewer).
Noisy copies exist for every clean file at SNR 0/5/10/20 dB (generation log columns: split, original, generated, target SNR, measured SNR, sample_rate, duration). Noise type/source and RNG seed: UNKNOWN (not in log).

## Audio
48 kHz, mono, 16-bit PCM WAV (all files checked in the clean set: `{48000}`, channels `{1}`); noisy files also 48 kHz. No normalisation applied. MF-PAM needs 16 kHz → resampling required (adaptation).

## F0 labels (`.f0`, RAPT output from laryngograph, per PTDB-TUG report §7.2)
- 4 whitespace-separated columns: (1) F0 Hz, (2) probability of voicing, (3) local RMS, (4) normalised cross-correlation peak. Column 0 is F0; **0.0 = unvoiced**.
- Native hop 10 ms, 32 ms analysis window (report).
- Voiced/unvoiced representation: F0 > 0 (column 1 probability is available; whether it disagrees with F0>0 was NOT yet checked → open item).
- Statistics (F0>0 frames, col 0): 
  | Split | frames | unvoiced % | F0 min / max / mean / median (Hz) |
  |---|---|---|---|
  | train | 2,144,858 | 75.4 | 53.3 / 534.8 / 149.9 / 144.1 |
  | valid | 622,202 | 78.1 | 53.4 / 385.3 / 155.6 / 160.5 |
  | test | 661,951 | 78.3 | 53.3 / 346.5 / 152.6 / 149.8 |
  (Unvoiced fraction is high — 75–78 % — unusual for speech; verify against column 1 and the PTDB report before use. Nothing exceeds 535 Hz, so the 360-bin range (32.7–5836 Hz) covers all labels.)
- F0 min ≈ 53 Hz < DIO's default floor (71 Hz) used by the official label generator.

## Temporal alignment (unresolved)
For every clean file: `n_f0_frames = audio_duration/0.010 − 6.0` (range −6.4…−5.6, i.e. 5–6 fewer frames than duration/10 ms) in train, valid and test. So ~60 ms of trailing/leading frames are missing relative to a naive frame-0-at-t=0 10 ms grid; the RAPT frame-time origin (window-centred vs. start) is **not documented in the files**. The user's other pipeline assumes `t_k = k·10 ms` (Generate_Aligned_F0_Labels.ipynb) — this is an assumption, not verified against audio. → Open Question Q3; must be resolved before implementation (acceptance: "Audio/F0 alignment verified").
How audio samples map to F0 frames (proposed, needs approval): frame k ↔ time `k·0.010 s` (assumed); MF-PAM frame j ↔ time `j·0.008 s` (assumed same origin as DIO/official). Map by linear interpolation of voiced F0 onto the 8 ms grid (voicing = nearest frame).

## Alignment check result (2026-09-25, read-only, 60 random train files)
DIO (50–600 Hz, 10 ms hop, on 48→16 kHz audio) vs RAPT column 0, fraction of RAPT-voiced frames within 50 cents when DIO frame = RAPT frame + lag:
lag 0: 51.9 % · lag 1: 70.3 % · **lag 2: 80.4 %** · lag 3: 71.7 % · lag 4: 54.2 %.
So RAPT frame i corresponds to audio time ≈ (i + 2…3)·10 ms (≈ 25 ms), not i·10 ms; consistent with the ~6-frame shortfall. Only matters if RAPT labels are used (they are not for training under Q1).

## Verification still to be automated (T3/T4, scripts/verify_ptdb.py)
sample-rate/channel check on all files; NaN/Inf scan; f0 column-1 vs F0>0 agreement; F0 vs waveform alignment check on spot-sample by an independent pitch tracker.
