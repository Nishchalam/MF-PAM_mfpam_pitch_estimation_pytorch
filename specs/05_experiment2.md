# 05 — Experiment MF-PAM-PTDB-002: MF-PAM inside the RMVPE-RRCGD protocol

STATUS: APPROVED 2026-09-25 (batch size 8; noisy test conditions included; commit at the end of training only, no test monitoring). Experiment 1 (MF-PAM-PTDB-001, `specs/03`) keeps running unchanged.

## Purpose
Everything of the RMVPE-RRCGD PTDB pipeline (`RMVPE-RRCGD-spot-computation/mel_check_for_audio_chunks/optimised_6_acs_rrcgd_ptdb_10ms_seeded`)
is kept; only the network is replaced by the unmodified MF-PAM (`model.py`, 362,479 params). Result column to compare: **RMVPE-style RPA against RAPT** on TEST.

## User decisions (2026-09-25)
1. Hop 10 ms via option **A**: audio resampled to **12.8 kHz** so that MF-PAM's fixed 128-sample hop = 10 ms; architecture untouched.
2. Protocol = train on TRAIN speakers; select best checkpoint on VALIDATION speakers; report TEST for that checkpoint.
3. Epochs / batch size / optimiser as in the RMVPE-RRCGD pipeline (user: fairer). Loss = plain BCE (no focal / no positive weighting).

## Parameter table (RMVPE-RRCGD source → Exp 2)
| Parameter | RMVPE-RRCGD (source) | Exp 2 | Reason |
|---|---|---|---|
| Model | E2E (RRCGD front end + RMVPE net) | MF-PAM `Estimation_stage`, unmodified | the only intended change |
| Audio sample rate | 8 kHz (constants/EXP_SAMPLE_RATE) | **12.8 kHz** (soxr VHQ from 48 kHz) | MF-PAM hop = 128 samples ⇒ 10 ms only at 12.8 kHz (arch fixed) — D-E2-1 |
| Hop | 10 ms | 10 ms (128 samples @12.8 kHz) | matched |
| Input | complex roots of 32 ms window (RRCGD) | raw waveform | model-defined |
| Labels | `<utt>.npy`: RAPT `.f0` col 0 sampled at k·10 ms (nearest), one value/frame, Hz, 0=unvoiced | same files, same frame index (frame j ↔ `.npy[j]`) | matched; RAPT time-origin convention identical to RMVPE (naive k·10 ms) |
| Target | one-hot, 360 classes, 20-cent bins, `idx=round((1200·log2(hz/10)−1997.3794)/20)`, unvoiced/out-of-range = all zeros | same formula (copied verbatim) | matched (MF-PAM's 25-cent quantiser NOT used) — D-E2-2 |
| Output head | (RMVPE) | MF-PAM Linear(240→360)+sigmoid | model-defined; 360 outputs = 360 classes |
| Loss | `FL(alpha,gamma=0)` = weighted BCE-with-logits | BCE (alpha=1, gamma=0) on MF-PAM sigmoid outputs | user: no focal; equal in value to logits-BCE with alpha=1 |
| Optimiser | Adam lr 1e-3 (train_single.py) | same | matched |
| LR schedule | StepLR(step=5 epochs, gamma 0.98), stepped per iteration | same | matched |
| Batch size | EXP_BATCH_SIZE ∈ {8,16,32} (default 8) | 8 (user decision Q-A) | |
| Chunk | seq_l 2.55 s → 256 frames; non-overlapping chunks + last-256 remainder chunk | 256 frames ↔ 32768 samples @12.8 kHz, same chunking (MF-PAM yields exactly 256 frames, verified) | matched |
| Augmentation | none | none (official MF-PAM shift NOT used) | matched — D-E2-3 |
| Training data | clean only, TRAIN speakers, shuffle seed 42 (torch.Generator + seeded workers) | same | matched |
| Gradient clip | norm 3 | norm 3 | matched |
| Mixed precision | AMP on CUDA | **off (fp32)** | `nn.BCELoss` on sigmoid outputs is not autocast-safe; documented — D-E2-4 |
| Epochs | 30 max (`len(loader)*30` iterations) | 30 max | matched |
| Validation | every epoch, RMVPE-style metrics on VALIDATION speakers | same | matched (pipeline has a `valid/` split) |
| Selection | validation RPA (RMVPE-style), `>=`, keep newest best | same | matched |
| Early stopping | patience 4 epochs after best | same | matched |
| Seed | 42, deterministic | 42, deterministic where supported (cudnn.deterministic) | matched |
| Decoding | `to_local_average_cents(salience, None, 0.5)`: weighted average of ±4 bins around argmax, 0 if max ≤ 0.5; `f = 10·2^(c/1200)` | same function (copied verbatim, cents map `linspace(0,7180,360)+1997.3794`) | matched |
| Reference in eval | label one-hot decoded with the same function (i.e. quantised to the 20-cent grid) | same | matched |
| Metrics | mir_eval RPA, RCA, OA, VR(=VRR), VFA via `to_cent_voicing` + `raw_pitch_accuracy(ref_v,ref_c,est_v,est_c)`, per-file mean, 50 cents | same | matched (this is the correct mir_eval usage) |
| Frame alignment in eval | `pitch_pred[:T]`, T = label length | same | matched |
| Test | best-validation checkpoint, once | same on clean + noisy SNR 20/10/5/0 (user Q-B). No test monitoring during training (Q-C); results committed once after training | |

## Consequences / risks (to be documented in the report)
- Sample rate 12.8 kHz ≠ 8 kHz of the RMVPE-RRCGD model and ≠ 16 kHz MF-PAM was designed for (Snake/kernel choices tuned at 16 kHz). Nyquist 6.4 kHz covers the whole label range (≤ 2 kHz).
- Label/audio phase: RMVPE frame k = window starting at k·10 ms; MF-PAM frame j has a different (causal, left-aligned) receptive field. The identical label index is used, a fixed offset (~16–20 ms, cf. specs/02) is learnable by MF-PAM's large receptive field; this is not corrected by shifting. Alternative (shift labels) rejected as it would break "everything else matched".
- MF-PAM emits 1–4 frames more than `.npy` has (e.g. 688 vs 685): truncated to label length exactly as RMVPE does.
- lr 1e-3 (3× the MF-PAM default) and 30 epochs / patience 4 are RMVPE's, possibly suboptimal for MF-PAM; no MF-PAM-specific tuning will be done (would need validation-only tuning and a new experiment ID).
- fp32 instead of AMP: numerics only.

## Acceptance criteria (Exp 2)
- [ ] E2-A1 `model.py`/`module.py` unchanged vs upstream 9303df5; params 362,479
- [ ] E2-A2 train/valid/test speakers = specs/02; no overlap; clean files only in training
- [ ] E2-A3 all 12.8 kHz audio finite; frames(audio)/128 ≥ len(`.npy`) for every file; no file with |diff|>5 frames
- [ ] E2-A4 label formula reproduces RMVPE's target tensor on ≥ 20 files (compare with their `_frame_cache_*NCLASS360` chunk `.pt` files, max|diff| = 0)
- [ ] E2-A5 decoding function reproduces RMVPE's decoded F0 on those labels
- [ ] E2-A6 MF-PAM output for a 256-frame chunk is [B,256,360]
- [ ] E2-A7 metric parity: running the RMVPE `evaluate()` logic and ours on the same synthetic predictions gives identical RPA/RCA/OA/VR/VFA
- [ ] E2-A8 best checkpoint chosen on validation RPA only; test read only for the final evaluation (no monitoring, Q-C)
- [ ] E2-A9 seed, git commit, config, environment, dataset stats recorded
