# 03 — Experiment MF-PAM-PTDB-001

```
Dataset: PTDB-TUG (user copy, 10 ms hop)   Split: existing speaker-disjoint 60/20/20 (12/4/4 speakers)
Model: MF-PAM (model.py, quantised/BCE path, unchanged)   Input: raw waveform
Training: TRAIN speakers only   Validation: VALIDATION speakers only (checkpoint selection)
Testing: TEST speakers only, once, best-validation checkpoint
```
Status: DRAFT — items marked **[DECISION]** need the user's approval; items marked UNKNOWN stay UNKNOWN.

## Official vs. PTDB experiment
| Parameter | Official MF-PAM | PTDB experiment | Source | Reason |
|---|---|---|---|---|
| Architecture | model.py (362,479 params) | identical, no edits | SOURCE_CODE | constraint 6 |
| Input | raw waveform 16 kHz | raw waveform 16 kHz | CONFIG | same |
| Sampling rate | 16 kHz (files must be pre-resampled) | resample 48→16 kHz (**[DECISION]** method: on-disk 16 kHz cache via a fixed resampler, e.g. torchaudio/soxr-HQ) | CONFIG / dataset spec | model requires 16 kHz; PTDB is 48 kHz. Paper also "resampled to 16 kHz" (PAPER) |
| Output frame hop | 128 samples = 8 ms | 128 samples = 8 ms (architecture-fixed) | SOURCE_CODE | changing needs architecture change (stop condition) |
| F0 label source | pyworld DIO pseudo-labels on clean audio, 8 ms | **[DECISION]** PTDB RAPT reference F0 (col 0) resampled to 8 ms grid | dataset spec | PTDB has laryngograph ground truth, needed for comparability with other estimators. Deviation D1 |
| Label hop | 8 ms (DIO frame_period) | 10 ms → 8 ms by interpolation | | D2 |
| Label range | DIO 71–800 Hz | RAPT 53–535 Hz | | D1 |
| Label→target | `hz_to_onehot` (360 bins, 25 cent, fmin 32.7) | same function, imported unchanged | SOURCE_CODE | reuse official quantiser |
| F0 representation | quantised 360 bins, sigmoid | same | SOURCE_CODE | primary path |
| Loss | BCE (mean) | same | SOURCE_CODE | |
| Optimiser | Adam lr 3e-4, β (0.9,0.999) | same | CONFIG | |
| Scheduler | ExponentialLR 0.999/epoch | same | CONFIG | |
| Batch size | 64 | 64 | CONFIG | |
| Epochs | default 3100 (paper UNKNOWN) | **UNKNOWN → [DECISION]**: fixed cap chosen *a priori* (proposal: 3100 cap with keep-best-by-validation; early stop patience UNKNOWN in original → would be a deviation) | SOURCE_CODE | official default not tractable to verify; must not be tuned on test |
| Train chunk | 4.5 s crop, stride 1 s, then Shift(8000) → 4.0 s | same | SOURCE_CODE | |
| Shuffle | none (sorted order) | **[DECISION]** keep no-shuffle (faithful) vs. shuffle (standard). Proposal: shuffle with seeded generator, documented deviation D3 | SOURCE_CODE | no-shuffle over speaker-sorted data is unusual; needs your call |
| Augmentation | shift 8000; 90 % noisy / 10 % clean; RIR (dead code) | shift 8000 kept. Noisy: **[DECISION]** see Q2 | SOURCE_CODE | |
| Noisy data | paper: NOISEX-92 + MIT-IR, SNR −7…13; recipe UNKNOWN | user's precomputed noisy set, SNR {0,5,10,20} dB, noise type UNKNOWN; **or** clean-only training | dataset spec | paper's noise recipe not available. D4 |
| Train split | paper 3:1:1 (split identity UNKNOWN) | TRAIN speakers F01–F06, M01–M06 | dataset spec | user's authoritative split |
| Validation split | repo: uses *test* JSON as "validation", no selection | VALIDATION speakers F07,F08,M07,M08; **select best checkpoint on validation** | dataset spec; CLAUDE.md #4 | D5 (necessary for constraint 4) |
| Validation metric for selection | none | **[DECISION]** validation BCE (matches training objective) vs. validation RPA. Proposal: RPA on validation (clean), tie → lower BCE | | |
| Test split | repo test JSON | TEST speakers F09,F10,M09,M10, once | dataset spec | |
| Speaker-disjoint | not enforced/UNKNOWN | enforced + verified | dataset spec | |
| Eval input | noisy test input vs clean DIO | **[DECISION]** clean test audio (paper Table 1 = clean PTDB); optionally also snr20/10/5/0 as secondary reports | PAPER | |
| Voicing decision | threshold 0.0 (every frame voiced), RPA on ref-voiced frames | keep official RPA/RCA protocol AND add OA/VRR/VFA via `mir_eval.melody` with decision threshold 0.6 (function default) — **[DECISION]** and unfixed threshold choice must be made on validation | SOURCE_CODE | official code has no voicing metrics; user needs OA/VRR/VFA. D6 |
| RPA/RCA tolerance | 50 cents | 50 cents | PAPER/mir_eval | |
| Metric aggregation | per-file mean | report both per-file mean (official) and pooled frame-level | SOURCE_CODE | comparability with other estimators = pooled |
| Seed | 1234 | 1234 + record; numpy/torch/cudnn seeded, workers seeded (deterministic where feasible) | CONFIG | original numpy RNG unseeded → D7 |
| Logging/checkpoints | TB, every 2000 steps | + per-epoch CSV, best-val checkpoint, manifest | | additive only |

## Deviation register
D1 labels (DIO→PTDB RAPT) · D2 label hop 10→8 ms grid · D3 shuffling (if approved) · D4 noisy-data recipe ·
D5 validation-based selection · D6 added voicing metrics · D7 seeding · D8 audio resampling 48→16 kHz ·
D9 epochs/stopping rule · D10 `train.py` bug fixes needed to run at all (rir_dir; requirements).

## Specification conflicts (code vs. paper vs. repo docs)
1. CODE(train.py) "validation" = test set; PAPER: separate validation split. → new validation-based flow.
2. CODE labels = DIO everywhere; PAPER: DIO for VCTK; for PTDB/MDB/MIR "reference pitch/annotations" (paper text) — ambiguous whether DIO was used on PTDB. README silent.
3. README says direct/L1 is "preferred" (better VAD); paper and default path are BCE; direct path does not run. Primary = BCE.
4. CODE default 3100 epochs vs. paper epochs UNKNOWN.
5. CODE noise: 90/10 mix; PAPER SNR sets; no generator in repo. Paper test SNR set {−5,…,15}; user's set {0,5,10,20}.
6. PAPER RPA/RCA definitions as fractions of *voiced* frames; CODE deletes ref-unvoiced frames and per-file averages.
7. Code hop 128 (8 ms) vs PTDB label hop 10 ms.

## Stop conditions triggered / relevant (CLAUDE.md, user prompt §25)
- #3 contradictory training implementations (train.py vs train_direct.py) → resolved by choosing train.py; needs confirmation.
- #4 original hyperparameters partly unknown (epochs, noise recipe).
- #2 F0-audio alignment origin not established (Q3).
- #6 label source/grid changes affect comparability with published MF-PAM numbers.

---
## Decisions received from user (2026-09-25) — these SUPERSEDE the [DECISION] cells above
| Q | Decision | Consequence |
|---|---|---|
| Q1 | Labels = pyworld DIO on the PTDB audio, as in the official code (16 kHz, 8 ms hop, `f0[:-1]`, pyworld default floor/ceil) | D1 becomes "none" w.r.t. label source; PTDB RAPT `.f0` files are NOT used for training. Whether test reference = DIO (official) or RAPT: **pending Q8** |
| Q2 | Train on clean audio only (no 90/10 noisy mix); test on clean, plus noisy SNR 20/10/5/0 dB as secondary | D4 becomes "clean-only training"; noisy files used only at evaluation |
| Q3 | Labels are 10 ms hop | Empirical alignment check done (see 02): RAPT ref frame i ≈ audio time (i+2…3)·10 ms |
| Q4 | Best checkpoint by validation | as D5 |
| Q5 | Seeded shuffle, seed 42 | D3 approved; seed 42 replaces 1234 (D7) |
| Q6 | Select checkpoint by validation RPA. Voicing "threshold 50 cents" needs clarification (**pending Q9**) | |
| Q7 | fork = https://github.com/Woo-jin-Chung/MF-PAM_mfpam_pitch_estimation_pytorch — this is the OFFICIAL repo, not a fork | **pending Q7b** |

---
## Approval + final decisions (2026-09-25) — SPEC APPROVED BY USER
| Q | Final |
|---|---|
| Q7 | Fork the official repo under the user's account (pending: no `gh`/token in this session). Local remotes: `upstream` = official; branch `ptdb-reproduction` created |
| Q8 | Report BOTH references: DIO (official recipe) and RAPT `.f0`; RAPT is reported with fitted time offset 20 ms (`rapt`) and with naive k·10 ms (`rapt_naive`, as in the user's other experiments) |
| Q9 | Voicing threshold **0.5** on the sigmoid max-probability (fixed a priori, not tuned) |
| Selection | best validation **RPA (mir_eval, 50 cents, pooled over ref-voiced frames, DIO reference)**; tie → lower validation BCE. |

### New finding — D11: official RPA/RCA are not 50-cent metrics
`train.py` calls `mir_eval.melody.evaluate(ref_v, ref_c, est_v, est_c)` with the *outputs* of `to_cent_voicing`
(voicing booleans and cents) as if they were (time, frequency). Verified on mir_eval 0.8.2 (tests/test_metrics.py):
an error of 100 cents still counts as correct, tolerance ≈ 150–170 cents at 200 Hz. All code here reports
`official_style` (bug-replicated, per-file mean) **and** correct 50-cent metrics; checkpoint selection and headline numbers use the correct 50-cent version.
Consequently published-style numbers from the official code are optimistic vs. 50-cent RPA.

### RAPT alignment (fitted on 150 TRAIN files only)
RAPT frame i sits at i·10 ms + 20 ms (DIO agreement within 50 cents: 83.0 % at 20 ms vs 56.2 % at 0 ms). Consistent with the 32 ms RAPT window (window-start timestamps).

## Implementation deviations added
D12: DIO labels for validation/test are computed once on the full 16 kHz utterance and cached (deterministic, equals on-the-fly result); training labels are computed on the fly on each shifted crop (official).
D13: `torch.backends.cudnn.benchmark=True` kept → not bit-exact reproducible; seeds (42) are fixed for torch/numpy/python/DataLoader/worker RNGs.
D14: samples with non-finite gradient norm would skip the optimiser step (counted in the log; none occurred in trials).
D15: entry point is `scripts/train_ptdb.py` (official `train.py` left untouched).

---
## Revision 2 (2026-09-25, user request; training restarted from scratch at epoch 0)
Run 1 (stopped at epoch 107, archived in logs/archive/) is superseded: it lacked the requested columns.
### Metric protocols (all 50 cents, voicing threshold 0.5 on confidence)
- **paper_\***: the paper/official-code protocol with the tolerance bug removed: per-file mean, RPA/RCA on reference-voiced frames with pitch decoded for every frame; VRR/VFA/OA from voicing = confidence > 0.5.
- **rmvpe_\***: the RMVPE/RRCGD `optimised_6_acs` evaluation: predicted-unvoiced frames set to 0 Hz before scoring (voicing misses count as pitch errors), per-file mean.
- Each is reported against DIO and RAPT (RAPT with the fitted 20 ms offset; `rapt_naive` only in the final evaluation).
- `official_style` (buggy ~170-cent tolerance) is no longer logged per epoch; it remains in the final evaluation JSON for documentation.
- Checkpoint selection unchanged: best `val_RPA_50c` (pooled, strict, DIO reference, validation speakers).
### D16 — test-set monitoring during training (user-requested deviation from acceptance criterion T-3)
Every 20 epochs the best-by-validation checkpoint and the current (last) checkpoint are evaluated on the TEST speakers (clean audio), written to `test_monitor.csv` + `monitor/*.json`, and committed/pushed automatically (also a 10-minute timer commit of the logs). These numbers are **monitoring only**: they are not used for checkpoint selection, early stopping or any hyperparameter choice; training config is frozen. The final reported test result remains a single evaluation of the best-validation checkpoint after training. Because test curves are visible during training, the final test number should be described as "test-monitored, validation-selected".
