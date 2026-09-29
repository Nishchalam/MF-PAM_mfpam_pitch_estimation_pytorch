# 06 — Experiment MF-PAM-PTDB-003: clean+noisy training (matches the authors' 90/10 mix)

STATUS: APPROVED via explicit user instruction 2026-09-27 ("train on noisy+clean data rather than just clean vocals,
since thats what the authors had done"). Supersedes Q2 of `specs/03_experiment.md` for this run only;
MF-PAM-PTDB-001 (clean-only, already reported) is kept as-is for comparison, not overwritten.

## What the official code does (dataset.py `F0Dataset.__getitem__`, train=True)
```
noise = noisy - clean                                  # per-file, only when a matched noisy/clean pair is supplied
if rir_dir: fb_noisy = mixture of noise/reverb/both     # DEAD in this repo (specs/01 Defects #2) - not used
else:       fb_noisy = noisy
sources = Shift(8000, same=True)(stack([fb_noisy, clean]))     # same temporal crop+shift applied to both
noisyaudio = np.random.choice([fb_noisy, clean], p=(0.9, 0.1))  # MODEL INPUT: 90% noisy, 10% clean
cleanaudio = clean                                               # LABEL SOURCE: always clean (DIO run on this)
```
So: labels are always DIO on the clean signal; the model's input is the noisy file 90% of the time and the clean
file 10% of the time; both share one crop position and one shift offset (temporal alignment preserved either way).

## What MF-PAM-PTDB-003 does
Same 90/10 input mix and the same shared-crop/shared-shift construction, implemented on top of the existing DIO/
label/chunking pipeline of `specs/03` (unchanged): 16 kHz audio, 4.5 s crop/1 s stride, `augment.Shift(8000)`,
DIO labels (8 ms hop) on the CLEAN crop, official `hz_to_onehot` quantiser. Only the TRAIN loader changes; validation
and test are unaffected (validation selection stays on clean audio, as in specs/03; test already reports clean +
noisy SNR 20/10/5/0 as secondary conditions).

## Deviations from the authors' recipe (register, in addition to specs/03's D1-D17)
- **D18 (noise corpus/SNR):** the authors mix in NOISEX-92 noise at SNR drawn from {-7,-2,3,8,13} dB (train).
  We only have the dataset's own pre-generated noisy files at **SNR {20,10,5,0} dB** (noise type/source: UNKNOWN,
  see `specs/02`). "Noisy" in MF-PAM-PTDB-003 = one of these 4 files, chosen uniformly at random per crop.
- **D19 (RIR):** not used (the official RIR path is dead code, specs/01 Defects #2; unchanged from specs/03).
- **D20 (RNG):** the official `np.random.choice(p=(0.9,0.1))` call is unseeded (specs/01). We use Python's seeded
  `random` module (seeded per DataLoader worker via `worker_init_fn`, itself derived from the global seed 42) for
  both the noisy/clean choice and the SNR pick, so the run is repeatable; this is a deviation (seeded vs. official
  unseeded), consistent with D3/D7 already logged for MF-PAM-PTDB-001.
- Everything else (architecture, loss, optimiser, scheduler, batch size, chunking, shift augmentation, DIO label
  recipe, quantiser, checkpoint selection on validation, test-set treatment) is unchanged from `specs/03`.

## Acceptance criteria (in addition to specs/04)
- [ ] E3-1 every training batch's model input is drawn 90/10 noisy/clean (checked empirically over >=2000 crops, tolerance +/-3 pts)
- [ ] E3-2 labels (DIO) are always computed from the CLEAN crop regardless of which audio was fed to the model
- [ ] E3-3 noisy and clean crops share the same start offset and the same post-Shift alignment
- [ ] E3-4 train loader only reads TRAIN speakers' clean+noisy files; validation/test unaffected
