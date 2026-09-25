# 00 — Project

## Objective
Run a scientifically defensible MF-PAM baseline on an existing speaker-disjoint PTDB 60/20/20 split.

## Research motivation
Provide an MF-PAM baseline on PTDB-TUG, evaluated on the same speaker-disjoint test speakers
(F09, F10, M09, M10) as the user's other pitch estimators (e.g. RMVPE / RRCGD-RMVPE), with
frame-level predictions exported for later comparison.

## Official repository
https://github.com/Woo-jin-Chung/MF-PAM_mfpam_pitch_estimation_pytorch — cloned at
`/home/batch_2024/ee24s004/KT/MF-PAM`, HEAD `9303df5` ("Update model_direct.py"). Paper:
Chung et al., Interspeech 2023, arXiv 2306.09640. (Fork/`origin`/`upstream`/branch `ptdb-reproduction`
are NOT yet set up: the clone currently has only `origin` = official repo. Pending user's fork URL.)

## Target dataset
PTDB-TUG, user's copy `PTDB_data_10ms_hop` (see 02_ptdb_dataset.md). 20 speakers; existing split
12 train / 4 validation / 4 test speakers.

## Desired comparison
MF-PAM (test speakers only) vs. other estimators on identical test utterances, metrics RPA, RCA,
VRR, VFA, OA, with frame-level CSVs.

## Definition of reproduction
Running the *official implementation unchanged* (model, loss, optimiser, chunking, augmentation,
label-generation, evaluation) on the paper's data recipe. Exact reproduction of paper numbers is
NOT achievable here: paper data splits, noise/RIR recipe, epoch count are not released (UNKNOWN).

## Definition of adaptation
Any change to make the official pipeline consume the user's PTDB split: dataset adapter, label
source/grid, split handling, validation-based checkpoint selection, extra metrics, logging.
Model architecture is NOT adapted. Every adaptation is listed in 03_experiment.md.
Consequently MF-PAM-PTDB-001 is a *faithful adaptation baseline*, not an exact reproduction.
