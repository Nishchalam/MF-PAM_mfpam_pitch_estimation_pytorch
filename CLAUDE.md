# CLAUDE.md — MF-PAM on PTDB-TUG (project operating manual)

## Project purpose
This repository adapts the official MF-PAM pitch-estimation implementation
(https://github.com/Woo-jin-Chung/MF-PAM_mfpam_pitch_estimation_pytorch, upstream commit 9303df5)
to an existing speaker-disjoint PTDB-TUG dataset (60/20/20 by speaker).

## Core rule
The files under `specs/` are the source of truth for the experiment. Code must conform to the
approved specifications. Read the relevant spec before modifying any code.

## Scientific constraints (non-negotiable)
1. Never recreate the PTDB split.
2. Never randomly split individual utterances.
3. Never allow speaker overlap between train/validation/test.
4. Never use test data for model selection.
5. Never tune hyperparameters using test results.
6. Preserve the original MF-PAM architecture unless an approved spec explicitly requires a change.
7. Do not silently change sampling rate, F0 representation, loss, chunking, or augmentation.
8. Every deviation from the official implementation must be documented (specs/03_experiment.md table).
9. Unknown parameters are marked UNKNOWN, never guessed.
10. The experiment must be reproducible from its configuration and recorded metadata.

## Workflow
RECONNAISSANCE -> SPECIFICATION -> SPEC REVIEW -> PLAN -> IMPLEMENTATION -> VERIFICATION
-> SMOKE TEST -> TRAINING -> EVALUATION -> REPRODUCIBILITY REPORT

## Approval gates
- STOP at SPEC REVIEW and wait for explicit user approval. No code before spec approval.
- No implementation without an approved plan (`plans/`).
- STOP again before full training if the smoke test fails.

## Source hierarchy (on conflict)
1. Actual repository implementation  2. Repository configuration  3. Repository documentation
4. MF-PAM paper  5. Explicit PTDB requirements  6. Explicitly documented assumptions.
Never silently resolve conflicts; list them in specs/03_experiment.md.

## Specs (authoritative)
specs/00_project.md, specs/01_original_mfpam.md, specs/02_ptdb_dataset.md,
specs/03_experiment.md, specs/04_acceptance_criteria.md

## Plans
`plans/MF-PAM-PTDB-001-plan.md`

## Local facts
- PTDB data (do not reorganise/modify): `/home/batch_2024/ee24s004/KT/RMVPE-RRCGD-spot-computation/dataset/PTDB_data_10ms_hop/{train,valid,test}`
- Hardware: 2x RTX 4090; torch 2.10.0+cu128. `pyworld` is NOT installed in the current env.
- Report RPA/RCA/OA/VRR/VFA to the user as x100 percentages.
- Do not commit datasets, checkpoints, predictions, large logs, venvs, caches.
