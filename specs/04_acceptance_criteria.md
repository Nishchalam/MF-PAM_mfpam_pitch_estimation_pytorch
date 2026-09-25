# 04 — Acceptance criteria for MF-PAM-PTDB-001

Each criterion has an ID (referenced by the plan) and a check method.

## Dataset
- [ ] D-1 No speaker overlap between train/valid/test (script assertion on all three pairs)
- [ ] D-2 Split is 12/4/4 speakers = 60/20/20, matching specs/02 (2832/942/944 clean utts)
- [ ] D-3 All audio files readable, 48 kHz mono (pre-resample) / 16 kHz mono (post)
- [ ] D-4 All `.f0` files readable, 4 columns, finite
- [ ] D-5 Audio/F0 alignment verified (offset convention documented, residual ≤ 1 frame)
- [ ] D-6 Split file lists are generated read-only from directories; no utterance-level random split anywhere in code (grep check)

## Model
- [ ] M-1 `model.py` and `module.py` byte-identical to upstream 9303df5 (git diff empty)
- [ ] M-2 Parameter count = 362,479
- [ ] M-3 Input [B, 64000] → output [B, 500, 360]
- [ ] M-4 Target [B, 500, 360] one-hot/zero rows built with the official `hz_to_onehot`
- [ ] M-5 Round-trip F0→bin→Hz checked on ≥ 5 hand-computed values (error ≤ 12.5 cents + label offset)

## Training
- [ ] T-1 Train loader only yields train speakers (assert on filenames per batch/epoch)
- [ ] T-2 Validation loader only yields validation speakers
- [ ] T-3 Test files never opened before final evaluation (path-access log / code review)
- [ ] T-4 Best checkpoint selected on validation only; selection metric fixed before training
- [ ] T-5 Per-epoch log: epoch, train_loss, val_loss, lr, elapsed_time, RPA, RCA, VRR, VFA, OA

## Numerical stability
- [ ] N-1 No NaN/Inf in inputs; N-2 no NaN/Inf in targets; N-3 no NaN/Inf loss; N-4 gradients finite each step (or counted and reported)

## Smoke test (gate before full training)
- [ ] S-1 load one batch, forward, loss, backward, optimiser step, validation pass, checkpoint save + reload all succeed; shapes printed; reloaded model output identical to saved

## Evaluation
- [ ] E-1 Test evaluation uses the best-validation checkpoint only, once
- [ ] E-2 RPA/RCA reproduce the official definition on a synthetic case (unit test) and OA/VRR/VFA match `mir_eval` on the same case
- [ ] E-3 Frame-level CSV for every test utterance (944) with frame, time, reference_f0, predicted_f0, reference_voiced, predicted_voiced

## Reproducibility
- [ ] R-1 Seed recorded; R-2 git commit + dirty flag recorded; R-3 resolved config saved; R-4 environment (pip freeze, torch/cuda, GPU) recorded; R-5 dataset statistics + split file lists + hash recorded; R-6 manifest at `results/MF-PAM-PTDB-001/` complete
