# 01 — Original MF-PAM implementation (as actually present, commit 9303df5)

Sources: SOURCE_CODE / CONFIG / README / PAPER / INFERRED / UNKNOWN.
Model facts marked "(verified)" were checked by instantiating the model on CPU (no repo files changed).

## Execution paths
| Path | Files | Status |
|---|---|---|
| **Primary (paper): quantised, BCE** | train.py, model.py, dataset.py, module.py | Runs, with caveats (see Defects). README + paper describe it. |
| Direct F0, L1 | train_direct.py, model_direct.py | **Broken as shipped** (see Defects). README says "preferred for VAD" but paper does not evaluate it. |
Primary reproduction path = `train.py` / `model.py` (quantised). (INFERRED from README + PAPER + runnable state.)

## Architecture (model.py, module.py) — `Estimation_stage`
```
wav [B,T] -> unsqueeze [B,1,T] -> per-utterance std normalisation
-> zero-pad to valid_length -> upsample2 x2 -> upsample2 x2 (x4 total, sinc/Hann, zeros=56)
-> PNP-Conv1 (1->6)  -> PNP-Conv2 (6->12)          [p1 = out of PNP2 : 12ch]
-> P-Conv3 (12->24) [p2] -> P-Conv4 (24->48) [p3] -> P-Conv5 (48->96) [p4]
-> 2-layer unidirectional LSTM(96) [p5]
-> Light_BiFPN(num_channels=48) on (p1..p5)
-> concat 5x48=240ch -> Linear(240->360) -> sigmoid   => [B, frames, 360]
```
| Item | Value | Source |
|---|---|---|
| Input | raw mono waveform, 16 kHz | SOURCE_CODE/CONFIG |
| Input normalisation | `wav / (1e-3 + std)` over time, `normalize=True` | SOURCE_CODE |
| Upsampling | sinc interpolation x2 applied twice = x4; `zeros=56`, Hann window | SOURCE_CODE; PAPER (x4) |
| PNP-Conv (blocks 1,2) | P branch: Conv1d(k=4,s=4)+ReLU+Conv1d(1x1)+Snake(a); N branch same with Snake(a=0.2); outputs summed. P Snake a = 17 (blk1), 13 (blk2) | SOURCE_CODE |
| P-Conv (blocks 3,4,5) | Conv1d+ReLU+Conv1d(1x1)+Snake; kernel 8, 8, 12; stride 4; Snake a = 11, 7, 5 | SOURCE_CODE; PAPER (kernels +4,4,8,8,12) |
| Channels | in (1,6,12,24,48) out (6,12,24,48,96) | SOURCE_CODE; PAPER |
| Dilation | 1 (not used) | SOURCE_CODE; PAPER |
| Snake | `x + sin²(a·x)/a` | SOURCE_CODE |
| Weight rescale | `rescale_module(reference=0.1)` on all Conv1d | SOURCE_CODE (Demucs-style) |
| LSTM | nn.LSTM, 2 layers, hidden 96, `bi=False` | SOURCE_CODE |
| Light BiFPN | 48 ch; pre-resize to 500 frames/4 s via avg-pool (32,8,2) / pad / nearest-upsample(2); 4 top-down + 4 bottom-up nodes, fast-normalised learnable weights (eps 1e-8), Swish, depthwise-separable conv k=5 + BatchNorm2d(momentum 0.01, eps 1e-3) | SOURCE_CODE |
| Output | Linear(240→360), sigmoid | SOURCE_CODE |
| Parameters | 362,479 (verified) — matches paper's "0.362 M" | verified; PAPER |
| Output frame rate | 1 frame / 128 samples = 8 ms @16 kHz. 64000 samples → 500 frames (verified) | SOURCE_CODE/CONFIG |
| Output length vs. input | not exactly T/128: 72000→564, 96880→758 (verified); DIO gives ⌊T/128⌋ frames after `[:-1]` (562, 756). Validation code truncates both to min length. | verified |

## F0 representation / quantisation (dataset.py `hz_to_onehot`, train.py `onehot_to_hz`)
| Item | Value | Source |
|---|---|---|
| Bins | 360, 48 bins/octave (25 cent) | SOURCE_CODE; PAPER |
| f_min | 32.7 Hz; range 32.7–≈5836 Hz (paper: 5834.5) | SOURCE_CODE; PAPER |
| F0→bin | `idx = int(log(hz+1e-7 / 32.7)/log(2^(1/48)) + 0.5)` (`.long()` truncates toward zero) | SOURCE_CODE |
| Unvoiced/0 Hz target | all-zero 360-vector (idx<0 masked) | SOURCE_CODE |
| Decoding | `argmax` bin → `32.7·2^(idx/48)`; voiced if max prob > threshold. Default 0.6 (function); **validation uses 0.0**, i.e. every frame is "voiced" | SOURCE_CODE |
| Note | decoded Hz is bin index, not `idx` with +0.5 offset → ≤ half-bin systematic offset is inherent to the code | INFERRED |

## Labels (dataset.py `F0Dataset.__getitem__`)
F0 is **not read from files**. It is computed on the fly with **pyworld DIO** on the *clean* waveform:
`pw.dio(x.float64, 16000, frame_period = 128/16000*1000 = 8 ms)`, then `f0 = f0[:-1]`. DIO output 0 = unvoiced. (SOURCE_CODE; PAPER says DIO for ground truth on VCTK-DT.) DIO is called with default f0_floor=71 Hz / f0_ceil=800 Hz (pyworld defaults) — INFERRED from pyworld defaults, not set in repo. Hence labels are pseudo-labels, and label range is ≈71–800 Hz.

## Data, chunking, augmentation
| Item | Value | Source |
|---|---|---|
| Sampling rate | 16000; files must already be 16 kHz (RuntimeError otherwise, no resampling) | CONFIG; SOURCE_CODE |
| Data listing | `make_data_json.py` → JSON of `[path, num_frames]`; needs clean+noisy trees with **identical sorted order** (pairing = sorted position) | SOURCE_CODE |
| Train chunk | length 4.5 s (72000), stride 1 s (16000), pad=True; every window of every file is an example (cropped/padded via negative/positive `F.pad`) | SOURCE_CODE |
| Shift augmentation | `augment.Shift(8000, same=True)`: random offset in [0,8000) applied jointly to noisy+clean, output length 72000−8000 = 64000 = 4.0 s → 500 frames | SOURCE_CODE |
| Noisy/clean mix | 90 % noisy input, 10 % clean input (`np.random.choice`, **unseeded**) ; target always from clean | SOURCE_CODE |
| Reverb (RIR) | only if `--rir_dir` given; code reads `train.csv`/`test.csv` as bare names → NameError (bug). 0.3 noise / 0.3 rir / 0.4 both | SOURCE_CODE |
| Noise/RIR corpora | paper: MIT IR Survey + NOISEX-92; SNR ∈ {−7,−2,3,8,13} dB train, {−5,0,5,10,15} test. Repo ships no generation script | PAPER; UNKNOWN (recipe/files) |
| Validation/test item | whole file, no chunking (`split=False`, length None) | SOURCE_CODE |
| Shuffling | `shuffle=False` for dataset and DataLoader → fixed sorted order every epoch | SOURCE_CODE |

## Training (train.py, config.json)
| Item | Value | Source |
|---|---|---|
| Loss | `nn.BCELoss` over 360 sigmoid outputs, mean over all elements | SOURCE_CODE; PAPER (Eq. 4, sum) |
| Optimiser | Adam, lr 3e-4, betas (0.9, 0.999), no weight decay | CONFIG/SOURCE_CODE |
| Scheduler | `ExponentialLR(gamma=0.999)`, stepped per epoch | CONFIG/SOURCE_CODE |
| Batch size | 64 | CONFIG |
| Epochs | `--training_epochs` default 3100 | SOURCE_CODE (paper: UNKNOWN) |
| Gradient clipping / AMP / weight decay | none | SOURCE_CODE |
| Seed | 1234 (`torch.manual_seed`, `cuda.manual_seed`, `random.seed` in dataset). numpy RNG unseeded; cudnn.benchmark=True → not bit-reproducible | CONFIG/SOURCE_CODE |
| Workers | 4 train, 1 validation | CONFIG/SOURCE_CODE |
| Checkpointing | every 2000 steps `g_{step:08d}` (model, optimiser, steps, epoch); **no best-checkpoint logic** | SOURCE_CODE |
| Logging | TensorBoard; stdout every 5 steps | SOURCE_CODE |
| Hidden state (LSTM) | reset each batch | INFERRED |

## Validation procedure (as shipped)
Every 2000 steps: model.eval(), batch 1 over the *"test"* JSON (`get_dataset_filelist` returns
`[test_clean, test_noisy]` as `valdata`). Logs BCE, MAE (Hz, on ref-voiced frames), RPA×100, RCA×100
to TensorBoard. **The repo's "validation" set is its test set and is never used for selection** (paper: 3:1:1 train/val/test; how the validation split was used = UNKNOWN).

## Evaluation procedure (as shipped)
- decode with threshold 0.0 (pure pitch accuracy in voiced region).
- **Reference-voiced frames only** (ref F0 == 0 frames deleted from both ref and estimate), then `mir_eval.melody.to_cent_voicing` + `evaluate` → RPA, RCA (50-cent tolerance = mir_eval default; PAPER: 50 cents).
- Time axes built with `hop_length=128, sr=16000`; metrics computed **per file and averaged over files** (not frame-weighted).
- No OA, VRR, VFA, no separate voicing detector in the code path (README: direct model "gives more accurate VAD" — no code evaluating it).
- Estimate on noisy input, ref from clean DIO.

## Defects / inconsistencies found in the shipped code
1. `train_direct.py`: uses `h.segment_size, h.n_fft, h.num_mels` (absent from config.json) and calls `F0Dataset(traindata, segment_size, n_fft, num_mels, hop_size, …)` which does not match `F0Dataset.__init__`; loss uses undefined `onehot_hat` (should be `f0_hat`); `f0_hat_vad` used before definition in validation. → direct path cannot run unmodified. `model_direct.py`: `Linear(40→1)` after `avg_pool1d(…,6)` on 240 ch (output [B,T] in Hz, no activation, unbounded).
2. `train.py` `--rir_dir` default is a placeholder string (not None) → NameError (`train.csv`) unless code changed; RIR reverb path is dead.
3. Validation data = test data; no best-model selection.
4. Comment says 4.5 s chunk vs "4 s → 500 frames": both true (4.5 s crop, 4.0 s after shift).
5. `train.py` prints/undefined `cp_g` only defined if checkpoint dir exists (it does after makedirs).
6. `requirements.txt` lists no `tensorboard`, `matplotlib` (imported).
7. Training ordering is sorted/no shuffle (see above).
