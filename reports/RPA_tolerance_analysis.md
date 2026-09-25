# The MF-PAM RPA/RCA metric: stated 50-cent tolerance vs. what the official code computes

Data: MF-PAM-PTDB-001 (epoch-549 checkpoint), clean PTDB test set (944 utterances, 4 speakers), DIO reference, per-file mean.
Everything here is reproducible from `results/MF-PAM-PTDB-001/predictions/*.csv` and `tests/test_metrics.py`.

## 1. Claim in one paragraph
The paper states RPA/RCA use a 50-cent tolerance. The official `train.py` passes the *outputs* of `to_cent_voicing` (already in cents) into `mir_eval.melody.evaluate`,
which expects Hz and converts to cents a second time. The tolerance of 50 is then applied on a doubly-converted scale, which corresponds to
about 80-220 true cents depending on pitch (about 137 cents at 150 Hz). Our model scores **94.38 RPA** with a true 50-cent tolerance and **97.14 RPA** with the authors' code
(paper: 97.12). The paper's number is therefore reproduced by the authors' code, not by the 50-cent definition.

## 2. The code
`train.py` lines 197-199 (official repo, upstream commit 9303df5):
```python
ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(cln_time, cleanf0_rpa, est_time, f0_hat_rpa)   # (bool, CENTS, bool, CENTS)
rpa = mir_eval.melody.evaluate(ref_v, ref_c, est_v, est_c)['Raw Pitch Accuracy']                             # expects (ref_time, ref_freq_HZ, est_time, est_freq_HZ)
rca = mir_eval.melody.evaluate(ref_v, ref_c, est_v, est_c)['Raw Chroma Accuracy']
```
mir_eval 0.8.2 (`mir_eval/melody.py`): `evaluate(ref_time, ref_freq, est_time, est_freq, ...)` (line 770) calls `to_cent_voicing` on its four arguments (about line 835);
`to_cent_voicing` converts frequency to cents with `hz2cents` (lines 138, 404-405): `cents = 1200*log2(|f|/10)`;
`raw_pitch_accuracy` (lines 554, 616-617) counts a frame correct if `abs(ref_cent - est_cent) < cent_tolerance` with default `cent_tolerance=50`.
(mir_eval is unpinned in the authors' `requirements.txt`; the API has been stable across releases, but I did not test older versions.)

Consequence: the "frequencies" that `evaluate` receives are `c1 = 1200*log2(f/10)` (thousands of cents), not Hz, and the "times" are booleans.
The boolean time arrays happen to be equal for ref and est (all True after removing unvoiced reference frames), so no resampling is triggered and the code runs silently.

## 3. Mathematics
Let f_r, f_e be reference and estimated F0 in Hz and define, as mir_eval does, c(x) = 1200 log2(x/10).

**Correct computation** (one conversion):  a frame is correct iff |c(f_e) - c(f_r)| < 50, i.e. 2^(-1/24) < f_e/f_r < 2^(1/24) (a constant +/-2.93 % in frequency, at any pitch).

**Authors' computation** (two conversions):  first c1_r = c(f_r), c1_e = c(f_e), then inside `evaluate` c2 = c(c1) = 1200 log2(c1/10).  A frame is correct iff
```
| 1200 log2(c1_e/10) - 1200 log2(c1_r/10) | < 50      <=>      2^(-1/24) < c1_e / c1_r < 2^(1/24)  (= 1.02930)
```
So the condition is a constant *relative* tolerance on the cent value c1, not on frequency. In true cents the allowed error is

```
Delta = c1_e - c1_r  <  (2^(1/24) - 1) * c1_r  =  0.02930 * 1200 * log2(f_r/10)  =  35.16 * log2(f_r/10)   cents   (upper side)
                        (1 - 2^(-1/24)) * c1_r = 34.13 * log2(f_r/10)                                      (lower side)
```
It grows with pitch, because c1 grows with pitch. Evaluated:

| f_r (Hz) | c1 = 1200 log2(f_r/10) | tolerance up (cents) | tolerance down (cents) |
|---|---|---|---|
| 50 | 2786 | 81.6 | 79.3 |
| 100 | 3986 | 116.8 | 113.5 |
| 150 | 4688 | 137.4 | 133.5 |
| 200 | 5186 | 152.0 | 147.6 |
| 300 | 5888 | 172.5 | 167.6 |
| 400 | 6386 | 187.1 | 181.8 |
| 800 | 7586 | 222.3 | 216.0 |

Over the 207,254 voiced frames of our test set (DIO F0, mean 152 Hz) the effective upper tolerance has mean 136, median 139, 5th-95th percentile 109-162 cents,
i.e. about **2.7x** the stated 50 cents (0.9-1.6 semitones instead of 0.5 semitone).
Example: at f_r = 200 Hz an estimate 100 cents sharp (211.8 Hz) is inside the buggy tolerance (152) and outside the correct one (50).

Numerical confirmation (constant 200 Hz reference, estimate shifted by x cents; authors' call vs `evaluate(t, ref, t, est)`):
x = 0, 30, 49 -> both correct;  x = 60, 100, 150 -> authors' call correct, correct call wrong;  x >= 170 -> both wrong.
(`tests/test_metrics.py` asserts the 100-cent case.)

RCA has the same flaw: the octave folding is applied on the doubly-converted scale, so it is not a true chroma accuracy either.

## 4. What each version gives on our model (clean test, DIO reference, percent)
| Metric | RPA | RCA |
|---|---|---|
| Paper, MF-PAM, PTDB (Table 1) | 97.12 | 97.13 |
| **Authors' code as is** (double conversion), per-file mean | **97.14** | **97.15** |
| Authors' code as is, pooled over frames | 97.34 | 97.35 |
| Closed-form re-derivation of the double-conversion rule on saved predictions | 97.22 | - |
| **Corrected code, true 50 cents**, per-file mean | **94.38** | **94.72** |

The closed form (|c(c1_e) - c(c1_r)| < 50 applied by me on the saved CSVs) agrees with the mir_eval run to about 0.1 point
(mean per-file difference 0.07 %); I did not isolate the residual (mir_eval internals in the second pass).

Sensitivity of our RPA to the *true* tolerance (per-file mean):
25 cents 90.12 | **50 cents 94.38** | 75 cents 95.85 | 100 cents 96.64 | 117 cents 97.03 | 137 cents 97.35 | 152 cents 97.56 | 187 cents 97.92 | 250 cents 98.29.
The buggy check acts like a 117-187-cent tolerance for 100-400 Hz voices, and the resulting 97.0-97.9 range contains both the paper's 97.12 and our 97.14.

## 5. Corrected code
Pass Hz, not cents, into `evaluate` (or call the metric functions on the `to_cent_voicing` outputs):
```python
# option A: let evaluate() do the single conversion
scores = mir_eval.melody.evaluate(cln_time, cleanf0_rpa, est_time, f0_hat_rpa)
rpa, rca = scores['Raw Pitch Accuracy'], scores['Raw Chroma Accuracy']

# option B: convert once, then use the metric functions directly
ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(cln_time, cleanf0_rpa, est_time, f0_hat_rpa)
rpa = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)      # default cent_tolerance=50
rca = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
```
(the RMVPE-RRCGD code in `optimised_6_acs/evaluate.py` uses option B.)

## 6. Interpretation and caveats
* The paper's PTDB number is reproduced (97.14 vs 97.12) only with the authors' metric implementation; with the stated 50-cent tolerance the same model gives 94.38. This is strong circumstantial evidence, not proof, that the paper's numbers come from the double-conversion code: I cannot see their runs.
* Our split, label recipe (DIO), clean-only training, 16 kHz resampling and the manually stopped run (epoch 577) differ from the paper's undisclosed setup; the 0.02-point agreement should be read as "consistent", not as identical experiments.
* The tolerance bug affects every model evaluated with this code (including any baselines the authors ran through it), so cross-paper comparisons with strict 50-cent numbers (e.g. CREPE, pYIN in other papers) are optimistic for MF-PAM.
