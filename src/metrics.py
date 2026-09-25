"""Pitch metrics. Two protocols are provided and always labelled:
  * official_style  - exactly what train.py does: reference-voiced frames only, then
                      evaluate(*to_cent_voicing(...)) i.e. the cent/voicing outputs are fed back into evaluate() as
                      (time, freq). This makes the effective pitch tolerance ~150-170 cents, NOT 50 (see specs/03).
  * mir_eval 50c    - standard mir_eval.melody (50 cent tolerance); RPA/RCA on reference-voiced frames,
                      VRR/VFA/OA using the predicted voicing (confidence > threshold).
"""
import numpy as np
import mir_eval

FMIN, BPO = 32.7, 48
HOP, SR = 128, 16000


def decode(prob):
    """prob [T,360] -> (hz_raw [T], confidence [T]); same mapping as train.py onehot_to_hz without masking."""
    idx = prob.argmax(1)
    return (FMIN * 2.0 ** (idx / BPO)).astype(np.float64), prob.max(1).astype(np.float64)


def _t(n):
    return np.arange(n) * HOP / SR


def official_style(ref, est):
    keep = ref > 0
    if keep.sum() == 0:
        return {'RPA': np.nan, 'RCA': np.nan}
    r, e = ref[keep].astype(np.float64), est[keep].astype(np.float64)
    t = _t(len(r))
    v, c, ev, ec = mir_eval.melody.to_cent_voicing(t, r, t, e)
    m = mir_eval.melody.evaluate(v, c, ev, ec)
    return {'RPA': m['Raw Pitch Accuracy'], 'RCA': m['Raw Chroma Accuracy']}


def melody(ref, est_hz, conf, thr=0.5):
    """mir_eval.melody with predicted voicing = conf > thr (unvoiced frames keep their pitch as negative Hz)."""
    ref = ref.astype(np.float64)
    est = np.where(conf > thr, est_hz, -est_hz)
    t = _t(len(ref))
    m = mir_eval.melody.evaluate(t, ref, t, est)
    return {'RPA': m['Raw Pitch Accuracy'], 'RCA': m['Raw Chroma Accuracy'], 'VRR': m['Voicing Recall'],
            'VFA': m['Voicing False Alarm'], 'OA': m['Overall Accuracy']}


def rmvpe_style(ref, est_hz, conf, thr=0.5):
    """Protocol of the RMVPE/RRCGD experiment (optimised_6_acs evaluate.py): predicted-unvoiced frames are set to 0 Hz
    BEFORE scoring, so a missed voiced frame is also a pitch error; correct mir_eval calls, 50 cents."""
    from mir_eval.melody import (to_cent_voicing, raw_pitch_accuracy, raw_chroma_accuracy, overall_accuracy,
                                 voicing_recall, voicing_false_alarm)
    est = np.where(conf > thr, est_hz, 0.0)
    t = _t(len(ref))
    rv, rc, ev, ec = to_cent_voicing(t, ref.astype(np.float64), t, est)
    return {'RPA': raw_pitch_accuracy(rv, rc, ev, ec), 'RCA': raw_chroma_accuracy(rv, rc, ev, ec), 'OA': overall_accuracy(rv, rc, ev, ec),
            'VRR': voicing_recall(rv, ev), 'VFA': voicing_false_alarm(rv, ev)}


def summarise(utts, ref_key, thr=0.5):
    """utts: list of dicts with numpy 'est_hz','conf' and reference arrays (same length). Returns pooled and per-file-mean metrics."""
    refs = [u[ref_key] for u in utts]
    est = [u['est_hz'] for u in utts]
    conf = [u['conf'] for u in utts]
    R, E, C = np.concatenate(refs), np.concatenate(est), np.concatenate(conf)
    out = {'n_utts': len(utts), 'n_frames': int(len(R)), 'n_ref_voiced': int((R > 0).sum())}
    out['pooled_50c'] = melody(R, E, C, thr)
    out['pooled_official_style'] = official_style(R, E)
    pf = [melody(r, e, c, thr) for r, e, c in zip(refs, est, conf) if (r > 0).any()]
    out['perfile_50c'] = {k: float(np.mean([p[k] for p in pf])) for k in pf[0]}
    of = [official_style(r, e) for r, e in zip(refs, est)]
    of = [o for o in of if not np.isnan(o['RPA'])]
    out['perfile_official_style'] = {k: float(np.mean([o[k] for o in of])) for k in ('RPA', 'RCA')}
    rm = [rmvpe_style(r, e, c, thr) for r, e, c in zip(refs, est, conf) if (r > 0).any()]   # per-file mean, files with no voiced ref skipped
    out['perfile_rmvpe_style'] = {k: float(np.mean([m[k] for m in rm])) for k in rm[0]}
    return out


def lean_summary(utts, ref_key, thr=0.5, pooled=False):
    """Per-epoch / monitoring summary for one reference.
      paper_*  : paper/official-code protocol at a correct 50-cent tolerance - per-file mean, RPA/RCA on reference-voiced
                 frames with the pitch decoded for every frame (voicing-independent); VRR/VFA/OA use conf > thr.
      rmvpe_*  : RMVPE/RRCGD evaluation protocol - predicted-unvoiced frames set to 0 Hz before scoring, per-file mean.
      pooled_RPA_50c (optional): frame-pooled strict 50-cent RPA (checkpoint-selection metric, DIO reference).
    Files without any voiced reference frame are skipped."""
    refs = [u[ref_key] for u in utts]
    keep = [i for i, r in enumerate(refs) if (r > 0).any()]
    P = [melody(refs[i], utts[i]['est_hz'], utts[i]['conf'], thr) for i in keep]
    R = [rmvpe_style(refs[i], utts[i]['est_hz'], utts[i]['conf'], thr) for i in keep]
    out = {}
    for k in ('RPA', 'RCA', 'VRR', 'VFA', 'OA'):
        out[f'paper_{k}'] = float(np.mean([p[k] for p in P]))
        out[f'rmvpe_{k}'] = float(np.mean([p[k] for p in R]))
    if pooled:
        out['pooled_RPA_50c'] = melody(np.concatenate(refs), np.concatenate([u['est_hz'] for u in utts]),
                                       np.concatenate([u['conf'] for u in utts]), thr)['RPA']
    out['n_files'] = len(keep)
    return out
