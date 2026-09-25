"""Verify the PTDB split, audio/F0 files and audio<->F0 alignment; writes dataset_summary.json (+ rapt_alignment.json).
Uses TRAIN files only to fit the RAPT time offset. Read-only w.r.t. the dataset."""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, soundfile as sf
from src.ptdb_common import *
import pyworld as pw


def summarise_split(cfg, name):
    items = list_split(cfg, name); dur, nf, f0s, unread, mism, agree, hop = 0.0, 0, [], [], [], [0, 0], cfg['labels']['rapt_hop_s']
    per_spk = {}
    for it in items:
        try:
            i = sf.info(it['wav']); assert i.samplerate == cfg['dataset']['native_sample_rate'] and i.channels == 1, (i.samplerate, i.channels)
            d = np.loadtxt(it['f0'], ndmin=2); assert d.shape[1] == 4 and np.isfinite(d).all()
        except Exception as e:
            unread.append((it['stem'], str(e))); continue
        dur += i.duration; nf += len(d); f0s.append(d[:, 0]); mism.append(len(d) - i.duration / hop)
        agree[0] += int(((d[:, 0] > 0) == (d[:, 1] > 0.5)).sum()); agree[1] += len(d)
        s = per_spk.setdefault(it['speaker'], {'utts': 0, 'hours': 0.0}); s['utts'] += 1; s['hours'] += i.duration / 3600
    a = np.concatenate(f0s); v = a[a > 0]
    return {'speakers': speakers_of(items), 'n_utts': len(items), 'hours': dur / 3600, 'f0_frames': int(nf), 'unreadable': unread,
            'unvoiced_frac': float((a == 0).mean()), 'f0_min': float(v.min()), 'f0_max': float(v.max()), 'f0_mean': float(v.mean()),
            'f0_median': float(np.median(v)), 'frames_minus_dur_over_hop': [float(min(mism)), float(np.mean(mism)), float(max(mism))],
            'voicing_prob_gt0.5_vs_f0gt0_agreement': agree[0] / agree[1], 'per_speaker': per_spk}


def rapt_offset(cfg, n_files=150, seed=42):
    """Fit RAPT frame time origin against DIO(2.5 ms hop) on TRAIN files. Offset o: RAPT frame i sits at i*10ms + o."""
    items = list_split(cfg, 'train'); random.Random(seed).shuffle(items); items = items[:n_files]
    offs = np.arange(0, 0.0601, 0.0025); ok = np.zeros(len(offs)); tot = np.zeros(len(offs)); fp = 0.0025
    for it in items:
        x, sr = sf.read(cache_wav(cfg, it), dtype='float64'); d, _ = pw.dio(x, sr, f0_floor=50, f0_ceil=600, frame_period=fp * 1000)
        r = np.loadtxt(it['f0'], ndmin=2)[:, 0]; t = np.arange(len(r)) * cfg['labels']['rapt_hop_s']
        for k, o in enumerate(offs):
            idx = np.rint((t + o) / fp).astype(int); m = (idx < len(d)) & (r > 0); dd = d[idx[m]]; rr = r[m]
            good = (dd > 0) & (np.abs(1200 * np.log2((dd + 1e-9) / rr)) < 50)
            ok[k] += good.sum(); tot[k] += m.sum()
    acc = ok / tot; best = float(offs[acc.argmax()])
    return {'n_files': len(items), 'offsets_s': offs.tolist(), 'agree_50c': acc.tolist(), 'best_offset_s': best, 'best_agree': float(acc.max()), 'agree_at_0': float(acc[0])}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb.yaml'); a = ap.parse_args()
    cfg = load_config(a.config); out = os.path.join(REPO_ROOT, cfg['paths']['results']); os.makedirs(out, exist_ok=True)
    S = {k: summarise_split(cfg, k) for k in ('train', 'validation', 'test')}
    assert_disjoint({k: v['speakers'] for k, v in S.items()})
    for k in S: assert S[k]['speakers'] == sorted(cfg['dataset']['expected_speakers'][k]), (k, S[k]['speakers'])
    assert all(not S[k]['unreadable'] for k in S), {k: S[k]['unreadable'][:3] for k in S}
    tot = sum(S[k]['n_utts'] for k in S)
    S['split_ratio_utts'] = {k: S[k]['n_utts'] / tot for k in ('train', 'validation', 'test')}
    S['speaker_overlap'] = {'train&validation': [], 'train&test': [], 'validation&test': []}
    json.dump(S, open(os.path.join(out, 'dataset_summary.json'), 'w'), indent=2)
    for k in ('train', 'validation', 'test'):
        v = S[k]; print(k, v['speakers'], v['n_utts'], f"{v['hours']:.3f}h", f"unvoiced {v['unvoiced_frac']:.3f}", 'frames-dur/hop', v['frames_minus_dur_over_hop'], 'voicing agree', round(v['voicing_prob_gt0.5_vs_f0gt0_agreement'], 4))
    print('speaker overlap: none (asserted); ratio', S['split_ratio_utts'])
    al = rapt_offset(cfg); json.dump(al, open(os.path.join(out, 'rapt_alignment.json'), 'w'), indent=2)
    print('RAPT offset best', al['best_offset_s'], 'agree', round(al['best_agree'], 4), 'at 0:', round(al['agree_at_0'], 4))
