"""Shared helpers for the MF-PAM-PTDB-001 experiment (split listing, cache paths, label grids)."""
import os
import re
import sys
import yaml
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:      # official modules (model, module, dataset, augment) live in the repo root
    sys.path.insert(0, REPO_ROOT)

CLEAN_RE = re.compile(r'^([FM]\d{2})_(.+)\.wav$')   # clean utterance; noisy copies start with "snrNN_"
SPLIT_KEYS = {'train': 'train_dir', 'validation': 'validation_dir', 'test': 'test_dir'}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def list_split(cfg, split):
    """Sorted list of clean utterances of a split, read from the existing directories (never re-split)."""
    d = os.path.join(cfg['dataset']['root'], cfg['dataset'][SPLIT_KEYS[split]])
    items = []
    for fn in sorted(os.listdir(d)):
        m = CLEAN_RE.match(fn)
        if m:
            stem = fn[:-4]
            items.append({'split': split, 'stem': stem, 'speaker': m.group(1),
                          'wav': os.path.join(d, fn), 'f0': os.path.join(d, stem + '.f0'),
                          'dir': d})
    return items


def speakers_of(items):
    return sorted({i['speaker'] for i in items})


def assert_disjoint(sets):
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = set(sets[names[i]]) & set(sets[names[j]])
            assert not inter, f'speaker overlap {names[i]} & {names[j]}: {sorted(inter)}'


def snr_tag(snr):
    return f'snr{int(snr):02d}'


def noisy_wav(item, snr):
    return os.path.join(item['dir'], f"{snr_tag(snr)}_{item['stem']}.wav")


def cache_wav(cfg, item, snr=None):
    name = item['stem'] if snr is None else f"{snr_tag(snr)}_{item['stem']}"
    return os.path.join(cfg['dataset']['cache_root'], '16k', item['split'], name + '.wav')


def cache_dio(cfg, item):
    return os.path.join(cfg['dataset']['cache_root'], 'labels_dio', item['split'], item['stem'] + '.npy')


def dio_labels(x16k, hop=128, sr=16000):
    """Official label recipe (dataset.py F0Dataset.__getitem__): pyworld DIO, frame_period = hop/sr*1000 ms, drop last frame."""
    import pyworld as pw
    f0, _ = pw.dio(np.asarray(x16k, dtype=np.float64), sr, frame_period=hop / sr * 1000)
    return f0[:-1].astype(np.float32)


def rapt_on_grid(f0_path, n_frames, hop=128, sr=16000, rapt_hop=0.010, offset_s=0.0):
    """RAPT (.f0 column 0) sampled onto the model's frame grid (frame j at j*hop/sr s).
    RAPT frame i is assumed to sit at time i*rapt_hop + offset_s; nearest RAPT frame is used (no interpolation
    across voiced/unvoiced boundaries). Frames without a RAPT frame are unvoiced (0)."""
    r = np.loadtxt(f0_path, ndmin=2)[:, 0]
    t = np.arange(n_frames) * hop / sr
    idx = np.rint((t - offset_s) / rapt_hop).astype(int)
    ok = (idx >= 0) & (idx < len(r))
    out = np.zeros(n_frames, dtype=np.float32)
    out[ok] = r[idx[ok]]
    return out
