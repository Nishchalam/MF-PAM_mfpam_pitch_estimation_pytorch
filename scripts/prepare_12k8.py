"""Resample PTDB audio 48k -> 12.8 kHz (soxr VHQ) into <cache_root>/12k8/<split>/. Only requested splits are touched."""
import argparse, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, soundfile as sf, soxr
from src.ptdb_common import load_config, list_split, noisy_wav
from src.exp2_data import wav12


def _one(a):
    cfg, it, snr = a
    src = it['wav'] if snr is None else noisy_wav(it, snr); dst = wav12(cfg, it, snr)
    if not os.path.exists(dst):
        x, sr = sf.read(src, dtype='float64'); assert sr == cfg['dataset']['native_sample_rate'] and x.ndim == 1
        y = soxr.resample(x, sr, cfg['dataset']['sample_rate'], quality=cfg['dataset']['resampler']['quality'])
        assert np.isfinite(y).all(); os.makedirs(os.path.dirname(dst), exist_ok=True); sf.write(dst, y.astype(np.float32), cfg['dataset']['sample_rate'], subtype='FLOAT')
    return 1


def prepare(cfg, splits, noisy=False, workers=20):
    jobs = [(cfg, it, None) for sp in splits for it in list_split(cfg, sp)]
    if noisy: jobs += [(cfg, it, s) for sp in splits for it in list_split(cfg, sp) for s in cfg['dataset']['noisy_snr_db']]
    with Pool(workers) as p: n = sum(p.imap_unordered(_one, jobs, chunksize=16))
    print(f'prepared {n} files at 12.8 kHz for {splits} (noisy={noisy})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb_exp2.yaml')
    ap.add_argument('--splits', nargs='+', default=['train', 'validation']); ap.add_argument('--noisy', action='store_true'); a = ap.parse_args()
    prepare(load_config(a.config), a.splits, a.noisy)
