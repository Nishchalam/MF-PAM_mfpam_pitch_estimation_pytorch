"""Resample clean (and optionally noisy) PTDB audio 48k->16k with soxr and cache DIO labels.
Only the requested splits are touched; training/validation prep never opens test files."""
import argparse, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, soundfile as sf, soxr
from src.ptdb_common import load_config, list_split, cache_wav, cache_dio, noisy_wav, dio_labels


def _one(args):
    cfg, item, snr, want_dio = args
    src = item['wav'] if snr is None else noisy_wav(item, snr)
    dst = cache_wav(cfg, item, snr)
    if not os.path.exists(dst):
        x, sr = sf.read(src, dtype='float64')
        assert sr == cfg['dataset']['native_sample_rate'] and x.ndim == 1, (src, sr, x.shape)
        y = soxr.resample(x, sr, cfg['dataset']['sample_rate'], quality=cfg['dataset']['resampler']['quality'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst, y.astype(np.float32), cfg['dataset']['sample_rate'], subtype='FLOAT')
    if want_dio:
        d = cache_dio(cfg, item)
        if not os.path.exists(d):
            y, _ = sf.read(dst, dtype='float32')
            os.makedirs(os.path.dirname(d), exist_ok=True)
            np.save(d, dio_labels(y, cfg['labels']['hop_size'], cfg['dataset']['sample_rate']))
    return item['stem']


def prepare(cfg, splits, noisy=False, workers=20):
    jobs = []
    for sp in splits:
        for it in list_split(cfg, sp):
            jobs.append((cfg, it, None, sp != 'train'))          # train DIO is computed on the fly on shifted crops
            if noisy:
                jobs += [(cfg, it, s, False) for s in cfg['dataset']['noisy_snr_db']]
    with Pool(workers) as p:
        n = len(list(p.imap_unordered(_one, jobs, chunksize=16)))
    print(f'prepared {n} files for splits {splits} (noisy={noisy})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mfpam_ptdb.yaml')
    ap.add_argument('--splits', nargs='+', default=['train', 'validation'])
    ap.add_argument('--noisy', action='store_true')
    a = ap.parse_args()
    prepare(load_config(a.config), a.splits, a.noisy)
