"""Acceptance checks E4-1..E4-3 for MF-PAM-PTDB-005 (128 ms windows). Needs the train/validation cache from prepare_16k.py.
Run: python tests/test_window004.py"""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, soundfile as sf, torch
from src.ptdb_common import load_config, list_split, speakers_of, cache_wav, cache_dio
from src.ptdb_dataset import TrainWindowsNoisy, EvalSet
from src.evaluation import forward_windowed
from dataset import hz_to_onehot
from model import Estimation_stage

cfg = load_config('configs/mfpam_ptdb_005_8k_128ms.yaml')
tr = list_split(cfg, 'train')
ds = TrainWindowsNoisy(cfg, tr)
W, HOP = ds.win, ds.hop
assert (W, ds.nfr) == (2048, 16)
print('train windows/epoch', len(ds), 'utts', len(tr))

# E4-1 / E4-2: shapes, labels == cached clean DIO slice at the window grid, 90/10 mix, shared start for noisy+clean
random.seed(0)
n, noisy = 3000, 0
for i in random.sample(range(len(ds)), n):
    f0, quant, x, spk, used = ds[i]
    assert f0.shape == (16,) and quant.shape == (16, 360) and x.shape == (W,) and spk in speakers_of(tr)
    assert torch.equal(quant.long(), hz_to_onehot(f0).long()) and torch.isfinite(x).all()
    noisy += used
frac = noisy / n
assert abs(frac - 0.9) < 0.03, frac
print('E4-2 noisy fraction', frac)

# find the window start of a few samples by exact match against the clean file, then verify label slice + noisy start
random.seed(1)
for i in random.sample(range(len(ds)), 40):
    k, off = ds.index[i]
    clean_path, noisy_paths, spk, nfile, dio = ds.files[k]
    random.seed(1000 + i); f0, quant, x, spk, used = ds[i]                 # draw start
    st = None
    clean, _ = sf.read(clean_path, dtype='float32')
    for s in range(0, max(nfile - W, 0) + 1, HOP):
        if s < off - 1 or s > off + ds.shift: continue
        if used == 0 and np.array_equal(clean[s:s + W], x.numpy()): st = s; break
        if used == 1:
            for p in noisy_paths:
                nz, _ = sf.read(p, start=s, frames=W, dtype='float32')
                if np.array_equal(nz, x.numpy()): st = s; break
            if st is not None: break
    assert st is not None and st % HOP == 0, 'window not found at a hop-aligned start'
    ref = np.zeros(16, np.float32); seg = dio[st // HOP: st // HOP + 16]; ref[:len(seg)] = seg
    assert np.array_equal(f0.numpy(), ref), 'label != cached clean-utterance DIO slice'
    assert len(np.load(cache_dio(cfg, tr[k]))) == len(dio) == nfile // HOP
print('E4-1 labels == clean DIO slice at the window grid (40/40); shapes OK')

# E4-3: chunked inference == per-chunk outputs, frame count
torch.manual_seed(0)
m = Estimation_stage().eval()
va = list_split(cfg, 'validation')
u = EvalSet(cfg, va[:1])[0]['audio'][None]
T = u.shape[1]
with torch.no_grad():
    p = forward_windowed(m, u, W, max_batch=7)
    exp_n = -(-T // W)
    assert p.shape == (1, exp_n * 16, 360), (p.shape, exp_n)
    x = torch.nn.functional.pad(u[0], (0, (-T) % W))
    for c in (0, exp_n // 2, exp_n - 1):
        single = m(x[c * W:(c + 1) * W][None])
        assert torch.allclose(p[0, c * 16:(c + 1) * 16], single[0], atol=1e-5), c
    assert p.shape[1] >= len(np.load(cache_dio(cfg, va[0])))
print('E4-3 windowed inference OK:', T, 'samples ->', p.shape[1], 'frames; DIO frames', len(np.load(cache_dio(cfg, va[0]))))
print('ALL OK')
