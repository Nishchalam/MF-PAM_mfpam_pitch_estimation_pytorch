"""Smoke test (gate before full training). Uses TRAIN + VALIDATION only."""
import os, sys, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.ptdb_common import *
from src.ptdb_dataset import TrainChunks, EvalSet
from src.evaluation import infer
from src.metrics import summarise, decode, FMIN, BPO
from src.train_ptdb import worker_init, seed_all
from model import Estimation_stage
from dataset import hz_to_onehot

cfg = load_config('configs/mfpam_ptdb.yaml'); dev = torch.device('cuda:0'); seed_all(cfg['experiment']['seed'])
tr, va = list_split(cfg, 'train'), list_split(cfg, 'validation')
assert_disjoint({'train': speakers_of(tr), 'validation': speakers_of(va)}); print('speakers train', speakers_of(tr), 'val', speakers_of(va))
# M-1: official model files unchanged
d = subprocess.check_output('git diff 9303df5 --stat -- model.py module.py augment.py utils.py dataset.py', shell=True, cwd=REPO_ROOT, text=True)
assert d.strip() == '', d; print('official model/augment/dataset files unchanged vs upstream 9303df5')
# M-5: quantiser round trip on hand-computed values (25-cent bins; index 132 for 220 Hz, 0 for 32.7 Hz, 48 for 65.4 Hz)
for hz, idx in [(32.7, 0), (65.4, 48), (220.0, 132), (440.0, 180), (100.0, 77)]:
    o = hz_to_onehot(torch.tensor([hz], dtype=torch.float64)); assert o.shape == (1, 360) and o.sum() == 1
    k = int(o.argmax()); back = FMIN * 2 ** (k / BPO); cents = 1200 * np.log2(back / hz)
    print(f'{hz:7.1f} Hz -> bin {k} (expected {idx}) -> {back:8.3f} Hz ({cents:+.1f} cents)'); assert k == idx and abs(cents) <= 12.6, (k, idx, cents)
assert hz_to_onehot(torch.tensor([0.0], dtype=torch.float64)).sum() == 0; print('0 Hz -> all-zero target OK')

ts = TrainChunks(cfg, tr); print('train chunks per epoch:', len(ts))
dl = DataLoader(ts, batch_size=cfg['training']['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(42), num_workers=cfg['training']['num_workers'], drop_last=True, worker_init_fn=worker_init, pin_memory=True)
model = Estimation_stage().to(dev); n = sum(p.numel() for p in model.parameters()); print('params', n); assert n == 362479
opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999)); crit = nn.BCELoss(); it = iter(dl)
t0 = time.time(); ok_steps = 0
for s in range(6):
    tl = time.time(); f0, q, wav, spk = next(it); tl = time.time() - tl
    assert set(spk) <= set(speakers_of(tr))
    wav, tgt = wav.to(dev), q.to(dev).float()
    if s == 0: print('shapes: f0', tuple(f0.shape), 'quant', tuple(q.shape), 'wav', tuple(wav.shape), 'dtype', q.dtype)
    assert torch.isfinite(wav).all() and torch.isfinite(f0).all() and torch.isfinite(tgt).all()
    ts0 = time.time(); out = model(wav); assert out.shape == tgt.shape == (64, 500, 360), (out.shape, tgt.shape)
    loss = crit(out, tgt); assert torch.isfinite(loss); opt.zero_grad(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')); assert torch.isfinite(gn); opt.step(); torch.cuda.synchronize()
    print(f'step {s} loss {loss.item():.5f} gradnorm {gn.item():.3f} wait-for-data {tl:.2f}s compute {time.time()-ts0:.2f}s  voiced frac {(f0>0).float().mean():.3f}')
# validation on 30 utts + timing extrapolation
sub = Subset(EvalSet(cfg, va), list(range(0, len(va), len(va) // 30))[:30])
t1 = time.time(); utts = infer(model, sub, dev, with_loss=True); vt = time.time() - t1
for u in utts: assert np.isfinite(u['est_hz']).all() and np.isfinite(u['conf']).all() and np.isfinite(u['bce'])
r = summarise(utts, 'dio', 0.5); print('val(30 utts, untrained): bce %.4f RPA50 %.3f OA %.3f' % (np.mean([u['bce'] for u in utts]), r['pooled_50c']['RPA'], r['pooled_50c']['OA']))
print(f'validation time {vt:.1f}s for 30 utts -> ~{vt/30*len(va):.0f}s for {len(va)}')
# frame alignment: DIO frames vs model frames on a full validation file
u0 = EvalSet(cfg, va)[0]; print('val file', u0['stem'], 'audio', tuple(u0['audio'].shape), 'dio frames', len(u0['dio']), 'model frames', model(u0['audio'][None].to(dev)).shape[1], 'rapt frames', len(u0['rapt']))
# checkpoint save / reload
p = os.path.join(cfg['paths']['checkpoints'], '_smoke.pt'); os.makedirs(os.path.dirname(p), exist_ok=True)
torch.save({'model': model.state_dict(), 'optim': opt.state_dict()}, p); m2 = Estimation_stage().to(dev); m2.load_state_dict(torch.load(p, map_location=dev)['model'])
model.eval(); m2.eval(); x = u0['audio'][None].to(dev)
with torch.no_grad(): assert torch.equal(model(x), m2(x))
os.remove(p); print('checkpoint save/reload identical')
print('SMOKE TEST PASSED')
