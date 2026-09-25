"""Final evaluation with the best-validation checkpoint. Default split = TEST (run once, after training).
Reports DIO / RAPT(fitted offset) / RAPT(naive) references, pooled + per-file, official-style + 50-cent metrics,
clean and noisy (SNR 20/10/5/0) inputs; exports frame-level CSVs."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd, torch
from src.ptdb_common import *
from src.ptdb_dataset import EvalSet
from src.evaluation import infer, all_refs
from model import Estimation_stage
from scripts.prepare_16k import prepare

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mfpam_ptdb.yaml'); ap.add_argument('--split', default='test', choices=['test', 'validation'])
    ap.add_argument('--ckpt'); ap.add_argument('--out'); ap.add_argument('--conditions', nargs='+')
    a = ap.parse_args(); cfg = load_config(a.config)
    ckpt = a.ckpt or os.path.join(REPO_ROOT, cfg['paths']['checkpoints'], 'best.pt')
    out = a.out or os.path.join(REPO_ROOT, cfg['paths']['results']); os.makedirs(out, exist_ok=True)
    conds = a.conditions or cfg['evaluation']['test_conditions']; thr = cfg['evaluation']['voicing_threshold']
    prepare(cfg, [a.split], noisy=any(c != 'clean' for c in conds))
    st = torch.load(ckpt, map_location='cuda:0', weights_only=False); model = Estimation_stage().cuda(); model.load_state_dict(st['model'])
    items = list_split(cfg, a.split); print(f'evaluating {len(items)} utts, split={a.split}, ckpt epoch {st["epoch"]} (best {st["best"]})')
    res = {'meta': {'split': a.split, 'checkpoint': ckpt, 'checkpoint_epoch': st['epoch'], 'best_val': st['best'], 'voicing_threshold': thr,
                    'pitch_tolerance_cents': 50, 'rapt_offset_s': cfg['labels']['rapt_offset_s'], 'seed': cfg['experiment']['seed']}}
    for c in conds:
        snr = None if c == 'clean' else int(c[3:])
        utts = infer(model, EvalSet(cfg, items, snr), torch.device('cuda:0'), num_workers=4)
        res[c] = all_refs(utts, thr)
        d = res[c]['dio']['pooled_50c']; r = res[c]['rapt']['pooled_50c']
        print(f'{c:6s} DIO  RPA {100*d["RPA"]:.2f} RCA {100*d["RCA"]:.2f} VRR {100*d["VRR"]:.2f} VFA {100*d["VFA"]:.2f} OA {100*d["OA"]:.2f} | '
              f'RAPT RPA {100*r["RPA"]:.2f} RCA {100*r["RCA"]:.2f} VRR {100*r["VRR"]:.2f} VFA {100*r["VFA"]:.2f} OA {100*r["OA"]:.2f}', flush=True)
        pdir = os.path.join(out, 'predictions' if c == 'clean' else f'predictions_{c}'); os.makedirs(pdir, exist_ok=True)
        for u in utts:
            n = len(u['est_hz'])
            pd.DataFrame({'frame': np.arange(n), 'time': np.arange(n) * cfg['labels']['hop_size'] / cfg['dataset']['sample_rate'],
                          'reference_f0_dio': u['dio'], 'reference_voiced_dio': (u['dio'] > 0).astype(int),
                          'reference_f0_rapt': u['rapt'], 'reference_voiced_rapt': (u['rapt'] > 0).astype(int),
                          'reference_f0_rapt_naive': u['rapt_naive'], 'reference_voiced_rapt_naive': (u['rapt_naive'] > 0).astype(int),
                          'predicted_f0': np.where(u['conf'] > thr, u['est_hz'], 0.0), 'predicted_voiced': (u['conf'] > thr).astype(int),
                          'predicted_f0_raw': u['est_hz'], 'confidence': u['conf']}).to_csv(os.path.join(pdir, u['stem'] + '.csv'), index=False, float_format='%.4f')
    name = 'test_metrics.json' if a.split == 'test' else 'validation_final_metrics.json'
    json.dump(res, open(os.path.join(out, name), 'w'), indent=2, default=float); print('wrote', os.path.join(out, name))
