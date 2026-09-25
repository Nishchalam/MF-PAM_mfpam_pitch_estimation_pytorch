"""Final Exp 2 evaluation: best-validation checkpoint on TEST (clean + noisy), RMVPE-RRCGD metrics; frame-level CSVs."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, pandas as pd, torch
from src.ptdb_common import *
from src.exp2_data import Exp2Eval
from src.train_exp2 import infer, score
from src.rmvpe_protocol import file_metrics
from scripts.prepare_12k8 import prepare
from model import Estimation_stage

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb_exp2.yaml'); ap.add_argument('--split', default='test', choices=['test', 'validation'])
    ap.add_argument('--ckpt'); ap.add_argument('--out'); ap.add_argument('--conditions', nargs='+'); a = ap.parse_args(); cfg = load_config(a.config)
    ckpt = a.ckpt or os.path.join(REPO_ROOT, cfg['paths']['checkpoints'], 'best.pt'); out = a.out or os.path.join(REPO_ROOT, cfg['paths']['results']); os.makedirs(out, exist_ok=True)
    conds = a.conditions or cfg['evaluation']['test_conditions']; th = cfg['evaluation']['pitch_th']; prepare(cfg, [a.split], noisy=any(c != 'clean' for c in conds))
    st = torch.load(ckpt, map_location='cuda:0', weights_only=False); model = Estimation_stage().cuda(); model.load_state_dict(st['model']); items = list_split(cfg, a.split)
    res = {'meta': {'split': a.split, 'checkpoint_epoch': st['epoch'], 'checkpoint_iteration': st['iteration'], 'val_at_checkpoint': st['val'], 'pitch_th': th, 'tolerance_cents': 50,
                    'protocol': 'RMVPE-RRCGD evaluate(): per-file mean, predicted-unvoiced=0 Hz, reference=RAPT .npy quantised to 20-cent grid', 'seed': cfg['experiment']['seed']}}
    for c in conds:
        snr = None if c == 'clean' else int(c[3:]); utts = infer(model, Exp2Eval(cfg, items, snr), torch.device('cuda:0')); res[c] = score(utts, th); res[c]['n_files'] = len(utts)
        print(c, ' '.join(f'{k} {100*v:.2f}' for k, v in res[c].items() if k != 'n_files'), flush=True)
        pdir = os.path.join(out, 'predictions' if c == 'clean' else f'predictions_{c}'); os.makedirs(pdir, exist_ok=True)
        for u in utts:
            m, fp, fr = file_metrics(u['prob'], u['label'], 10, th); n = len(fp)
            pd.DataFrame({'frame': np.arange(n), 'time': np.arange(n) * 0.01, 'reference_f0': u['hz'], 'reference_f0_quantised': fr, 'reference_voiced': (u['hz'] > 0).astype(int),
                          'predicted_f0': fp, 'predicted_voiced': (fp > 0).astype(int), 'confidence': u['prob'].max(1)}).to_csv(os.path.join(pdir, u['stem'] + '.csv'), index=False, float_format='%.4f')
    name = 'test_metrics.json' if a.split == 'test' else 'validation_final_metrics.json'; json.dump(res, open(os.path.join(out, name), 'w'), indent=2, default=float); print('wrote', os.path.join(out, name))
