"""Exp 2 dataset check (metadata only): speakers, labels(.npy) vs audio frames at 12.8 kHz / 128-sample hop, chunkability."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, soundfile as sf
from src.ptdb_common import *
cfg = load_config('configs/mfpam_ptdb_exp2.yaml'); out = {}; sr, hop = cfg['dataset']['sample_rate'], cfg['labels']['hop_size']
for sp in ('train', 'validation', 'test'):
    items = list_split(cfg, sp); d = []; short = 0; nz = 0; tot = 0
    for it in items:
        a = np.load(os.path.join(it['dir'], it['stem'] + '.npy')).reshape(-1); assert np.isfinite(a).all()
        frames_audio = int(sf.info(it['wav']).duration * sr) // hop; d.append(frames_audio - len(a)); short += len(a) < cfg['training']['n_steps']; nz += (a > 0).sum(); tot += len(a)
    out[sp] = {'speakers': speakers_of(items), 'n_utts': len(items), 'label_frames': tot, 'voiced_frac': float(nz / tot), 'audio_frames_minus_label_frames': [int(min(d)), float(np.mean(d)), int(max(d))], 'files_shorter_than_256_frames': int(short)}
    print(sp, out[sp]); assert min(d) >= 0 and short == 0
assert_disjoint({k: v['speakers'] for k, v in out.items()})
for k in out: assert out[k]['speakers'] == sorted(cfg['dataset']['expected_speakers'][k])
res = os.path.join(REPO_ROOT, cfg['paths']['results']); os.makedirs(res, exist_ok=True); json.dump(out, open(os.path.join(res, 'dataset_summary.json'), 'w'), indent=2); print('OK, no speaker overlap')
