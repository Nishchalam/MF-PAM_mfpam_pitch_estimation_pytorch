"""Parity of Exp-2 protocol code with the RMVPE-RRCGD source (A4 labels, A5 decoding, A7 metrics) + A6 shape.
Their functions are exec'd straight from their source files (ast extraction), so this compares against the real code."""
import ast, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy, voicing_recall, voicing_false_alarm
from src.rmvpe_protocol import onehot_labels, to_local_average_cents, file_metrics

R = '/home/batch_2024/ee24s004/KT/RMVPE-RRCGD-spot-computation/mel_check_for_audio_chunks/optimised_6_acs_rrcgd_ptdb_10ms_seeded'
D = '/home/batch_2024/ee24s004/KT/RMVPE-RRCGD-spot-computation/dataset/PTDB_data_10ms_hop'


def their(path, name, ns):
    tree = ast.parse(open(path).read()); fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][0]
    exec(compile(ast.Module([fn], []), path, 'exec'), ns); return ns[name]


def test_decoding_and_metrics():
    ns = {'np': np, 'torch': torch, 'defaultdict': defaultdict, 'raw_pitch_accuracy': raw_pitch_accuracy, 'to_cent_voicing': to_cent_voicing, 'raw_chroma_accuracy': raw_chroma_accuracy,
          'overall_accuracy': overall_accuracy, 'voicing_recall': voicing_recall, 'voicing_false_alarm': voicing_false_alarm, 'PITCH_TH': 0.5, 'bce': lambda a, b: F.binary_cross_entropy_with_logits(a, b)}
    tlac = their(R + '/src/utils.py', 'to_local_average_cents', {'np': np}); ns['to_local_average_cents'] = tlac
    ev = their(R + '/evaluate.py', 'evaluate', ns)
    rng = np.random.default_rng(0); files = sorted(glob.glob(D + '/valid/[FM]*.npy'))[:12]; items, mine = [], []
    for f in files:
        hz = np.load(f).astype(np.float64); lab = onehot_labels(hz)
        assert np.array_equal(to_local_average_cents(lab.numpy(), None, 0.5), tlac(lab.numpy(), None, 0.5))
        prob = lab.numpy().copy(); T = len(hz)
        for t in range(T):                                        # realistic errors: wrong bins, low confidence, octave-like jumps, false alarms
            r = rng.random()
            if prob[t].max() > 0 and r < 0.25: b = int(prob[t].argmax()); prob[t] = 0; prob[t, int(np.clip(b + rng.integers(-40, 40), 0, 359))] = 0.9
            elif prob[t].max() > 0 and r < 0.35: prob[t] *= 0.3
            elif prob[t].max() == 0 and r < 0.1: prob[t, int(rng.integers(0, 360))] = 0.8
        prob = (prob + 0.02 * rng.random(prob.shape)).clip(0, 1).astype(np.float32)
        logits = torch.logit(torch.from_numpy(prob).clamp(1e-6, 1 - 1e-6)); items.append({'roots': torch.zeros(len(hz), 4), 'pitch': lab, 'logits': logits}); mine.append(file_metrics(prob, lab.numpy(), 10, 0.5)[0])

    class M:
        def __init__(s): s.i = 0
        def __call__(s, roots): out = items[s.i]['logits'][None]; s.i += 1; return out
    m = ev(items, M(), 10, 'cpu', 0.5)
    for k in ('RPA', 'RCA', 'OA', 'VR', 'VFA'):
        assert np.allclose(m[k], [x[k] for x in mine], atol=1e-6), (k, m[k][:3], [x[k] for x in mine][:3])
    print('decoding + metric parity OK on', len(files), 'files; mean RPA', np.mean(m['RPA']).round(4))


def test_label_parity_with_cache():
    n_ok = n_cmp = 0
    for f in sorted(glob.glob(D + '/train/_frame_cache_WL32ms_HOP10ms_FREF10.0_NCLASS360/*__chunk0000.pt'))[:60]:
        stem = os.path.basename(f).split('__')[0]; th = torch.load(f, map_location='cpu', weights_only=False)['pitch']
        mine = onehot_labels(np.load(f'{D}/train/{stem}.npy'))[:len(th)]
        n_cmp += 1; n_ok += bool(torch.equal(th, mine))
    print(f'label parity vs RMVPE-RRCGD cache chunk0: {n_ok}/{n_cmp} identical'); assert n_cmp >= 20 and n_ok / n_cmp >= 0.9


def test_model_chunk_shape():
    from model import Estimation_stage
    with torch.no_grad(): assert tuple(Estimation_stage().eval()(torch.randn(2, 256 * 128)).shape) == (2, 256, 360)


if __name__ == '__main__':
    test_decoding_and_metrics(); test_label_parity_with_cache(); test_model_chunk_shape(); print('exp2 parity tests passed')
