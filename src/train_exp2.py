"""MF-PAM-PTDB-002: MF-PAM trained with the RMVPE-RRCGD PTDB protocol (specs/05). TRAIN -> updates, VALIDATION -> selection.
No test data is read here."""
import argparse, csv, os, random, time
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from src.ptdb_common import load_config, list_split, speakers_of, assert_disjoint, REPO_ROOT
from src.exp2_data import Exp2Train, Exp2Eval
from src.rmvpe_protocol import file_metrics
from src.manifest import write_manifest
from model import Estimation_stage       # official, unmodified

FIELDS = ['epoch', 'iteration', 'train_loss', 'val_loss', 'learning_rate', 'elapsed_time_s', 'val_RPA', 'val_RCA', 'val_OA', 'val_VR', 'val_VFA', 'skipped_steps', 'best']


def set_seed(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic: torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def seed_worker(_):
    s = torch.initial_seed() % 2 ** 32; np.random.seed(s); random.seed(s)


@torch.no_grad()
def infer(model, ds, device, pitch_th=0.5, workers=2):
    """-> list of dict(stem, prob[T,360] numpy, label[T,360] numpy, hz[T]) with prob truncated to label length (as RMVPE-RRCGD)."""
    model.eval(); out = []
    for b in DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers):
        p = model(b['audio'].to(device))[0]; T = b['label'].shape[1]; assert p.shape[0] >= T, (p.shape, T)
        out.append({'stem': b['stem'][0], 'prob': p[:T].float().cpu().numpy(), 'label': b['label'][0].numpy(), 'hz': b['hz'][0].numpy()})
    return out


def score(utts, pitch_th=0.5, hop_ms=10):
    M = [file_metrics(u['prob'], u['label'], hop_ms, pitch_th)[0] for u in utts]
    return {k: float(np.mean([m[k] for m in M])) for k in ('RPA', 'RCA', 'OA', 'VR', 'VFA')}


def main(cfg_path, max_iters=None):
    cfg = load_config(cfg_path); T = cfg['training']; seed = cfg['experiment']['seed']
    res = os.path.join(REPO_ROOT, cfg['paths']['results']); ck = os.path.join(REPO_ROOT, cfg['paths']['checkpoints']); os.makedirs(ck, exist_ok=True)
    set_seed(seed, cfg['experiment']['deterministic']); dev = torch.device('cuda:0')
    tr, va = list_split(cfg, 'train'), list_split(cfg, 'validation'); exp = cfg['dataset']['expected_speakers']
    assert speakers_of(tr) == sorted(exp['train']) and speakers_of(va) == sorted(exp['validation']); assert_disjoint({'t': speakers_of(tr), 'v': speakers_of(va)})
    write_manifest(cfg, cfg_path, res, REPO_ROOT)
    model = Estimation_stage().to(dev); assert sum(p.numel() for p in model.parameters()) == 362479
    opt = torch.optim.Adam(model.parameters(), T['learning_rate'])
    train_ds, val_ds = Exp2Train(cfg, tr), Exp2Eval(cfg, va)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(train_ds, T['batch_size'], shuffle=True, drop_last=True, num_workers=T['num_workers'], pin_memory=True, persistent_workers=True, worker_init_fn=seed_worker, generator=g)
    L = len(dl); iterations = L * T['epochs'] if not max_iters else max_iters
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=L * T['scheduler']['step_size_epochs'], gamma=T['scheduler']['gamma'])
    crit = nn.BCELoss(); best_rpa, best_it, skipped = 0.0, 0, 0
    logf = open(os.path.join(res, 'training_log.csv'), 'w', newline=''); w = csv.DictWriter(logf, FIELDS); w.writeheader()
    print(f'train chunks {len(train_ds)}  iters/epoch {L}  total iters {iterations}  val utts {len(va)}', flush=True)
    t0, run, n_run, i = time.time(), 0.0, 0, 0
    while i < iterations:
        for f0wav, lab, spk in dl:
            i += 1; model.train(); assert set(spk) <= set(speakers_of(tr))
            x, y = f0wav.to(dev, non_blocking=True), lab.to(dev, non_blocking=True)
            out = model(x)[:, :y.shape[1], :]; assert out.shape == y.shape, (out.shape, y.shape)
            loss = crit(out, y)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); skipped += 1; print(f'[warn] non-finite loss at iter {i}, skipping', flush=True)
            else:
                opt.zero_grad(set_to_none=True); loss.backward()
                if T['gradient_clip']: torch.nn.utils.clip_grad_norm_(model.parameters(), T['gradient_clip'])
                opt.step(); sched.step(); run += loss.item(); n_run += 1
            if i % 200 == 0: print(f'iter {i}/{iterations} loss {loss.item():.5f}', flush=True)
            if i % L == 0 or i == iterations:
                utts = infer(model, val_ds, dev); m = score(utts, cfg['evaluation']['pitch_th'])
                vl = float(np.mean([crit(torch.from_numpy(u['prob']), torch.from_numpy(u['label'])).item() for u in utts]))
                better = m['RPA'] >= best_rpa
                if better:
                    best_rpa, best_it = m['RPA'], i
                    torch.save({'model': model.state_dict(), 'iteration': i, 'epoch': i // L, 'val': m, 'cfg': cfg}, os.path.join(ck, 'best.pt'))
                torch.save({'model': model.state_dict(), 'iteration': i, 'epoch': i // L, 'val': m}, os.path.join(ck, 'last.pt'))
                row = {'epoch': i / L, 'iteration': i, 'train_loss': run / max(n_run, 1), 'val_loss': vl, 'learning_rate': opt.param_groups[0]['lr'], 'elapsed_time_s': round(time.time() - t0, 1),
                       'val_RPA': m['RPA'], 'val_RCA': m['RCA'], 'val_OA': m['OA'], 'val_VR': m['VR'], 'val_VFA': m['VFA'], 'skipped_steps': skipped, 'best': int(better)}
                w.writerow(row); logf.flush(); run, n_run = 0.0, 0
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            if i >= iterations or i - best_it > L * T['patience_epochs']:
                if i < iterations: print(f'Early stopping (patience={T["patience_epochs"]} epochs) at iter {i}', flush=True)
                iterations = i; break
    print(f'Finished: best val RPA {best_rpa:.4f} @ iter {best_it} (epoch {best_it / L:.1f})'); return best_rpa, best_it


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb_exp2.yaml'); ap.add_argument('--max_iters', type=int); a = ap.parse_args(); main(a.config, a.max_iters)
