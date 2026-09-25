"""MF-PAM training on the PTDB speaker-disjoint split. TRAIN speakers -> updates; VALIDATION speakers -> checkpoint
selection. Test files are never listed or opened here."""
import argparse, csv, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.ptdb_common import load_config, list_split, speakers_of, assert_disjoint, REPO_ROOT
from src.ptdb_dataset import TrainChunks, EvalSet
from src.evaluation import infer
from src.metrics import lean_summary
from scripts.auto_commit import commit_and_push
from src.manifest import write_manifest
from model import Estimation_stage       # official, unmodified

_M = ['RPA', 'RCA', 'VRR', 'VFA', 'OA']
def _cols(prefix):
    return [f'{prefix}_{proto}_{ref}_{k}' for ref in ('dio', 'rapt') for proto in ('paper', 'rmvpe') for k in _M]
LOG_FIELDS = (['epoch', 'train_loss', 'val_loss', 'learning_rate', 'elapsed_time_s', 'val_RPA_50c'] + _cols('val')
              + ['grad_nonfinite', 'steps', 'best'])
MON_FIELDS = ['epoch', 'which', 'ckpt_epoch', 'test_RPA_50c_pooled_dio'] + _cols('test')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def flat(prefix, ref, m):
    return {f'{prefix}_{proto}_{ref}_{k}': m[f'{proto}_{k}'] for proto in ('paper', 'rmvpe') for k in _M}


def validate(model, val_set, device, thr):
    utts = infer(model, val_set, device, num_workers=4, with_loss=True)
    d, r = lean_summary(utts, 'dio', thr, pooled=True), lean_summary(utts, 'rapt', thr)
    return {'val_loss': float(np.mean([u['bce'] for u in utts])), 'val_RPA_50c': d['pooled_RPA_50c'], **flat('val', 'dio', d), **flat('val', 'rapt', r)}


def monitor_test(model, test_set, device, thr, epoch, which, ckpt_epoch, res_dir):
    """MONITORING ONLY (deviation D16, requested by the user): test metrics of the best / last checkpoint every N epochs.
    Never used for selection, stopping or tuning."""
    utts = infer(model, test_set, device, num_workers=4)
    d, r = lean_summary(utts, 'dio', thr, pooled=True), lean_summary(utts, 'rapt', thr)
    row = {'epoch': epoch, 'which': which, 'ckpt_epoch': ckpt_epoch, 'test_RPA_50c_pooled_dio': d['pooled_RPA_50c'], **flat('test', 'dio', d), **flat('test', 'rapt', r)}
    path = os.path.join(res_dir, 'test_monitor.csv'); new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, MON_FIELDS)
        if new: w.writeheader()
        w.writerow(row)
    os.makedirs(os.path.join(res_dir, 'monitor'), exist_ok=True)
    json.dump({'epoch': epoch, 'which': which, 'checkpoint_epoch': ckpt_epoch, 'voicing_threshold': thr, 'pitch_tolerance_cents': 50,
               'reference_dio': d, 'reference_rapt': r}, open(os.path.join(res_dir, 'monitor', f'test_ep{epoch:04d}_{which}.json'), 'w'), indent=1)
    return row


def main(cfg_path, max_epochs=None, dataset_summary=None):
    cfg = load_config(cfg_path)
    T, seed = cfg['training'], cfg['experiment']['seed']
    res_dir = os.path.join(REPO_ROOT, cfg['paths']['results']); ck_dir = os.path.join(REPO_ROOT, cfg['paths']['checkpoints'])
    os.makedirs(ck_dir, exist_ok=True)
    seed_all(seed)
    torch.backends.cudnn.benchmark = True                    # as official
    device = torch.device('cuda:0')

    tr, va = list_split(cfg, 'train'), list_split(cfg, 'validation')
    exp = cfg['dataset']['expected_speakers']
    assert speakers_of(tr) == sorted(exp['train']) and speakers_of(va) == sorted(exp['validation'])
    assert_disjoint({'train': speakers_of(tr), 'validation': speakers_of(va)})
    train_spk = set(speakers_of(tr))
    write_manifest(cfg, cfg_path, res_dir, REPO_ROOT, dataset_summary)

    model = Estimation_stage().to(device)
    assert sum(p.numel() for p in model.parameters()) == 362479
    opt = torch.optim.Adam(model.parameters(), lr=T['learning_rate'], betas=(T['adam_b1'], T['adam_b2']))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=T['scheduler']['gamma'])
    train_set, val_set = TrainChunks(cfg, tr), EvalSet(cfg, va)
    crit = nn.BCELoss()
    mon_every = cfg.get('monitoring', {}).get('test_every_epochs', 0)
    test_set = EvalSet(cfg, list_split(cfg, 'test')) if mon_every else None
    if mon_every: best_model = Estimation_stage().to(device)
    thr = cfg['evaluation']['voicing_threshold']

    epoch0, steps, best = 0, 0, {'val_RPA_50c': -1.0, 'val_loss': 1e9, 'epoch': -1}
    last = os.path.join(ck_dir, 'last.pt')
    if os.path.exists(last):
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['optim']); sched.load_state_dict(st['sched'])
        epoch0, steps, best = st['epoch'], st['steps'], st['best']
        print(f'resumed from epoch {epoch0}')
    log_path = os.path.join(res_dir, 'training_log.csv')
    new_log = not os.path.exists(log_path) or epoch0 == 0
    logf = open(log_path, 'w' if new_log else 'a', newline=''); w = csv.DictWriter(logf, LOG_FIELDS)
    if new_log: w.writeheader()
    print(f'train chunks/epoch {len(train_set)}  train utts {len(tr)} val utts {len(va)}', flush=True)

    total = max_epochs or T['epochs']
    for epoch in range(epoch0 + 1, total + 1):
        t0 = time.time(); model.train()
        g = torch.Generator(); g.manual_seed(seed + epoch)
        dl = DataLoader(train_set, batch_size=T['batch_size'], shuffle=T['shuffle'], generator=g, num_workers=T['num_workers'],
                        pin_memory=True, drop_last=True, worker_init_fn=worker_init)
        loss_sum, n, bad = 0.0, 0, 0
        for f0, quant, wav, spk in dl:
            assert set(spk) <= train_spk, f'non-train speaker in training batch: {set(spk) - train_spk}'
            wav, tgt = wav.to(device, non_blocking=True), quant.to(device, non_blocking=True).float()
            assert torch.isfinite(wav).all() and torch.isfinite(tgt).all(), 'non-finite input/target'
            out = model(wav)
            assert out.shape == tgt.shape, (out.shape, tgt.shape)
            loss = crit(out, tgt)
            assert torch.isfinite(loss), f'non-finite loss at step {steps}'
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))     # measures only, no clipping
            if torch.isfinite(gn): opt.step()
            else: bad += 1
            loss_sum += loss.item(); n += 1; steps += 1
            if steps % 50 == 0: print(f'ep {epoch} step {steps} loss {loss.item():.5f}', flush=True)
        lr = opt.param_groups[0]['lr']; sched.step()
        m = validate(model, val_set, device, cfg['evaluation']['voicing_threshold'])
        better = (m['val_RPA_50c'] > best['val_RPA_50c']) or (m['val_RPA_50c'] == best['val_RPA_50c'] and m['val_loss'] < best['val_loss'])
        if better:
            best = {'val_RPA_50c': float(m['val_RPA_50c']), 'val_loss': float(m['val_loss']), 'epoch': epoch}
        row = {'epoch': epoch, 'train_loss': loss_sum / max(n, 1), 'learning_rate': lr, 'elapsed_time_s': round(time.time() - t0, 1),
               'grad_nonfinite': bad, 'steps': steps, 'best': int(better), **m}
        w.writerow({k: row[k] for k in LOG_FIELDS}); logf.flush()
        state = {'model': model.state_dict(), 'optim': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch, 'steps': steps, 'best': best, 'cfg': cfg}
        if better: torch.save(state, os.path.join(ck_dir, 'best.pt'))
        torch.save(state, last)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if mon_every and epoch % mon_every == 0:
            t1 = time.time()
            rl = monitor_test(model, test_set, device, thr, epoch, 'last', epoch, res_dir)
            best_model.load_state_dict(torch.load(os.path.join(ck_dir, 'best.pt'), map_location=device, weights_only=False)['model'])
            rb = monitor_test(best_model, test_set, device, thr, epoch, 'best', best['epoch'], res_dir)
            msg = (f"auto: test inference at epoch {epoch} (monitoring only) | best@{best['epoch']} DIO paper RPA {100*rb['test_paper_dio_RPA']:.2f} "
                   f"RMVPE-RAPT RPA {100*rb['test_rmvpe_rapt_RPA']:.2f} | last DIO paper RPA {100*rl['test_paper_dio_RPA']:.2f} "
                   f"RMVPE-RAPT RPA {100*rl['test_rmvpe_rapt_RPA']:.2f} | val RPA50 {100*m['val_RPA_50c']:.2f}")
            print(msg, f'[{time.time()-t1:.0f}s]', commit_and_push(msg), flush=True)
    with open(os.path.join(res_dir, 'validation_metrics.csv'), 'w', newline='') as f:   # validation columns of the log
        src = list(csv.DictReader(open(log_path)))
        ww = csv.DictWriter(f, ['epoch'] + [k for k in LOG_FIELDS if k.startswith('val_')] + ['best']); ww.writeheader()
        for r in src: ww.writerow({k: r[k] for k in ww.fieldnames})
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb.yaml'); ap.add_argument('--max_epochs', type=int)
    a = ap.parse_args(); print(main(a.config, a.max_epochs))
