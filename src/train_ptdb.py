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
from src.metrics import summarise
from src.manifest import write_manifest
from model import Estimation_stage       # official, unmodified

LOG_FIELDS = ['epoch', 'train_loss', 'val_loss', 'learning_rate', 'elapsed_time_s', 'val_RPA_50c', 'val_RCA_50c', 'val_VRR', 'val_VFA',
              'val_OA', 'val_RPA_official_style', 'val_RCA_official_style', 'val_RPA_rapt_50c', 'val_OA_rapt', 'grad_nonfinite', 'steps', 'best']


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def validate(model, val_set, device, thr):
    utts = infer(model, val_set, device, num_workers=4, with_loss=True)
    d, r = summarise(utts, 'dio', thr), summarise(utts, 'rapt', thr)
    return {'val_loss': float(np.mean([u['bce'] for u in utts])),
            'val_RPA_50c': d['pooled_50c']['RPA'], 'val_RCA_50c': d['pooled_50c']['RCA'], 'val_VRR': d['pooled_50c']['VRR'],
            'val_VFA': d['pooled_50c']['VFA'], 'val_OA': d['pooled_50c']['OA'],
            'val_RPA_official_style': d['perfile_official_style']['RPA'], 'val_RCA_official_style': d['perfile_official_style']['RCA'],
            'val_RPA_rapt_50c': r['pooled_50c']['RPA'], 'val_OA_rapt': r['pooled_50c']['OA']}


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
    with open(os.path.join(res_dir, 'validation_metrics.csv'), 'w', newline='') as f:   # validation columns of the log
        src = list(csv.DictReader(open(log_path)))
        ww = csv.DictWriter(f, ['epoch'] + [k for k in LOG_FIELDS if k.startswith('val_')] + ['best']); ww.writeheader()
        for r in src: ww.writerow({k: r[k] for k in ww.fieldnames})
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb.yaml'); ap.add_argument('--max_epochs', type=int)
    a = ap.parse_args(); print(main(a.config, a.max_epochs))
