import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import hz_to_onehot
from src.metrics import decode, summarise

REF_KEYS = ['dio', 'rapt', 'rapt_naive']


@torch.no_grad()
def infer(model, dataset, device, num_workers=4, with_loss=False):
    """Run model on full utterances (batch 1, like official validation). Returns list of per-utterance dicts (numpy)."""
    model.eval()
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    bce = nn.BCELoss()
    utts = []
    for b in dl:
        prob = model(b['audio'].to(device))                       # [1, T', 360]
        n = min(prob.size(1), b['dio'].size(1))                   # official code truncates to the shorter one
        prob = prob[:, :n]
        u = {'stem': b['stem'][0], 'speaker': b['speaker'][0]}
        if with_loss:
            tgt = hz_to_onehot(b['dio'][0, :n]).to(device).float()
            u['bce'] = bce(prob[0], tgt).item()
        p = prob[0].float().cpu().numpy()
        u['est_hz'], u['conf'] = decode(p)
        for k in REF_KEYS:
            u[k] = b[k][0, :n].numpy()
        utts.append(u)
    return utts


def all_refs(utts, thr):
    return {k: summarise(utts, k, thr) for k in REF_KEYS}
