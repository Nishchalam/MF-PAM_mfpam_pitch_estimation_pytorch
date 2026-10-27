import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dataset import hz_to_onehot
from src.metrics import decode, summarise

REF_KEYS = ['dio', 'rapt', 'rapt_naive']


def forward_windowed(model, audio, window, hop=128, max_batch=256):
    """specs/07: run the model on consecutive NON-overlapping `window`-sample chunks (zero-padded tail), batch the chunks,
    and concatenate the frames. audio [1, T] -> [1, ceil(T/window)*window/hop, 360]."""
    x = audio[0]
    x = F.pad(x, (0, (-x.numel()) % window)).view(-1, window)
    p = torch.cat([model(x[i:i + max_batch]) for i in range(0, len(x), max_batch)])      # [N, window/hop, 360]
    assert p.size(1) == window // hop, p.shape
    return p.reshape(1, -1, p.size(-1))


@torch.no_grad()
def infer(model, dataset, device, num_workers=4, with_loss=False, window=None):
    """Run model on full utterances (batch 1, like official validation). Returns list of per-utterance dicts (numpy).
    window (samples): None = whole utterance in one pass (001/003); else chunked inference, see forward_windowed."""
    model.eval()
    dl = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    bce = nn.BCELoss()
    utts = []
    for b in dl:
        audio = b['audio'].to(device)
        prob = model(audio) if window is None else forward_windowed(model, audio, window)   # [1, T', 360]
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
