"""Experiment 2 data: 12.8 kHz audio + RMVPE-RRCGD `.npy` labels (frame j <-> samples [j*128, ...))."""
import os, sys, math
import numpy as np, soundfile as sf, torch
from torch.utils.data import Dataset
from src.ptdb_common import list_split, snr_tag
from src.rmvpe_protocol import onehot_labels


def wav12(cfg, item, snr=None):
    name = item['stem'] if snr is None else f"{snr_tag(snr)}_{item['stem']}"
    return os.path.join(cfg['dataset']['cache_root'], '12k8', item['split'], name + '.wav')


def npy_path(item):
    return os.path.join(item['dir'], item['stem'] + '.npy')


class Exp2Train(Dataset):
    """Chunking of RMVPE-RRCGD PTDB: n_steps=256 frames; non-overlapping chunks + one remainder chunk = last 256 frames."""
    def __init__(self, cfg, items):
        self.n, self.hop = cfg['training']['n_steps'], cfg['labels']['hop_size']
        self.files, self.index = [], []
        for it in items:
            lab = np.load(npy_path(it)).astype(np.float64).reshape(-1); T = len(lab)
            assert T >= self.n, (it['stem'], T)
            k = len(self.files); self.files.append((wav12(cfg, it), lab, it['speaker']))
            for i in range(T // self.n): self.index.append((k, i * self.n))
            if T % self.n: self.index.append((k, T - self.n))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        k, b = self.index[i]; path, lab, spk = self.files[k]
        x, _ = sf.read(path, start=b * self.hop, frames=self.n * self.hop, dtype='float32')
        assert len(x) == self.n * self.hop, (path, b, len(x))
        return torch.from_numpy(x), onehot_labels(lab[b:b + self.n]), spk


class Exp2Eval(Dataset):
    """Whole utterances; audio clean or noisy (snr) at 12.8 kHz, labels always from the clean `.npy`."""
    def __init__(self, cfg, items, snr=None):
        self.cfg, self.items, self.snr = cfg, items, snr

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]; x, _ = sf.read(wav12(self.cfg, it, self.snr), dtype='float32')
        hz = np.load(npy_path(it)).astype(np.float64).reshape(-1)
        return {'audio': torch.from_numpy(x), 'label': onehot_labels(hz), 'hz': torch.from_numpy(hz), 'stem': it['stem'], 'speaker': it['speaker']}
