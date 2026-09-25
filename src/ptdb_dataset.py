"""PTDB adapter for MF-PAM. Reads the EXISTING split directories (never re-splits); official quantiser/augmentation reused."""
import math
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from src.ptdb_common import cache_wav, cache_dio, dio_labels, rapt_on_grid
import augment                      # official, unmodified
from dataset import hz_to_onehot    # official quantiser, unmodified


class TrainChunks(Dataset):
    """Official Audioset/F0Dataset train semantics on clean audio only: 4.5 s crop every 1 s, Shift(8000) -> 4.0 s,
    DIO labels computed on the shifted crop, one-hot (25 cent, 360 bins) target from official hz_to_onehot."""

    def __init__(self, cfg, items):
        sr = cfg['dataset']['sample_rate']
        self.hop = cfg['labels']['hop_size']
        self.sr = sr
        self.length = int(cfg['training']['chunk_length_s'] * sr)
        self.stride = int(cfg['training']['chunk_stride_s'] * sr)
        self.files, self.index = [], []
        for it in items:
            path = cache_wav(cfg, it)
            n = sf.info(path).frames
            k = len(self.files)
            self.files.append((path, it['speaker'], it['stem']))
            if n < self.length:
                ex = 1 if cfg['training']['pad'] else 0
            elif cfg['training']['pad']:
                ex = int(math.ceil((n - self.length) / self.stride) + 1)
            else:
                ex = (n - self.length) // self.stride + 1
            self.index += [(k, self.stride * j) for j in range(ex)]
        self.shift = augment.Shift(cfg['augmentation']['shift'], True)   # same=True, training mode (as official)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, off = self.index[i]
        path, spk, stem = self.files[k]
        x, _ = sf.read(path, start=off, frames=self.length, dtype='float32')
        if len(x) < self.length:
            x = np.pad(x, (0, self.length - len(x)))
        wav = torch.from_numpy(x).view(1, 1, 1, -1)                     # [sources, B, C, T]
        x = self.shift(wav)[0, 0, 0]                                      # [T - shift]
        f0 = torch.from_numpy(dio_labels(x.numpy(), self.hop, self.sr))
        quant = hz_to_onehot(f0).to(torch.uint8)                          # values 0/1, cast on GPU to float
        return f0, quant, x, spk


class EvalSet(Dataset):
    """Full-length utterances (batch 1). Input = clean or noisy 16 kHz audio; references always come from the CLEAN file:
    DIO (cached), RAPT (.f0) with the fitted offset, and RAPT with naive k*10 ms timing."""

    def __init__(self, cfg, items, snr=None):
        self.cfg, self.items, self.snr = cfg, items, snr
        self.rapt_offset = cfg['labels']['rapt_offset_s']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it, cfg = self.items[i], self.cfg
        x, _ = sf.read(cache_wav(cfg, it, self.snr), dtype='float32')
        dio = np.load(cache_dio(cfg, it))
        n, hop, sr, rh = len(dio), cfg['labels']['hop_size'], cfg['dataset']['sample_rate'], cfg['labels']['rapt_hop_s']
        out = {'audio': torch.from_numpy(x), 'dio': torch.from_numpy(dio),
               'rapt_naive': torch.from_numpy(rapt_on_grid(it['f0'], n, hop, sr, rh, 0.0)),
               'stem': it['stem'], 'speaker': it['speaker']}
        out['rapt'] = torch.from_numpy(rapt_on_grid(it['f0'], n, hop, sr, rh, self.rapt_offset if self.rapt_offset is not None else 0.0))
        return out
