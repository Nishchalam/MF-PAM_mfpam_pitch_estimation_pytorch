"""Label construction, decoding and metrics of the RMVPE-RRCGD PTDB pipeline, copied to keep Experiment 2 protocol-identical.
Sources (mel_check_for_audio_chunks/optimised_6_acs_rrcgd_ptdb_10ms_seeded): src/dataset.py PTDB._parse (targets),
src/utils.py to_local_average_cents, evaluate.py evaluate. Parity is tested in tests/test_exp2_parity.py."""
import numpy as np
import torch
from mir_eval.melody import raw_pitch_accuracy, to_cent_voicing, raw_chroma_accuracy, overall_accuracy
from mir_eval.melody import voicing_recall, voicing_false_alarm

CONST = 1997.3794084376191
N_CLASS = 360


def onehot_labels(hz):
    """hz [T] (0 = unvoiced) -> float32 [T,360]; verbatim logic of PTDB._parse."""
    T = len(hz)
    pitch_label = torch.zeros(T, N_CLASS, dtype=torch.float32)
    for i, h in enumerate(np.asarray(hz, dtype=np.float64)):
        if float(h) > 0:
            cent = 1200.0 * np.log2(float(h) / 10.0)
            index = int(round((cent - CONST) / 20.0))
            if 0 <= index < N_CLASS:
                pitch_label[i, index] = 1.0
    return pitch_label


def to_local_average_cents(salience, center=None, thred=0.0):
    """find the weighted average cents near the argmax bin (verbatim from RMVPE-RRCGD src/utils.py)"""
    if not hasattr(to_local_average_cents, 'cents_mapping'):
        to_local_average_cents.cents_mapping = (np.linspace(0, 7180, 360) + 1997.3794084376191)
    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience))
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(salience * to_local_average_cents.cents_mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum if np.max(salience) > thred else 0
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :], None, thred) for i in range(salience.shape[0])])
    raise Exception("label should be either 1d or 2d ndarray")


def cents_to_hz(cents):
    return np.array([10 * (2 ** (c / 1200)) if c else 0 for c in cents])


def file_metrics(pitch_pred, pitch_label, hop_ms=10, pitch_th=0.5):
    """One file: probabilities [T,360], one-hot labels [T,360] (numpy). verbatim logic of evaluate.evaluate loop body."""
    cents_pred = to_local_average_cents(pitch_pred, None, pitch_th)
    cents_label = to_local_average_cents(pitch_label, None, pitch_th)
    freq_pred = cents_to_hz(cents_pred)
    freq = cents_to_hz(cents_label)
    time_slice = np.array([i * hop_ms / 1000 for i in range(len(cents_label))])
    ref_v, ref_c, est_v, est_c = to_cent_voicing(time_slice, freq, time_slice, freq_pred)
    return {"RPA": raw_pitch_accuracy(ref_v, ref_c, est_v, est_c), "RCA": raw_chroma_accuracy(ref_v, ref_c, est_v, est_c),
            "OA": overall_accuracy(ref_v, ref_c, est_v, est_c), "VFA": voicing_false_alarm(ref_v, est_v),
            "VR": voicing_recall(ref_v, est_v)}, freq_pred, freq
