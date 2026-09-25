"""E-2: metrics on synthetic cases (run: python -m pytest tests or python tests/test_metrics.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
from src.metrics import melody, official_style

def test_all():
    ref = np.array([200.] * 50 + [0.] * 50)
    est = np.array([200 * 2 ** (30 / 1200)] * 25 + [200 * 2 ** (80 / 1200)] * 25 + [150.] * 50)
    conf = np.array([0.9] * 50 + [0.9] * 25 + [0.1] * 25)
    m = melody(ref, est, conf, 0.5)
    assert abs(m['RPA'] - 0.5) < 1e-9          # 25/50 within 50 cents
    assert abs(m['VRR'] - 1.0) < 1e-9          # all ref-voiced frames predicted voiced
    assert abs(m['VFA'] - 0.5) < 1e-9          # 25/50 unvoiced frames predicted voiced
    assert abs(m['OA'] - (25 + 25) / 100) < 1e-9   # correct-voiced 25 + correct-unvoiced 25
    # octave error counts for RCA but not RPA
    m2 = melody(np.array([200.] * 10), np.array([400.] * 10), np.full(10, .9))
    assert m2['RPA'] == 0 and m2['RCA'] == 1
    # documented quirk: official-style tolerance is ~170 cents, not 50
    assert official_style(np.array([200.] * 10), np.array([200 * 2 ** (100 / 1200)] * 10))['RPA'] == 1.0
    assert melody(np.array([200.] * 10), np.array([200 * 2 ** (100 / 1200)] * 10), np.full(10, .9))['RPA'] == 0.0
if __name__ == '__main__':
    test_all(); print('metrics tests passed')
