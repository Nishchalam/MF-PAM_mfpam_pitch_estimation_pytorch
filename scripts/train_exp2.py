import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
from src.train_exp2 import main
ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb_exp2.yaml'); ap.add_argument('--max_iters', type=int); a = ap.parse_args(); main(a.config, a.max_iters)
