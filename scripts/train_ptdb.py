"""python scripts/train_ptdb.py --config configs/mfpam_ptdb.yaml   (entry point; official train.py is left untouched)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
from src.train_ptdb import main
ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/mfpam_ptdb.yaml'); ap.add_argument('--max_epochs', type=int)
a = ap.parse_args()
print(main(a.config, a.max_epochs))
