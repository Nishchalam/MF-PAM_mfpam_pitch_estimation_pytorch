#!/bin/bash
# MF-PAM-PTDB-004b (seed 43 replicate of 004): train + plateau watchdog (GPU already free).
cd "$(dirname "$0")/.." || exit 1
PY=/home/nishchala/miniconda3/envs/mfpam_env/bin/python
CFG=configs/mfpam_ptdb_004b_8k_noisy_seed43.yaml
LOG=logs/train_MF-PAM-PTDB-004b.log
nohup $PY -W ignore scripts/train_ptdb.py --config $CFG >> $LOG 2>&1 &
TP=$!
sleep 30
nohup $PY -W ignore scripts/watchdog_stop_on_plateau.py --pid $TP --log results/MF-PAM-PTDB-004b/training_log.csv --eval-config $CFG >> logs/watchdog_MF-PAM-PTDB-004b.log 2>&1 &
echo "$(date) train pid $TP, watchdog pid $!" >> $LOG
