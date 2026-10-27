#!/bin/bash
# MF-PAM-PTDB-004: wait until no other process uses the GPU (3 consecutive empty polls, 60 s apart), then train + watchdog.
cd "$(dirname "$0")/.." || exit 1
PY=/home/nishchala/miniconda3/envs/mfpam_env/bin/python
CFG=configs/mfpam_ptdb_004_8k_noisy.yaml
LOG=logs/train_MF-PAM-PTDB-004.log
empty=0
while [ $empty -lt 3 ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then empty=$((empty+1)); else empty=0; fi
  sleep 60
done
echo "$(date) GPU free, launching training" >> $LOG
nohup $PY -W ignore scripts/train_ptdb.py --config $CFG >> $LOG 2>&1 &
TP=$!
sleep 30
nohup $PY -W ignore scripts/watchdog_stop_on_plateau.py --pid $TP --log results/MF-PAM-PTDB-004/training_log.csv --eval-config $CFG >> logs/watchdog_MF-PAM-PTDB-004.log 2>&1 &
echo "$(date) train pid $TP, watchdog pid $!" >> $LOG
