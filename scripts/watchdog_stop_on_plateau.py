"""Watch a training_log.csv, stop the training process once validation RPA plateaus (same criterion used
manually for MF-PAM-PTDB-001: consecutive 100-epoch blocks each gaining <=0.10 percentage points on val_RPA_50c,
only evaluated once >=300 epochs are logged), then run the final test evaluation once. One-shot: exits when done.

Usage: python scripts/watchdog_stop_on_plateau.py --pid <training_pid> --log results/MF-PAM-PTDB-003/training_log.csv \
       --eval-config configs/mfpam_ptdb_003_noisy.yaml [--min-epochs 300] [--block 100] [--delta 0.001] \
       [--stall-minutes 180] [--poll-seconds 60]
"""
import argparse, csv, os, signal, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(os.path.dirname(sys.executable), 'python')


def read_log(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(int(r['epoch']), float(r['val_RPA_50c'])) for r in csv.DictReader(f) if r.get('val_RPA_50c')]


def block_max(rows, lo, hi):
    vals = [v for e, v in rows if lo < e <= hi]
    return max(vals) if vals else None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--eval-config', required=True)
    ap.add_argument('--min-epochs', type=int, default=300)
    ap.add_argument('--block', type=int, default=100)
    ap.add_argument('--delta', type=float, default=0.0010)   # 0.10 percentage points
    ap.add_argument('--consecutive', type=int, default=2)
    ap.add_argument('--stall-minutes', type=int, default=180)
    ap.add_argument('--poll-seconds', type=int, default=60)
    a = ap.parse_args()

    print(f'[watchdog] watching pid {a.pid}, log {a.log}, block={a.block} delta<={a.delta} '
          f'consecutive={a.consecutive}, min_epochs={a.min_epochs}, stall={a.stall_minutes}min', flush=True)

    last_len, last_growth_t = 0, time.time()
    printed_blocks = set()   # only for avoiding duplicate log lines; the streak itself is ALWAYS recomputed
    reason = None            # from scratch below, so a restart mid-run (e.g. watchdog killed, relaunched) can
                              # never mis-count the streak by skipping straight to the latest block.

    while True:
        rows = read_log(a.log)
        if len(rows) > last_len:
            last_len, last_growth_t = len(rows), time.time()
        if not pid_alive(a.pid):
            reason = 'training process exited on its own (reached epoch cap, crashed, or was killed externally)'
            break
        if rows and (time.time() - last_growth_t) / 60 > a.stall_minutes:
            reason = f'no new epoch logged for over {a.stall_minutes} minutes -- training appears stalled'
            break
        if rows:
            n_epochs = rows[-1][0]
            n_blocks = n_epochs // a.block
            if n_epochs >= a.min_epochs and n_blocks >= 2:
                low_streak, gain, b = 0, None, None
                for b in range(2, n_blocks + 1):                     # recompute the FULL history every tick
                    cur = block_max(rows, (b - 1) * a.block, b * a.block)
                    prev = block_max(rows, (b - 2) * a.block, (b - 1) * a.block)
                    if cur is None or prev is None:
                        continue
                    gain = cur - prev
                    if b not in printed_blocks:
                        printed_blocks.add(b)
                        print(f'[watchdog] epoch {n_epochs}: block {(b-1)*a.block+1}-{b*a.block} val_RPA_50c max '
                              f'{100*cur:.2f}  vs block {(b-2)*a.block+1}-{(b-1)*a.block} max {100*prev:.2f}  '
                              f'gain {100*gain:+.3f} pt', flush=True)
                    low_streak = low_streak + 1 if gain <= a.delta else 0
                if gain is not None and low_streak >= a.consecutive:
                    reason = (f'validation RPA plateaued: {low_streak} consecutive 100-epoch blocks each '
                              f'gained <= {100*a.delta:.2f} pt (last gain {100*gain:+.3f} pt) at epoch {n_epochs}')
                    break
        time.sleep(a.poll_seconds)

    print(f'[watchdog] STOPPING: {reason}', flush=True)
    if pid_alive(a.pid):
        os.kill(a.pid, signal.SIGTERM)
        for _ in range(30):
            if not pid_alive(a.pid):
                break
            time.sleep(2)
        if pid_alive(a.pid):
            os.kill(a.pid, signal.SIGKILL)
    time.sleep(5)   # let the last checkpoint write finish

    rows = read_log(a.log)
    if rows:
        best_epoch = None
        best_val = -1
        for e, v in rows:
            if v >= best_val:
                best_val, best_epoch = v, e
        print(f'[watchdog] stopped after epoch {rows[-1][0]}; best-validation epoch {best_epoch} '
              f'(val_RPA_50c {100*best_val:.2f})', flush=True)

    print('[watchdog] running final test evaluation on the best-validation checkpoint...', flush=True)
    r = subprocess.run([PY, '-W', 'ignore', 'scripts/evaluate_ptdb.py', '--config', a.eval_config],
                       cwd=REPO, capture_output=True, text=True)
    print(r.stdout, flush=True)
    if r.returncode != 0:
        print('[watchdog] EVALUATION FAILED:', r.stderr[-4000:], flush=True)
        sys.exit(1)
    print('[watchdog] done.', flush=True)


if __name__ == '__main__':
    main()
