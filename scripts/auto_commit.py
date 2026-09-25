"""Commit (and push) the small monitoring files of the running experiment. Never adds code, data, checkpoints or predictions.
Push token is read from env GH_PUSH_TOKEN (never stored). Usage: python scripts/auto_commit.py "<message>"   (no message -> auto text)"""
import base64, csv, fcntl, os, subprocess, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = 'results/MF-PAM-PTDB-001'
PATHS = [f'{RES}/training_log.csv', f'{RES}/validation_metrics.csv', f'{RES}/test_monitor.csv', f'{RES}/monitor']
TRAILER = 'Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>'


def git(*a, check=True):
    return subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True, check=check)


def commit_and_push(message=None):
    with open(os.path.join(REPO, '.auto_commit.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        present = [p for p in PATHS if os.path.exists(os.path.join(REPO, p))]
        if not present:
            return 'nothing to add'
        git('add', '--', *present)
        if git('diff', '--cached', '--quiet', check=False).returncode == 0:
            return 'no changes'
        if message is None:
            try:
                last = list(csv.DictReader(open(os.path.join(REPO, PATHS[0]))))[-1]
                message = f"auto: training log snapshot at epoch {last['epoch']} (val RPA50 {100*float(last['val_RPA_50c']):.2f})"
            except Exception:
                message = 'auto: training log snapshot'
        git('commit', '-q', '-m', message + '\n\n' + TRAILER, '--', *present)
        tok = os.environ.get('GH_PUSH_TOKEN')
        if not tok:
            return 'committed locally (no GH_PUSH_TOKEN, not pushed)'
        b = base64.b64encode(f'Nishchalam:{tok}'.encode()).decode()
        r = subprocess.run(['git', '-c', f'http.extraHeader=Authorization: Basic {b}', 'push', '-q', 'origin', 'ptdb-reproduction'],
                           cwd=REPO, capture_output=True, text=True)
        return 'committed + pushed' if r.returncode == 0 else 'committed; push failed: ' + r.stderr.replace(tok, '***')[-200:]


if __name__ == '__main__':
    print(commit_and_push(sys.argv[1] if len(sys.argv) > 1 else None))
