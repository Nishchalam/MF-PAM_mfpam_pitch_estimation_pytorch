import json, os, platform, subprocess, sys, shutil
import torch, yaml


def _run(cmd, cwd=None):
    try:
        return subprocess.check_output(cmd, shell=True, cwd=cwd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f'UNAVAILABLE: {e}'


def write_manifest(cfg, cfg_path, out_dir, repo, dataset_summary=None):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(cfg_path, os.path.join(out_dir, 'config.yaml'))
    with open(os.path.join(out_dir, 'config_resolved.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, 'git_commit.txt'), 'w') as f:
        f.write(f"commit: {_run('git rev-parse HEAD', repo)}\nbranch: {_run('git rev-parse --abbrev-ref HEAD', repo)}\n"
                f"upstream_base: 9303df5 (official MF-PAM)\ndirty_files:\n{_run('git status --short', repo)}\n"
                f"diff_model_files_vs_upstream:\n{_run('git diff 9303df5 -- model.py module.py augment.py utils.py dataset.py', repo) or '(none)'}\n")
    with open(os.path.join(out_dir, 'environment.txt'), 'w') as f:
        f.write(f"python {sys.version}\nplatform {platform.platform()}\ntorch {torch.__version__} cuda {torch.version.cuda} "
                f"cudnn {torch.backends.cudnn.version()}\n\n{_run('nvidia-smi')}\n\n{_run(sys.executable + ' -m pip freeze')}\n")
    if dataset_summary is not None:
        with open(os.path.join(out_dir, 'dataset_summary.json'), 'w') as f:
            json.dump(dataset_summary, f, indent=2)
