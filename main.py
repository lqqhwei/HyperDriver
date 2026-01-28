import subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

def run(cmd):
    print(f"\n[RUN] {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise SystemExit(f"[ERR] Command failed：{cmd}")

run(f"{sys.executable} {ROOT / 'preprocess_loop.py'}")
run(f"{sys.executable} {ROOT / 'train_hyperdriver.py'}")
run(f"{sys.executable} {ROOT / 'baselines_centrality.py'}")
run(f"{sys.executable} {ROOT / 'eval_driver.py'}")
run(f"{sys.executable} {ROOT / 'plot_nature_figs.py'}")
