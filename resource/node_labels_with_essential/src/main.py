import subprocess, sys, shutil
from pathlib import Path

LABEL_DIR = Path(__file__).resolve().parents[1]
LABEL_DIR_SRC = LABEL_DIR / "src"

def run(cmd):
    print(f"\n[RUN] {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        raise SystemExit(f"[ERR] Command failed:{cmd}")

run(f"{sys.executable} {LABEL_DIR_SRC / 'build_deg_essential_orfs.py'}")
run(f"{sys.executable} {LABEL_DIR_SRC / 'make_OGEE_essential_orfs.py'}")
run(f"{sys.executable} {LABEL_DIR_SRC / 'make_essential_columns.py'}")
