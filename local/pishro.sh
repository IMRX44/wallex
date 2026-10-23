#!/usr/bin/env bash
# Pishro local launcher (Linux / macOS): creates .venv, installs dependencies, opens the menu.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -c 'import sys; v=sys.version_info; assert (3,10)<=v[:2]<=(3,13), f"Python 3.10-3.13 needed, found {v.major}.{v.minor}"'
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
export PYTHONUTF8=1
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  pip install -q --upgrade pip || true
  pip install -q -r requirements-dev.txt || pip install -q -i https://package-mirror.liara.ir/repository/pypi/simple -r requirements-dev.txt
  touch .venv/.installed
fi
python local/menu.py
