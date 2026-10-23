@echo off
chcp 65001 >nul
rem Pishro local launcher (Windows): creates .venv, installs dependencies, opens the menu.
cd /d "%~dp0\.."
set PYTHONUTF8=1
if not exist .venv (
  py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
if not exist .venv\.installed (
  python -m pip install -q --upgrade pip
  pip install -q -r requirements-dev.txt || pip install -q -i https://package-mirror.liara.ir/repository/pypi/simple -r requirements-dev.txt
  if not errorlevel 1 type nul > .venv\.installed
)
python local\menu.py
pause
