@echo off
cd /d "%~dp0"
echo This reproduces the original numerically unstable neural update.
".venv\Scripts\python.exe" scripts\run.py --mode upstream --scene diagnostic --duration 8
if errorlevel 1 pause
