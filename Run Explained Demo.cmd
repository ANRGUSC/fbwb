@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\run.py --mode corrected --scene explain --duration 18 --panel --output results\explained-bilateral
if errorlevel 1 pause
