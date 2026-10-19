@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\run.py --mode corrected --vision optic-flow --decoder population --scene explain --duration 18 --panel --output results\legacy-optical-flow
if errorlevel 1 pause
