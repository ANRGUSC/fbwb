@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\run.py --mode corrected --vision flyvis --decoder population --scene explain --duration 18 --panel --output results\flyvis-population
if errorlevel 1 pause
