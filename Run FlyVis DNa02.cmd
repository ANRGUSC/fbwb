@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\run.py --mode corrected --vision flyvis --decoder dna02 --scene explain --duration 18 --panel --output results\flyvis-dna02
if errorlevel 1 pause
