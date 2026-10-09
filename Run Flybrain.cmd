@echo off
cd /d "%~dp0"
echo Running unit-consistent neural dynamics in the corridor scene.
echo This is an experimental model, not validated biological navigation.
echo The first 6 simulation seconds use the ordinary hover controller.
".venv\Scripts\python.exe" scripts\run.py --mode corrected --scene corridor --duration 12 --panel
if errorlevel 1 pause
