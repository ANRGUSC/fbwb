@echo off
cd /d "%~dp0"
echo Running the stable-integration diagnostic, with upstream gains unchanged.
".venv\Scripts\python.exe" scripts\run.py --mode stable --duration 12
if errorlevel 1 pause
