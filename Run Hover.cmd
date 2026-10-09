@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\run.py --mode hover --duration 30
if errorlevel 1 pause
