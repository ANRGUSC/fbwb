@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\live_panel.py --directory results\explained-bilateral --replay
