@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo .venv Python not found.
  echo Please create the virtual environment first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run dashboard.py
