@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Resolve project root (directory of this script)
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

set ENVDIR=%SCRIPT_DIR%env\nedd_env

if not exist "%ENVDIR%\python.exe" (
  echo ERROR: Portable env not found at %ENVDIR%
  exit /b 1
)

REM First-run fixups (best-effort; ignore errors if not present)
if exist "%ENVDIR%\Scripts\conda-unpack.exe" (
  "%ENVDIR%\Scripts\conda-unpack.exe"
)

REM Run Streamlit app
"%ENVDIR%\python.exe" -m streamlit run ".\main.py"

