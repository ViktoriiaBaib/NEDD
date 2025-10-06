@echo off
title Launch NEDD
cd /d "%~dp0"

echo Running path fixups...
if exist "env\nedd_env\Scripts\conda-unpack.exe" (
  "env\nedd_env\Scripts\conda-unpack.exe"
)

echo Starting Streamlit server...
"env\nedd_env\python.exe" -m streamlit run main.py
pause