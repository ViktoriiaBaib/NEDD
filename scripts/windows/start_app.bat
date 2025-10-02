@echo off
setlocal

set ENV_NAME=nedd

rem Find conda
for /f "delims=" %%i in ('where conda') do set CONDA_EXE=%%i
call "%CONDA_EXE%\..\..\condabin\conda.bat" activate %ENV_NAME%

rem Move into project root (two levels up from scripts/windows/)
cd /d "%~dp0..\.."

rem Launch Streamlit app
python -m streamlit run main.py