@echo off
setlocal

rem ====== Configuration ======
set APP_NAME=My App
set ENV_NAME=nedd
set REPO_ROOT=%~dp0..\   rem repo root relative to scripts folder
set ENV_FILE=%REPO_ROOT%environment.windows.yml
set ENTRYPOINT=%REPO_ROOT%main.py

echo.
echo === %APP_NAME% Installer (Windows) ===
echo This will set up a conda environment and a desktop shortcut.
echo.

rem ----- Find conda -----
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Conda not found. We can install Miniconda (recommended).
    choice /M "Download and install Miniconda (yes/no)?"
    if errorlevel 2 (
        echo You chose not to install Miniconda. Cannot continue.
        pause
        exit /b 1
    )
    set MINICONDA_EXE=%TEMP%\Miniconda3-latest-Windows-x86_64.exe
    echo Downloading Miniconda...
    powershell -Command "Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile '%MINICONDA_EXE%'"
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to download Miniconda.
        pause
        exit /b 1
    )
    echo Installing Miniconda (silent)...
    "%MINICONDA_EXE%" /InstallationType=JustMe /AddToPath=1 /S /D=%USERPROFILE%\Miniconda3
    if %ERRORLEVEL% NEQ 0 (
        echo Miniconda installation failed.
        pause
        exit /b 1
    )
    set "PATH=%USERPROFILE%\Miniconda3;%USERPROFILE%\Miniconda3\Library\bin;%USERPROFILE%\Miniconda3\Scripts;%PATH%"
)

for /f "delims=" %%i in ('where conda') do set CONDA_EXE=%%i
call "%CONDA_EXE%\..\..\condabin\conda.bat" init cmd >nul 2>nul

echo Creating/updating environment from %ENV_FILE% ...
if not exist "%ENV_FILE%" (
    echo ERROR: %ENV_FILE% not found.
    pause
    exit /b 1
)

rem Create if missing; otherwise update (idempotent)
call conda env list | findstr /I "\b%ENV_NAME%\b" >nul
if %ERRORLEVEL% NEQ 0 (
    call conda env create -f "%ENV_FILE%"
) else (
    call conda env update -n %ENV_NAME% -f "%ENV_FILE%"
)
if %ERRORLEVEL% NEQ 0 (
    echo Conda environment creation/update failed.
    pause
    exit /b 1
)

echo Activating env...
call "%CONDA_EXE%\..\..\condabin\conda.bat" activate %ENV_NAME%
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate env.
    pause
    exit /b 1
)

echo Creating start script in repo root...
copy /Y "%~dp0start_app.bat" "%REPO_ROOT%start_app.bat" >nul

echo Creating desktop shortcut...
powershell -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1" ^
  -ShortcutPath "$([Environment]::GetFolderPath('Desktop'))\Start %APP_NAME%.lnk" ^
  -TargetPath "%REPO_ROOT%start_app.bat" ^
  -IconPath "%SystemRoot%\System32\SHELL32.dll" ^
  -Description "%APP_NAME% Launcher"

echo.
echo Installation complete. A desktop shortcut "Start %APP_NAME%" has been created.
echo Double-click it to run the app.
pause
