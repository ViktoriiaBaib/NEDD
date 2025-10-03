# Launch Streamlit using the packed, portable conda environment
# Usage: Right-click -> Run with PowerShell (or double-click if .ps1 is associated)

$ErrorActionPreference = "Stop"

# Resolve project root and env path (robust to where you run it from)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ScriptDir

$EnvDir = Join-Path $ScriptDir "env\nedd_env"

if (!(Test-Path $EnvDir)) {
  Write-Host "ERROR: Portable env not found at $EnvDir" -ForegroundColor Red
  exit 1
}

# First run after unzip: fix paths inside the env (idempotent; cheap if already done)
$CondaUnpack = Join-Path $EnvDir "Scripts\conda-unpack.exe"
if (Test-Path $CondaUnpack) {
  Write-Host "Running conda-unpack (first-run fixups)..." -ForegroundColor Cyan
  & $CondaUnack 2>$null
  if ($LASTEXITCODE -ne 0) {
    # Some builds name it conda-unpack.exe; others install conda-unpack as a Python module.
    # Fallback to module invocation if exe invocation fails.
    & "$EnvDir\python.exe" -m conda_pack --unpack
  }
}

# Run Streamlit app with the env's python (no activation required)
& "$EnvDir\python.exe" -m streamlit run ".\main.py"

Pop-Location

