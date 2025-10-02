#!/bin/bash
set -e

APP_NAME="My App"
ENV_NAME="nedd"
REPO_ROOT="$( cd "$( dirname "$0" )/../.." && pwd )"
ENV_FILE="$REPO_ROOT/env/environment.macos.yml"

echo
echo "=== $APP_NAME Installer (macOS) ==="
echo

# Check for conda
if ! command -v conda &> /dev/null; then
  echo "Conda not found. Installing Miniforge (recommended for macOS)..."
  curl -L -o ~/Downloads/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
  bash ~/Downloads/Miniforge3.sh -b -p $HOME/miniforge3
  export PATH="$HOME/miniforge3/bin:$PATH"
  echo "Installed Miniforge."
fi

# Ensure conda initialized
eval "$(conda shell.bash hook)"

# Create or update environment
if conda env list | grep -q "^$ENV_NAME"; then
  echo "Updating existing environment: $ENV_NAME"
  conda env update -n $ENV_NAME -f "$ENV_FILE" --prune
else
  echo "Creating new environment: $ENV_NAME"
  conda env create -f "$ENV_FILE"
fi

# Copy launcher to project root for double-click
cp "$REPO_ROOT/scripts/macos/start_app.command" "$REPO_ROOT/start_app.command"
chmod +x "$REPO_ROOT/start_app.command"

echo
echo "Installation complete. Use 'start_app.command' in the project folder to run."
