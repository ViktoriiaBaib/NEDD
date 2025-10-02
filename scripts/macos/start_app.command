#!/bin/bash
# Double-clickable launcher for macOS

APP_NAME="My App"
ENV_NAME="nedd"

# Ensure conda is in PATH
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# cd into project root (two levels up from scripts/macos/)
cd "$(dirname "$0")/../.."

# Run Streamlit app
python -m streamlit run main.py