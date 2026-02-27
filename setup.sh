#!/usr/bin/env bash
set -euo pipefail

echo "=== Graph ML Lab — Setup ==="

# check python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PY_VERSION"

# create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "Virtual environment activated."

# upgrade pip
pip install --upgrade pip --quiet

# install pytorch (CPU by default — edit for CUDA if needed)
echo "Installing PyTorch..."
pip install torch --quiet 2>/dev/null || pip install torch --quiet

# install remaining dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then start the app with:"
echo "  ./run.sh"
