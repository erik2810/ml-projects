@echo off
echo === Graph ML Lab — Setup ===

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: python not found. Install Python 3.10+ first.
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VERSION=%%i
echo Found Python %PY_VERSION%

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
echo Virtual environment activated.

pip install --upgrade pip --quiet

echo Installing PyTorch...
pip install torch --quiet

echo Installing project dependencies...
pip install -r requirements.txt --quiet

echo.
echo Setup complete. Activate the environment with:
echo   .venv\Scripts\activate.bat
echo.
echo Then start the app with:
echo   run.bat
