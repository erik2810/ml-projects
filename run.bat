@echo off

if not defined VIRTUAL_ENV (
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
    ) else (
        echo No virtual environment found. Run setup.bat first.
        exit /b 1
    )
)

set HOST=0.0.0.0
set PORT=8000

echo Starting Graph ML Lab on http://%HOST%:%PORT%
echo Press Ctrl+C to stop.
echo.

python -m uvicorn backend.app:app --host %HOST% --port %PORT% --reload
