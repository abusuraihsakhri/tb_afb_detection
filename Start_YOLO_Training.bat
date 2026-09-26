@echo off
setlocal
cd /d "%~dp0"

echo =======================================================
echo        TB AFB RESEARCH - YOLO TRAINING
echo =======================================================

python 02_CODE\scripts\02_train.py --data data.yaml
if %errorlevel% neq 0 (
    echo.
    echo Training failed. Review the error above.
    pause
    exit /b 1
)

echo.
echo Training completed. Ultralytics writes the best checkpoint under runs\detect\...\weights\best.pt.
pause
