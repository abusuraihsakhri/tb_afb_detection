@echo off
setlocal
cd /d "%~dp0"
set TB_AFB_ENABLE_TRAINING_TRIGGER=1

echo =======================================================
echo        TB AFB RESEARCH - ANNOTATION TOOL
echo =======================================================

python -c "import torch, ultralytics, fastapi, cv2, pydantic" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Required Python packages are missing.
    echo Run: python -m pip install -r requirements.txt
    pause
    exit /b 1
)

start "TB_AFB_API" cmd /c "python -m uvicorn 05_DEPLOYMENT.api.server:app --host 127.0.0.1 --port 8001"
timeout /t 3 >nul
start http://127.0.0.1:8001/ui/annotate.html
exit /b 0
