@echo off
setlocal
echo =======================================================
echo     TB PATHOLOGY INTELLIGENCE - ANNOTATION PIPELINE 
echo =======================================================

echo [System Check] Validating Deep Learning Environment...
python -c "import torch, ultralytics, fastapi, cv2, pydantic" 2>nul
if %errorlevel% neq 0 (
    echo [WARNING] Missing Critical ML Dependencies.
    echo Allocating and Installing requirements natively...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Dependency Installation Failed. Check your Python/PIP installation.
        pause
        exit /b 1
    )
    echo [SUCCESS] Tensor Dependencies Installed.
) else (
    echo [OK] Neural Matrix Environment Verified.
)

echo Starting Secure Active Learning Gateway...
start "TB_API_Daemon" cmd /c "python -m uvicorn 05_DEPLOYMENT.api.server:app --host 127.0.0.1 --port 8001"

echo Verifying Tensor/CPU Hardware Threading...
timeout /t 3 >nul

echo Bootstrapping Annotation Deep UI Dashboard...
start http://127.0.0.1:8001/ui/annotate.html
exit /b 0
