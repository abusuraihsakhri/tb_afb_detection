@echo off
echo =======================================================
echo     TB PATHOLOGY INTELLIGENCE - DEEP LEARNING ENGINE
echo =======================================================
echo Injecting Annotated Arrays into YOLOv8 PyTorch Backend...

python 02_CODE\scripts\02_train.py --data data.yaml

echo.
echo Training Event Complete. Check 03_MODELS\experiments\exp_1\weights for new best.pt.
pause
