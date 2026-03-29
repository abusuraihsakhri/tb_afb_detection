@echo off
echo =======================================================
echo     TB PATHOLOGY INTELLIGENCE - WSI TILE EXTRACTOR
echo =======================================================
echo.
set /p wsi_path="Enter the absolute path to your medical WSI file (e.g., C:\slides\patient_1.svs): "

echo.
echo Initiating Native Deep Zoom CPU Extraction Array...
echo System Target: Multi-core CPU + SSD I/O Stream
echo.

python 02_CODE\scripts\01_extract_tiles.py --wsi "%wsi_path%"

echo.
echo Extraction Pipeline Terminated. 
echo All clinical tiles have been securely written to 01_DATA\raw_tiles\.
pause
