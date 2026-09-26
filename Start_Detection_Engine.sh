#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import cv2, fastapi, pydantic, torch, ultralytics
print("Runtime dependencies available.")
PY

python3 -m uvicorn 05_DEPLOYMENT.api.server:app --host 127.0.0.1 --port 8001
