#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 - <<'PY'
import importlib
required = ["torch", "fastapi", "cv2", "pydantic"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    raise SystemExit("Missing dependencies: " + ", ".join(missing) + ". Run: python3 -m pip install -r requirements.txt")
PY

exec python3 -m uvicorn 05_DEPLOYMENT.api.server:app --host 127.0.0.1 --port 8001
