#!/usr/bin/env bash
set -euo pipefail

echo "Start the local API, then open http://127.0.0.1:8001/ui/annotate.html"
python3 -m uvicorn 05_DEPLOYMENT.api.server:app --host 127.0.0.1 --port 8001
