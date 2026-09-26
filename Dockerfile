FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libopenslide0 \
    openslide-tools \
    libvips-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

COPY . .
RUN python3 -m pip install --no-deps -e 02_CODE \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/01_DATA /app/03_MODELS /app/06_LOGS \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8001/api/v1/health || exit 1

CMD ["python3", "-m", "uvicorn", "05_DEPLOYMENT.api.server:app", "--host", "0.0.0.0", "--port", "8001"]
