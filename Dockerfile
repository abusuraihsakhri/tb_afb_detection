# 🛡️ PORTABLE CLINICAL DEP-ENV: CUDA 12.2 + Python 3.12 
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system-level medical binary decoders (OpenSlide, VIPS)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libopenslide0 \
    openslide-tools \
    libvips-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire project architecture
COPY . .

# Expose the Clinical API port
EXPOSE 8001

# Run the FastAPI server natively via Uvicorn
CMD ["python3", "-m", "uvicorn", "05_DEPLOYMENT.api.server:app", "--host", "0.0.0.0", "--port", "8001"]
