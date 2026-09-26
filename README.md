# TB AFB Detection

Research and development toolkit for tiled acid-fast bacilli (AFB) candidate detection in Ziehl-Neelsen microscopy images and whole-slide images.

> **Research use only.** This repository is not a validated medical device and does not provide a diagnostic result. Model and heuristic detections are candidates that require expert review.

## What is included

- Local FastAPI web application for raster-image screening and annotation.
- YOLO training workflow with CPU, CUDA, or Apple MPS device selection when supported by the installed PyTorch build.
- OpenSlide-based WSI tiling and sliding-window inference for supported whole-slide formats.
- AFB-focused postprocessing with global non-maximum suppression and morphology filters.
- Two-stain optical-density normalization using a Macenko-style method.
- A standalone WHO/IUATLD smear-grading utility for **confirmed microscopy counts with a real examined-field protocol**.
- Docker and Docker Compose runtime definitions.
- Lightweight regression tests and GitHub Actions CI.

The repository includes generic `yolov8n.pt` initialization weights. It does **not** include a validated AFB-specific trained checkpoint. Train a model and use the resulting `best.pt` for AFB inference.

## Local web application

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

OpenSlide system libraries are required for WSI workflows. Examples:

```bash
# Ubuntu/Debian
sudo apt install libopenslide0 openslide-tools

# macOS
brew install openslide
```

Then start the local API:

```bash
python -m uvicorn 05_DEPLOYMENT.api.server:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/ui/` for screening or `http://127.0.0.1:8001/ui/annotate.html` for annotation.

Windows users can use `Start_Detection_Engine.bat` or `Start_Annotation_Engine.bat`. macOS/Linux launchers are provided as `Start_Detection_Engine.sh` and `Start_Annotation_Engine.sh`.

### Web upload scope

The web endpoints accept raster files that OpenCV can decode: JPG, PNG, TIFF, and supported JPEG 2000 variants. The server enforces a 250 MB upload limit and a decoded-pixel limit.

Large pyramidal WSI formats such as SVS, NDPI, MRXS, VMS/VMU, SCN, and BIF should use the local OpenSlide pipeline rather than the raster upload endpoint. DICOM is not implemented by the current web decoder.

If no trained `best.pt` is present under `03_MODELS/` or `runs/`, the web application falls back to a simple color/morphology heuristic. That fallback is explicitly labeled as heuristic and must not be interpreted as a trained classifier.

## Training

Annotations saved by the local annotation tool are written to `01_DATA/processed_tiles/{train,val}` in normalized YOLO format.

Run training from the repository root:

```bash
python 02_CODE/scripts/02_train.py --data data.yaml --epochs 100 --batch 16
```

Ultralytics writes trained checkpoints under `runs/detect/.../weights/`. The API detects the newest local `best.pt` under `03_MODELS/` or `runs/`.

The HTTP training trigger is disabled by default. The trusted local annotation launchers enable it for that local session with `TB_AFB_ENABLE_TRAINING_TRIGGER=1`.

## WSI inference

Place research slides under `01_DATA/`, for example `01_DATA/raw_wsi/slide.svs`, and supply an AFB-trained checkpoint:

```bash
python 02_CODE/scripts/04_inference.py \
  --model runs/detect/train/weights/best.pt \
  --wsi raw_wsi/slide.svs \
  --conf 0.25
```

The output reports candidate detections, processing time, and tiles processed. It deliberately does **not** convert model candidates into a WHO/IUATLD smear grade.

For confirmed microscopy counts, the grading utility can be used independently:

```python
from tb_afb.inference.who_grader import WHOGrader

result = WHOGrader().calculate_grade(afb_count=42, fields_examined=100)
print(result["report_string"])
```

Install the package in editable mode first when importing it directly:

```bash
python -m pip install -e 02_CODE
```

## Tile extraction

```bash
python 02_CODE/scripts/01_extract_tiles.py \
  --wsi /path/to/slide.svs \
  --out 01_DATA/raw_tiles \
  --size 512 \
  --overlap 0
```

OpenSlide-native WSI extraction uses multiple worker processes. Standard raster images use the OpenCV fallback.

## Docker

```bash
docker compose up --build
```

The Compose configuration binds the application to localhost at port 8001, mounts local data/model/log directories, and keeps the training trigger disabled by default.

## Data handling and security

- `01_DATA/`, `03_MODELS/`, `runs/`, and `06_LOGS/` outputs are excluded from version control by default.
- Uploaded annotation images are decoded and re-encoded before storage; bounding boxes are range-validated.
- File resolution helpers reject paths that escape configured data roots.
- Upload size, decoded image size, individual WSI reads, and cached WSI handles are bounded.
- The local API has **no user authentication**. Keep it on localhost or place it behind an authenticated reverse proxy before network exposure.
- State-changing annotation/training browser requests require an ephemeral same-origin request token to reduce cross-origin form/CSRF abuse; this is not a substitute for authentication.
- Only load model checkpoints from trusted sources. PyTorch/Ultralytics checkpoint formats can execute unsafe deserialization paths depending on the loader and version.
- Git ignore rules do not de-identify data or provide encryption. Users remain responsible for PHI/PII handling and institutional policy.

See [SECURITY.md](SECURITY.md) for the security model and vulnerability reporting guidance.

## Tests

Lightweight checks do not require model downloads:

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=02_CODE/src pytest -q
python -m compileall -q 02_CODE 05_DEPLOYMENT/api
```

GitHub Actions runs syntax checks, critical Ruff checks, and the core unit tests on Python 3.10 and 3.12.

## Technology and compatibility

Core components are Python, PyTorch/Ultralytics, OpenCV, OpenSlide, FastAPI, NumPy, and fpdf2. The local UI uses standard HTML/CSS/JavaScript.

The UI is intended for current desktop versions of Chrome, Edge, Firefox, and Safari; there is no automated cross-browser test suite. WSI support additionally depends on the local OpenSlide installation and the slide vendor format.

## GitHub Pages and browser Python

GitHub Pages is not used for the application. The main workflow requires a Python API process, native OpenSlide libraries, PyTorch/Ultralytics, local model files, and potentially very large WSIs. Those requirements are not a practical fit for a static Pages/Pyodide deployment.

## License

The repository source is distributed under Apache License 2.0; see [LICENSE](LICENSE). Third-party dependencies and model artifacts retain their own licenses. In particular, Ultralytics currently offers its YOLO software/models under AGPL-3.0 or an Enterprise license, so deployments using Ultralytics must satisfy the applicable Ultralytics license terms.
