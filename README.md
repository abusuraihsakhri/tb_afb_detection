# TB-AFB Detection Research Toolkit

A local research toolkit for acid-fast bacillus (AFB) candidate detection, image tiling, annotation, YOLO training, and whole-slide-image (WSI) experimentation.

> **Research use only.** This repository is not a validated medical device, does not establish a tuberculosis diagnosis, and must not be used as the sole basis for clinical decisions.

## Main workflows

- **Patch-image screening UI:** local FastAPI interface for JPG/PNG/TIFF/JP2-family raster images.
- **Annotation UI:** draw normalized YOLO bounding boxes and store JPEG-normalized training samples locally.
- **YOLO training:** train an Ultralytics detector against the local `01_DATA/processed_tiles` dataset.
- **WSI CLI inference:** tiled inference for OpenSlide-compatible formats such as SVS and NDPI.
- **Local WSI viewer:** Deep Zoom viewing for slides placed in `01_DATA/raw_wsi`.

The repository contains base YOLO weights for convenience. Trained `best.pt` checkpoints are discovered only under `03_MODELS/` and `runs/`.

## Requirements

- Python 3.10+
- OpenSlide system libraries for WSI workflows
- Optional NVIDIA CUDA or Apple Metal/MPS acceleration

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install libopenslide0 openslide-tools
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e 02_CODE
```

macOS:

```bash
brew install openslide
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e 02_CODE
```

## Local web application

Start the API on loopback only:

```bash
python -m uvicorn 05_DEPLOYMENT.api.server:app --host 127.0.0.1 --port 8001
```

Open:

- Screening UI: `http://127.0.0.1:8001/ui/`
- Annotation UI: `http://127.0.0.1:8001/ui/annotate.html`
- Health endpoint: `http://127.0.0.1:8001/api/v1/health`

The web API rejects undecodable uploads instead of interpreting them as negative results. Complex OpenSlide WSI formats are handled by the CLI rather than the patch-image upload endpoint.

## WSI inference

Place slides below `01_DATA/raw_wsi/` and use a **trusted local** `.pt` checkpoint:

```bash
python 02_CODE/scripts/04_inference.py \
  --model /path/to/trusted/best.pt \
  --wsi raw_wsi/example.svs \
  --conf 0.25
```

WHO/IUATLD Ziehl-Neelsen smear grading depends on the number of microscope fields actually examined. The CLI therefore does not infer a grade from slide area. If a valid field count is known from the acquisition protocol, provide it explicitly:

```bash
python 02_CODE/scripts/04_inference.py \
  --model /path/to/trusted/best.pt \
  --wsi raw_wsi/example.svs \
  --fields-examined 100
```

## Training

The default dataset configuration is `02_CODE/data.yaml`.

```bash
python 02_CODE/scripts/generate_dummy.py
python 02_CODE/scripts/check_data_integrity.py
python 02_CODE/scripts/02_train.py --data data.yaml --epochs 10 --batch 4
```

Do not treat the synthetic generator as validation data for model performance.

## Docker

The Compose configuration publishes the service only on `127.0.0.1:8001` by default:

```bash
docker compose up --build
```

GPU container configuration is host-specific and is intentionally not enabled by default.

## Testing

```bash
python -m pip install -r requirements-dev.txt
pytest -q
python -m compileall 02_CODE 05_DEPLOYMENT
```

CI also performs a dependency vulnerability audit with `pip-audit`.

## Data handling and privacy

`01_DATA/`, `03_MODELS/`, `runs/`, and `06_LOGS/` are excluded from version control apart from optional placeholder files. The application does not intentionally transmit uploaded microscopy data to an external service. The WSI viewer loads OpenSeadragon from jsDelivr, so opening that viewer makes a normal browser request to that CDN.

Avoid placing patient-identifiable information in filenames, logs, annotations, or exported research reports.

## GitHub Pages

GitHub Pages is not an appropriate deployment target for the complete application. The working system requires FastAPI, PyTorch, OpenCV, OpenSlide/native libraries, local file storage, and optional GPU acceleration; those server-side/native requirements cannot run as a normal Pages site.

## Security

See [`SECURITY.md`](SECURITY.md). Model checkpoint files are deserialization inputs and should be loaded only from trusted sources.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
