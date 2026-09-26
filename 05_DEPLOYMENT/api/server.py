import json
import os
import subprocess
import sys
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF
from pydantic import BaseModel, Field, model_validator

try:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator
except ImportError:
    openslide = None
    DeepZoomGenerator = None

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

current_dir = Path(__file__).parent.resolve()
root_dir = current_dir.parent.parent
src_dir = root_dir / "02_CODE" / "src"
sys.path.append(str(src_dir))

from tb_afb.utils.paths import resolve_within

app = FastAPI(title="TB AFB Research API")

static_dir = current_dir / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

MAX_FILE_SIZE = 250 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
MAX_DECODED_PIXELS = 100_000_000
WEB_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".jp2",
    ".j2k",
    ".jpf",
    ".jpx",
}
CLASS_LABELS = {
    0: "AFB_Definite",
    1: "AFB_Probable",
    2: "AFB_Possible",
    3: "Debris",
    4: "RBC",
}


class BoundingBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)
    label: int = Field(default=0, ge=0, le=4)

    @model_validator(mode="after")
    def box_must_stay_inside_image(self):
        if self.x - self.width / 2 < 0 or self.x + self.width / 2 > 1:
            raise ValueError("Bounding box exceeds horizontal image bounds.")
        if self.y - self.height / 2 < 0 or self.y + self.height / 2 > 1:
            raise ValueError("Bounding box exceeds vertical image bounds.")
        return self


class InferenceResult(BaseModel):
    detections: list[dict[str, Any]]
    message: str
    grade: str
    hardware: str
    analysis_mode: str
    candidate_count: int


class ReportRequest(BaseModel):
    filename: str = Field(default="Unknown", max_length=200)
    grade: str = Field(default="Not calculated", max_length=100)
    count: int = Field(default=0, ge=0, le=1_000_000)
    hardware: str = Field(default="Unknown", max_length=200)
    pathologist_name: str = Field(default="Not specified", max_length=120)


ACTIVE_MODEL = None
ACTIVE_MODEL_PATH = None
TRAINING_PROCESS = None
WSI_HANDLES = OrderedDict()
MAX_WSI_HANDLES = 4


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return "NVIDIA CUDA"
    if device.type == "mps":
        return "Apple Metal/MPS"
    return "CPU"


def _candidate_weight_files() -> list[Path]:
    candidates = []
    for base in (root_dir / "03_MODELS", root_dir / "runs"):
        if base.exists():
            candidates.extend(path for path in base.rglob("best.pt") if path.is_file())
    return candidates


def load_active_model():
    """Load the newest locally generated/curated best.pt checkpoint."""
    global ACTIVE_MODEL, ACTIVE_MODEL_PATH

    if not ULTRALYTICS_AVAILABLE:
        return None

    weights = _candidate_weight_files()
    if not weights:
        return None
    weights.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    latest = weights[0].resolve()

    if latest != ACTIVE_MODEL_PATH:
        try:
            candidate = YOLO(str(latest))
        except Exception as exc:
            print(f"[MODEL] Could not load {latest}: {exc}", file=sys.stderr)
            return ACTIVE_MODEL
        ACTIVE_MODEL = candidate
        ACTIVE_MODEL_PATH = latest

    return ACTIVE_MODEL


def _validate_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Upload filename is missing.")
    safe_name = Path(filename).name
    lower = safe_name.lower()
    if not any(lower.endswith(ext) for ext in WEB_IMAGE_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported web-upload format. The web analyzer accepts raster images "
                "that OpenCV can decode; use the local WSI pipeline for SVS/NDPI/MRXS/DICOM."
            ),
        )
    return safe_name


async def _read_upload_limited(request: Request, file: UploadFile) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_FILE_SIZE + 1024 * 1024:
                raise HTTPException(status_code=413, detail="Upload exceeds the 250 MB limit.")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

    chunks = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Upload exceeds the 250 MB limit.")
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return b"".join(chunks)


def _decode_raster(contents: bytes) -> np.ndarray:
    image_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=415, detail="OpenCV could not decode this raster image.")
    height, width = image.shape[:2]
    if height <= 0 or width <= 0 or height * width > MAX_DECODED_PIXELS:
        raise HTTPException(status_code=413, detail="Decoded image dimensions exceed safe limits.")
    return image


def _heuristic_detections(image: np.ndarray) -> list[dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([130, 40, 20]), np.array([175, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 15 < area < 300:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio < 1.5:
            continue
        score = min(0.99, 0.70 + aspect_ratio / 10.0)
        class_id = 0 if score > 0.85 else 2
        detections.append(
            {
                "bbox": [x + width / 2.0, y + height / 2.0, width + 4, height + 4],
                "confidence": round(score, 3),
                "class_id": class_id,
                "label": CLASS_LABELS[class_id],
            }
        )
    return detections


def _model_detections(model, image: np.ndarray, device: torch.device) -> list[dict[str, Any]]:
    device_arg: Any = 0 if device.type == "cuda" else device.type
    result = model.predict(
        source=image,
        conf=0.50,
        max_det=5000,
        device=device_arg,
        verbose=False,
    )[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        detections.append(
            {
                "bbox": box.xywh[0].tolist(),
                "confidence": round(float(box.conf[0]), 4),
                "class_id": class_id,
                "label": CLASS_LABELS.get(class_id, f"class_{class_id}"),
            }
        )
    return detections


def _pdf_text(value: Any, max_length: int = 200) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")[:max_length]
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _training_enabled() -> bool:
    return os.environ.get("TB_AFB_ENABLE_TRAINING_TRIGGER", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "ultralytics_available": ULTRALYTICS_AVAILABLE,
        "openslide_available": openslide is not None,
    }


@app.post("/api/v1/analyze", response_model=InferenceResult)
async def analyze_slide(request: Request, file: UploadFile = File(...)):
    _validate_filename(file.filename)
    contents = await _read_upload_limited(request, file)
    image = _decode_raster(contents)

    device = _device()
    model = load_active_model()
    if model is not None:
        detections = _model_detections(model, image, device)
        mode = "yolo"
        message = "YOLO screening completed."
    else:
        detections = _heuristic_detections(image)
        mode = "heuristic"
        message = "Color-morphology heuristic screening completed; no trained checkpoint is active."

    return InferenceResult(
        detections=detections,
        message=message,
        grade="Not calculated",
        hardware=f"Hardware backend: {_device_label(device)}",
        analysis_mode=mode,
        candidate_count=len(detections),
    )


@app.post("/api/v1/save_annotation")
async def save_annotation(
    request: Request,
    file: UploadFile = File(...),
    boxes: str = Form(...),
):
    _validate_filename(file.filename)
    try:
        raw_boxes = json.loads(boxes)
        if not isinstance(raw_boxes, list) or not raw_boxes:
            raise ValueError("Annotation array must be non-empty.")
        if len(raw_boxes) > 5000:
            raise ValueError("Too many annotations in one request.")
        parsed_boxes = [BoundingBox.model_validate(item) for item in raw_boxes]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid annotation payload: {exc}")

    contents = await _read_upload_limited(request, file)
    image = _decode_raster(contents)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not normalize annotation image to JPEG.")

    base_name = uuid.uuid4().hex
    split = "val" if int(base_name[:2], 16) < 51 else "train"
    data_dir = root_dir / "01_DATA" / "processed_tiles" / split
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    (img_dir / f"{base_name}.jpg").write_bytes(encoded.tobytes())
    with (lbl_dir / f"{base_name}.txt").open("w", encoding="utf-8") as handle:
        for box in parsed_boxes:
            handle.write(f"{box.label} {box.x:.8f} {box.y:.8f} {box.width:.8f} {box.height:.8f}\n")

    return {
        "status": "success",
        "message": f"Saved {len(parsed_boxes)} annotations to the {split} split.",
    }


@app.post("/api/v1/trigger_training")
async def trigger_training():
    global TRAINING_PROCESS

    if not _training_enabled():
        raise HTTPException(
            status_code=403,
            detail="Training trigger is disabled. Set TB_AFB_ENABLE_TRAINING_TRIGGER=1 for a trusted local session.",
        )
    if TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None:
        raise HTTPException(status_code=409, detail="A training process is already running.")

    labels_dir = root_dir / "01_DATA" / "processed_tiles" / "train" / "labels"
    if not labels_dir.exists() or not any(labels_dir.glob("*.txt")):
        raise HTTPException(status_code=400, detail="No training annotations are available.")

    train_script = root_dir / "02_CODE" / "scripts" / "02_train.py"
    log_dir = root_dir / "06_LOGS"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (log_dir / "training_process.log").open("ab")
    try:
        TRAINING_PROCESS = subprocess.Popen(
            [sys.executable, str(train_script), "--data", "data.yaml"],
            cwd=str(root_dir),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    return {"status": "success", "message": "Training process started."}


@app.post("/api/v1/render_payload")
async def render_payload(request: Request, file: UploadFile = File(...)):
    _validate_filename(file.filename)
    contents = await _read_upload_limited(request, file)
    image = _decode_raster(contents)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not render image preview.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/api/v1/stats")
async def get_stats():
    train_dir = root_dir / "01_DATA" / "processed_tiles" / "train"
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"

    total_images = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
    total_annotations = 0
    if lbl_dir.exists():
        for txt_file in lbl_dir.glob("*.txt"):
            with txt_file.open("r", encoding="utf-8") as handle:
                total_annotations += sum(1 for line in handle if line.strip())

    return {
        "images_annotated": total_images,
        "afb_instances": total_annotations,
        "model_deployed": bool(_candidate_weight_files()),
        "training_trigger_enabled": _training_enabled(),
    }


@app.post("/api/v1/export_report")
async def export_report(data: ReportRequest):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, text="TB AFB RESEARCH SCREENING REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", size=10)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 8, text=f"Generated: {generated}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, text="Screening summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    rows = [
        ("Analysis target", data.filename),
        ("Smear grade", data.grade),
        ("Candidate detections", data.count),
        ("Hardware", data.hardware),
        ("Reviewer", data.pathologist_name),
    ]
    for label, value in rows:
        pdf.multi_cell(0, 6, text=f"{label}: {_pdf_text(value)}")

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        5,
        text=(
            "Research-use output. Candidate detections and heuristic scores are not a validated diagnostic result. "
            "A smear grade is only valid when based on an appropriate microscopy field-count protocol and human review."
        ),
    )

    return Response(
        content=bytes(pdf.output()),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=TB_AFB_Research_Report.pdf"},
    )


def _deepzoom_for(wsi_id: str):
    if openslide is None or DeepZoomGenerator is None:
        raise HTTPException(status_code=501, detail="OpenSlide DeepZoom is not available.")

    safe_id = Path(wsi_id).name
    if safe_id != wsi_id:
        raise HTTPException(status_code=400, detail="Invalid WSI identifier.")

    if safe_id in WSI_HANDLES:
        WSI_HANDLES.move_to_end(safe_id)
        return WSI_HANDLES[safe_id][1]

    base_data_dir = root_dir / "01_DATA" / "raw_wsi"
    try:
        wsi_path = resolve_within(base_data_dir, safe_id, must_exist=True)
    except (PermissionError, FileNotFoundError):
        raise HTTPException(status_code=404, detail="WSI file not found.")

    try:
        slide = openslide.OpenSlide(str(wsi_path))
        generator = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=False)
    except Exception:
        raise HTTPException(status_code=415, detail="OpenSlide could not open this WSI.")

    WSI_HANDLES[safe_id] = (slide, generator)
    WSI_HANDLES.move_to_end(safe_id)
    while len(WSI_HANDLES) > MAX_WSI_HANDLES:
        _, (old_slide, _) = WSI_HANDLES.popitem(last=False)
        old_slide.close()
    return generator


@app.get("/api/v1/wsi/info/{wsi_id}")
async def get_wsi_info(wsi_id: str):
    generator = _deepzoom_for(wsi_id)
    width, height = generator.level_dimensions[-1]
    return {
        "width": width,
        "height": height,
        "tile_size": 254,
        "tile_overlap": 1,
        "level_count": generator.level_count,
    }


@app.get("/api/v1/wsi/tile/{wsi_id}/{z}/{x}/{y}")
async def get_wsi_tile(wsi_id: str, z: int, x: int, y: int):
    if z < 0 or x < 0 or y < 0:
        raise HTTPException(status_code=400, detail="Tile coordinates must be non-negative.")
    generator = _deepzoom_for(wsi_id)
    try:
        tile = generator.get_tile(z, (x, y))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid tile coordinates.")

    buffer = BytesIO()
    tile.save(buffer, format="JPEG")
    return Response(content=buffer.getvalue(), media_type="image/jpeg")
