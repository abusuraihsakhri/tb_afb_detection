import json
import random
import subprocess
import sys
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

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

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent
STATIC_DIR = CURRENT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="TB-AFB Research API")
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 250 * 1024 * 1024
SUPPORTED_IMAGE_EXTENSIONS = {
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
    def box_must_fit_image(self):
        if self.x - self.width / 2 < 0 or self.x + self.width / 2 > 1:
            raise ValueError("Bounding box exceeds horizontal image bounds.")
        if self.y - self.height / 2 < 0 or self.y + self.height / 2 > 1:
            raise ValueError("Bounding box exceeds vertical image bounds.")
        return self


class InferenceResult(BaseModel):
    detections: list
    message: str
    grade: str
    hardware: str


ACTIVE_MODEL = None
ACTIVE_MODEL_STAMP = None
TRAINING_PROCESS: Optional[subprocess.Popen] = None
WSI_HANDLES = OrderedDict()
MAX_OPEN_WSI_HANDLES = 4


async def read_upload_limited(file: UploadFile, limit: int = MAX_FILE_SIZE) -> bytes:
    """Read at most ``limit`` bytes and reject oversized bodies based on actual bytes read."""
    contents = await file.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {limit // (1024 * 1024)} MB.")
    return contents


def decode_image(contents: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=415,
            detail="The uploaded file could not be decoded as a supported raster image.",
        )
    return image


def validate_filename(filename: Optional[str]) -> str:
    normalized = Path(filename or "").name.lower()
    if not normalized:
        raise HTTPException(status_code=400, detail="A filename is required.")
    if not any(normalized.endswith(ext) for ext in SUPPORTED_IMAGE_EXTENSIONS):
        raise HTTPException(status_code=415, detail="Unsupported image format for the web API.")
    return normalized


def model_search_roots() -> list[Path]:
    return [PROJECT_ROOT / "03_MODELS", PROJECT_ROOT / "runs"]


def find_latest_weights() -> Optional[Path]:
    candidates: list[Path] = []
    for root in model_search_roots():
        if root.exists():
            candidates.extend(p for p in root.rglob("best.pt") if p.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def inference_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    if device == "0":
        return "NVIDIA CUDA"
    if device == "mps":
        return "Apple Metal/MPS"
    return "CPU"


def load_active_model():
    """Load the newest trusted local training checkpoint and reload if it changes in place."""
    global ACTIVE_MODEL, ACTIVE_MODEL_STAMP

    if not ULTRALYTICS_AVAILABLE:
        return None
    latest = find_latest_weights()
    if latest is None:
        ACTIVE_MODEL = None
        ACTIVE_MODEL_STAMP = None
        return None

    stat = latest.stat()
    stamp = (str(latest.resolve()), stat.st_mtime_ns, stat.st_size)
    if stamp != ACTIVE_MODEL_STAMP:
        ACTIVE_MODEL = YOLO(str(latest))
        ACTIVE_MODEL_STAMP = stamp
    return ACTIVE_MODEL


def research_summary(candidate_count: int) -> str:
    return f"Research candidate count: {int(candidate_count)}"


def run_fallback_candidate_detector(image: np.ndarray) -> list[dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_magenta = np.array([130, 40, 20])
    upper_magenta = np.array([175, 255, 255])
    mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
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
        confidence = min(0.99, 0.70 + (aspect_ratio / 10.0))
        detections.append(
            {
                "bbox": [x + width / 2.0, y + height / 2.0, width + 4, height + 4],
                "confidence": round(confidence, 4),
                "class_id": 0 if confidence > 0.85 else 2,
                "label": "AFB_Definite" if confidence > 0.85 else "AFB_Possible",
            }
        )
    return detections


def safe_pdf_text(value, max_length: int = 200) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")[:max_length]
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _wsi_path(wsi_id: str) -> Path:
    safe_name = Path(wsi_id).name
    base = (PROJECT_ROOT / "01_DATA" / "raw_wsi").resolve()
    path = (base / safe_name).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid WSI path.") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="WSI file not found.")
    return path


def _deepzoom_for(wsi_id: str):
    if openslide is None or DeepZoomGenerator is None:
        raise HTTPException(status_code=501, detail="OpenSlide DeepZoom is not available on this host.")

    safe_name = Path(wsi_id).name
    if safe_name in WSI_HANDLES:
        slide, dz = WSI_HANDLES.pop(safe_name)
        WSI_HANDLES[safe_name] = (slide, dz)
        return slide, dz

    slide = openslide.OpenSlide(str(_wsi_path(safe_name)))
    dz = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=False)
    WSI_HANDLES[safe_name] = (slide, dz)
    while len(WSI_HANDLES) > MAX_OPEN_WSI_HANDLES:
        _, (old_slide, _) = WSI_HANDLES.popitem(last=False)
        old_slide.close()
    return slide, dz


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "ultralytics_available": ULTRALYTICS_AVAILABLE,
        "openslide_available": openslide is not None,
        "trained_weights_available": find_latest_weights() is not None,
    }


@app.post("/api/v1/analyze", response_model=InferenceResult)
async def analyze_slide(request: Request, file: UploadFile = File(...)):
    validate_filename(file.filename)
    contents = await read_upload_limited(file)
    image = decode_image(contents)

    device = inference_device()
    model = load_active_model()
    if model is not None:
        try:
            results = model.predict(source=image, device=device, verbose=False)[0]
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model inference failed.") from exc

        detections = []
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < 0.50:
                continue
            class_id = int(box.cls[0])
            if class_id not in {0, 1, 2}:
                continue
            detections.append(
                {
                    "bbox": box.xywh[0].tolist(),
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "label": CLASS_LABELS.get(class_id, f"class_{class_id}"),
                }
            )
        message = "Ultralytics checkpoint inference executed. Research use only."
    else:
        detections = run_fallback_candidate_detector(image)
        message = "Color/morphology fallback executed. Heuristic output is not a diagnostic result."

    return InferenceResult(
        detections=detections,
        message=message,
        grade=research_summary(len(detections)),
        hardware=f"Backend: {device_label(device)}",
    )


@app.post("/api/v1/save_annotation")
async def save_annotation(request: Request, file: UploadFile = File(...), boxes: str = Form(...)):
    validate_filename(file.filename)
    try:
        boxes_list = json.loads(boxes)
        if not isinstance(boxes_list, list) or not boxes_list:
            raise ValueError("At least one bounding box is required.")
        parsed_boxes = [BoundingBox(**item) for item in boxes_list]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid annotation payload.") from exc

    contents = await read_upload_limited(file)
    image = decode_image(contents)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to normalize the annotation image.")

    base_name = uuid.uuid4().hex
    split = "val" if random.random() < 0.20 else "train"
    data_dir = PROJECT_ROOT / "01_DATA" / "processed_tiles" / split
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
        "message": f"Saved {len(parsed_boxes)} annotation(s) to the {split} split.",
    }


@app.post("/api/v1/trigger_training")
async def trigger_training():
    global TRAINING_PROCESS

    if TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None:
        raise HTTPException(status_code=409, detail="A training process is already running.")

    label_dir = PROJECT_ROOT / "01_DATA" / "processed_tiles" / "train" / "labels"
    if not label_dir.exists() or not any(label_dir.glob("*.txt")):
        raise HTTPException(status_code=400, detail="No training annotations are available.")

    train_script = PROJECT_ROOT / "02_CODE" / "scripts" / "02_train.py"
    data_yaml = PROJECT_ROOT / "02_CODE" / "data.yaml"
    TRAINING_PROCESS = subprocess.Popen(
        [sys.executable, str(train_script), "--data", str(data_yaml)],
        cwd=str(PROJECT_ROOT),
    )
    return {"status": "success", "message": "Training process started."}


@app.post("/api/v1/render_payload")
async def render_payload(request: Request, file: UploadFile = File(...)):
    validate_filename(file.filename)
    image = decode_image(await read_upload_limited(file))
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise HTTPException(status_code=500, detail="Image conversion failed.")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@app.get("/api/v1/stats")
async def get_stats():
    train_dir = PROJECT_ROOT / "01_DATA" / "processed_tiles" / "train"
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"
    total_images = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
    total_annotations = 0
    if lbl_dir.exists():
        for path in lbl_dir.glob("*.txt"):
            with path.open("r", encoding="utf-8") as handle:
                total_annotations += sum(1 for line in handle if line.strip())
    return {
        "images_annotated": total_images,
        "afb_instances": total_annotations,
        "model_deployed": find_latest_weights() is not None,
        "training_running": TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None,
    }


@app.post("/api/v1/export_report")
async def export_report(data: dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, text="TB-AFB RESEARCH SCREENING REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", size=10)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 8, text=f"Generated: {generated}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    rows = [
        ("Analysis target", data.get("filename", "Unknown")),
        ("Candidate summary", data.get("grade", "Not available")),
        ("Candidate detections", data.get("count", 0)),
        ("Hardware backend", data.get("hardware", "Unknown")),
        ("Reviewer", data.get("pathologist_name", "Not specified")),
    ]
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, text="SUMMARY", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for label, value in rows:
        pdf.cell(0, 7, text=f"{label}: {safe_pdf_text(value)}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        5,
        text=(
            "Research-use-only software. Candidate detections and heuristic outputs are not a validated "
            "diagnosis and must not be used as the sole basis for clinical decisions."
        ),
    )
    return Response(
        content=bytes(pdf.output()),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=TB_AFB_Research_Report.pdf"},
    )


@app.get("/api/v1/wsi/info/{wsi_id}")
async def get_wsi_info(wsi_id: str):
    slide, dz = _deepzoom_for(wsi_id)
    return {
        "width": int(slide.dimensions[0]),
        "height": int(slide.dimensions[1]),
        "tile_size": int(dz.tile_size),
        "tile_overlap": int(dz.overlap),
        "levels": int(dz.level_count),
    }


@app.get("/api/v1/wsi/tile/{wsi_id}/{z}/{x}/{y}")
async def get_wsi_tile(wsi_id: str, z: int, x: int, y: int):
    _, dz = _deepzoom_for(wsi_id)
    try:
        tile = dz.get_tile(z, (x, y))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid tile coordinates.") from exc
    buffer = BytesIO()
    tile.save(buffer, format="JPEG")
    return Response(content=buffer.getvalue(), media_type="image/jpeg")
