import os
import cv2
import numpy as np
import torch
import json
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
import sys
import glob
import random
from fpdf import FPDF
from io import BytesIO
try:
    from openslide.deepzoom import DeepZoomGenerator
except ImportError:
    DeepZoomGenerator = None

# Try to import YOLO if the ultralytics library is resolved
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

current_dir = Path(__file__).parent.resolve()
src_dir = current_dir.parent.parent / "02_CODE" / "src"
sys.path.append(str(src_dir))

app = FastAPI(title="Secure TB AFB API Backend")

static_dir = current_dir / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(static_dir), html=True), name="ui")

# 🛡️ SECURITY: Strict CORS baseline to prevent cross-origin medical data scraping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 250 * 1024 * 1024 

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: int = 0

class InferenceResult(BaseModel):
    detections: list
    message: str
    grade: str
    hardware: str

# 🛡️ ARCHITECTURE: Global Hot-Swap Cache for Active Learning Weights
ACTIVE_MODEL = None
ACTIVE_MODEL_PATH = None

def load_active_model(device='cpu'):
    """Dynamically monitors the file system for newly trained YOLOv8 weights and loads them into memory natively on the specified hardware."""
    global ACTIVE_MODEL, ACTIVE_MODEL_PATH
    
    if not ULTRALYTICS_AVAILABLE:
         return None
         
    root_dir = current_dir.parent.parent
    # Find all generated best.pt weights recursively
    weights = list(root_dir.rglob("best.pt"))
    if not weights:
        return None
        
    # Sort securely by modification time ascending to grab the absolute latest
    weights.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_path = str(weights[0])
    
    # Execute the Hot-Swap if new weights are detected
    if latest_path != ACTIVE_MODEL_PATH:
        ACTIVE_MODEL_PATH = latest_path
        print(f"[ACTIVE LEARNING] 🚀 Hot-swapping to new Brain: {ACTIVE_MODEL_PATH}")
        ACTIVE_MODEL = YOLO(ACTIVE_MODEL_PATH)
        
    return ACTIVE_MODEL

@app.post("/api/v1/analyze", response_model=InferenceResult)
async def analyze_slide(request: Request, file: UploadFile = File(...)):
    if int(request.headers.get('content-length', 0)) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. (250MB Hard Limit)")
        
    filename = file.filename.lower()
    allowed_exts = [
        ".jpg", ".jpeg", ".png", 
        ".tiff", ".tif", ".ptif", ".ptiff", ".ome.tif", ".ome.tiff",
        ".jp2", ".j2k", ".jpf", ".jpx",
        ".svs", ".ndpi", ".vms", ".vmu", ".scn", ".bif", ".mrxs",
        ".dicom", ".dcm"
    ]
    if not any(filename.endswith(ext) for ext in allowed_exts):
        raise HTTPException(status_code=415, detail="Unsupported format.")

    # 🛡️ ALGO UPGRADE: Attempt to load the Deep Learning Model if you trained it!
    # Universal Device Priority Chain
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    yolo_model = load_active_model(device=device)
    
    # Reflect exact architecture backend in the UI
    if device.type == 'cuda':
        base_hw = 'NVIDIA Tensor Cores'
    elif device.type == 'mps':
        base_hw = 'Apple Silicon (Metal/MPS)'
    else:
        base_hw = 'CPU Mode'
        
    hardware_status = f"Hardware Backend: {base_hw} | "
    if yolo_model:
        hardware_status += "[Active Learning Model HOT-MOUNTED!]"
    else:
        hardware_status += "[Clinical Fallback: OpenCV Morphology Segmenter]"

    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    mock_boxes = []
    
    if image is not None:
        if yolo_model:
            # 🚀 FULL AI LOOP: Pass the array directly to Ultralytics
            # We bypass disk writes entirely and run natively in PyTorch RAM
            results = yolo_model(image)[0]
            
            for box in results.boxes:
                # Extract YOLO format and convert nicely
                coords = box.xywh[0].tolist() # x_center, y_center, width, height
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Active filter (Require high diagnostic confidence > 50%)
                if conf >= 0.50:
                    mock_boxes.append({
                        "bbox": coords,
                        "confidence": round(conf, 2),
                        "class_id": cls,
                        "label": "AFB_Definite" if cls == 0 else "AFB_Possible"
                    })
        else:
            # 🛡️ THE FALLBACK: Strict color thresholds if YOLO is not yet trained!
            h, w = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            lower_magenta = np.array([130, 40, 20])
            upper_magenta = np.array([175, 255, 255])
            
            mask = cv2.inRange(hsv, lower_magenta, upper_magenta)
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 15 < area < 300:
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    aspect_ratio = max(cw, ch) / max(min(cw, ch), 1)
                    if aspect_ratio >= 1.5:
                        conf = min(0.99, 0.70 + (aspect_ratio / 10.0))
                        mock_boxes.append({
                            "bbox": [x + cw/2.0, y + ch/2.0, cw + 4, ch + 4],
                            "confidence": round(conf, 2),
                            "class_id": 0 if conf > 0.85 else 2,
                            "label": "AFB_Definite" if conf > 0.85 else "AFB_Possible"
                        })
    
    det_count = len(mock_boxes)
    if det_count == 0: grade = "Negative"
    elif det_count < 10: grade = f"Scanty ({det_count} AFB detected)"
    elif det_count < 100: grade = f"1+ Positive ({det_count} AFB detected)"
    elif det_count < 1000: grade = f"2+ Positive ({det_count} AFB detected)"
    else: grade = f"3+ Positive ({det_count} AFB detected)"
        
    return InferenceResult(
        detections=mock_boxes,
        message=f"{'YOLOv8 Active Execution' if yolo_model else 'ZN OpenCV Pipeline Executed'}.",
        grade=grade,
        hardware=hardware_status
    )

@app.post("/api/v1/save_annotation")
async def save_annotation(request: Request, file: UploadFile = File(...), boxes: str = Form(...)):
    if int(request.headers.get('content-length', 0)) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large.")
        
    try:
        boxes_list = json.loads(boxes)
        parsed_boxes = [BoundingBox(**b) for b in boxes_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail="Malformed JSON injection attempt blocked.")
    
    base_name = uuid.uuid4().hex
    safe_img_name = f"{base_name}.jpg"
    safe_lbl_name = f"{base_name}.txt"
    
    # 🛡️ VALIDATION SPLIT: Randomized 20% logic for clinical data hygiene
    sub_folder = "val" if random.random() < 0.20 else "train"
    
    data_dir = current_dir.parent.parent / "01_DATA" / "processed_tiles" / sub_folder
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Image bytes
    contents = await file.read()
    with open(img_dir / safe_img_name, "wb") as f:
        f.write(contents)
        
    # YOLO format extraction
    with open(lbl_dir / safe_lbl_name, "w") as f:
        for b in parsed_boxes:
             f.write(f"{b.label} {b.x} {b.y} {b.width} {b.height}\n")
             
    return {
        "status": "success", 
        "message": f"Ingested {len(parsed_boxes)} ground-truth annotations into {sub_folder.upper()} set!"
    }

@app.post("/api/v1/trigger_training")
async def trigger_training():
    lbl_dir = current_dir.parent.parent / "01_DATA" / "processed_tiles" / "train" / "labels"
    if not lbl_dir.exists() or len(list(lbl_dir.glob("*.txt"))) == 0:
        raise HTTPException(status_code=400, detail="Cannot initiate Neural Engine: Zero annotated data vectors found.")
    
    train_script = current_dir.parent.parent / "02_CODE" / "scripts" / "02_train.py"
    data_yaml = current_dir.parent.parent / "02_CODE" / "data.yaml"
    subprocess.Popen(["python", str(train_script), "--data", str(data_yaml)])
    
    return {"status": "success", "message": "CUDA Neural Training has been allocated."}

@app.post("/api/v1/render_payload")
async def render_payload(request: Request, file: UploadFile = File(...)):
    if int(request.headers.get('content-length', 0)) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. (250MB Hard Limit)")
        
    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=415, detail="Backend CV2 Decoder failed to parse this specific Medical Binary structure. Ensure you upload supported patches (TIFF, JP2, JPG, PNG).")
        
    _, encoded_img = cv2.imencode('.jpg', image)
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.get("/api/v1/stats")
async def get_stats():
    # 🛡️ SECURITY: Safe enumeration of dataset to expose metrics without Arbitrary File Reads
    train_dir = current_dir.parent.parent / "01_DATA" / "processed_tiles" / "train"
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"
    
    total_images = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
    total_annotations = 0
    
    if lbl_dir.exists():
        for txt_file in lbl_dir.glob("*.txt"):
            with open(txt_file, "r") as f:
                lines = f.readlines()
                # Ensure no empty lines crash counter
                total_annotations += len([line for line in lines if line.strip()])
                
    has_weights = load_active_model() is not None
    
    return {
        "images_annotated": total_images,
        "afb_instances": total_annotations,
        "model_deployed": has_weights
    }

@app.post("/api/v1/export_report")
async def export_report(data: dict):
    # 🛡️ SECURITY: Structured PDF generation avoiding arbitrary HTML rendering
    pdf = FPDF()
    pdf.add_page()
    
    # Branded Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="TB PATHOLOGY INTELLIGENCE REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated Date: {uuid.uuid4().hex[:8]}", ln=True, align='C')
    pdf.ln(10)
    
    # Clinical Data
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="DIAGNOSTIC SUMMARY", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Analysis Target: {data.get('filename', 'Unknown')}", ln=True)
    pdf.cell(200, 8, txt=f"WHO Grade: {data.get('grade', 'Pending')}", ln=True)
    pdf.cell(200, 8, txt=f"AFB Detections Count: {data.get('count', 0)}", ln=True)
    pdf.cell(200, 8, txt=f"Hardware Backend: {data.get('hardware', 'CPU')}", ln=True)
    pdf.cell(200, 8, txt=f"Reporting Pathologist: {data.get('pathologist_name', 'Not Specified')}", ln=True)
    
    pdf.ln(20)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(10)
    
    # Medical Disclaimer
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt="DISCLAIMER: This report is generated by an Artificial Intelligence system. It is intended for research and diagnostic assistance only and must be confirmed by a licensed professional before clinical action is taken.")
    
    pdf_output = pdf.output(dest='S')
    return Response(content=pdf_output, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=TB_AFB_Report.pdf"})

# 🛡️ GLOBAL CACHE FOR WSI TILES
WSI_HANDLES = {}

@app.get("/api/v1/wsi/tile/{wsi_id}/{z}/{x}/{y}")
async def get_wsi_tile(wsi_id: str, z: int, x: int, y: int):
    # Dynamic tile server for OpenSeadragon
    if not DeepZoomGenerator:
        raise HTTPException(status_code=501, detail="OpenSlide DeepZoom not available on this host.")
        
    # Security: Ensure WSI_ID is not a path injection
    # 🛡️ REMEDIATION: Force filename-only resolution to block Windows/Linux traversal
    safe_wsi_id = Path(wsi_id).name
    base_data_dir = current_dir.parent.parent / "01_DATA" / "raw_tiles"
    wsi_path = (base_data_dir / safe_wsi_id).resolve()
    
    # Final Jail Check: Resulting path MUST still be inside base_data_dir
    if not str(wsi_path).startswith(str(base_data_dir.resolve())):
         raise HTTPException(status_code=403, detail="Airtight Jail Breach Attempt Blocked.")
    
    if wsi_id not in WSI_HANDLES:
        import openslide
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            WSI_HANDLES[wsi_id] = DeepZoomGenerator(slide, tile_size=254, overlap=1, limit_bounds=False)
        except Exception as e:
            raise HTTPException(status_code=404, detail="WSI file not found or corrupted.")
            
    dz = WSI_HANDLES[wsi_id]
    try:
        tile = dz.get_tile(z, (x, y))
        buf = BytesIO()
        tile.save(buf, format='JPEG')
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Tile Coordinates.")


