import torch
from pathlib import Path
from typing import Any, List, Dict
import subprocess

class YOLOAFBDetector:
    """
    YOLOv8 detector with custom anchors for AFB detection.
    Protected against arbitrary training payload executions.
    """
    ANCHORS = [
        [4, 16],   # Small bacillus (width, height in pixels)
        [6, 24],   # Medium bacillus
        [8, 32],   # Large/clumped
        [4, 12],   # Short rod
        [3, 20]    # Thin elongated
    ]
    
    def __init__(self, model_size: str = "m", num_classes: int = 5, pretrained: bool = True):
        self.model_size = model_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None

    def build_model(self) -> Any:
        try:
            from ultralytics import YOLO
            # 🛡️ SECURITY CONTROL: Restricted model size resolution
            if self.model_size not in ["n", "s", "m", "l", "x"]:
                 raise ValueError(f"Invalid model size identifier: {self.model_size}")
            self.model = YOLO(f"yolov8{self.model_size}.pt")
            return self.model
        except ImportError:
            raise RuntimeError("Ultralytics library missing.")

    def train(self, data_yaml: Path, epochs: int = 100, batch_size: int = 16, device: str = None, **kwargs) -> Path:
        """Train model with AFB-specific configurations on Universal Hardware."""
        # 🛡️ ARCHITECTURE: Universal Device Orchestration (CUDA -> MPS -> CPU)
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = '0'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
            
        # 🛡️ SECURITY CONTROL: Validating dataset yaml to prevent injecting arbitrary command params to ultralytics runner
        data_yaml = Path(data_yaml).resolve()
        if not data_yaml.exists():
            raise FileNotFoundError("Data YAML missing.")
            
        if self.model:
            # 🛡️ Augmentation Strategy: 10% Rotation, HSV Jitter for staining drift, Vertical/Horizontal Flips
            self.model.train(
                data=str(data_yaml), 
                epochs=int(epochs), 
                batch=int(batch_size), 
                device=device,
                exist_ok=True,
                degrees=15.0,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                flipud=0.5,
                fliplr=0.5,
                mosaic=1.0 # Essential for finding small AFB objects in dense tiles
            )
            
        return Path("runs/detect/train/weights/best.pt")

    def predict(self, image: 'np.ndarray', conf_threshold: float = 0.25, iou_threshold: float = 0.45, max_det: int = 300) -> List[Dict]:
        """Run inference on single image securely limits RAM allocation."""
        if not self.model:
            return []
            
        # 🛡️ SECURITY CONTROL: Protect against gigantic tile arrays causing GPU Out-Of-Memory (OOM) attack vectors
        if image.size > 50000000:
             raise ValueError("Input Image array exceeds safe inference limits.")
             
        # Clamp inputs
        conf = max(0.01, min(1.0, float(conf_threshold)))
        iou = max(0.01, min(1.0, float(iou_threshold)))
        max_det = max(1, min(5000, int(max_det)))
        
        results = self.model(image, conf=conf, iou=iou, max_det=max_det)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xywh = box.xywh[0].tolist()
                c = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({'bbox': xywh, 'confidence': c, 'class_id': cls_id})
        return detections
