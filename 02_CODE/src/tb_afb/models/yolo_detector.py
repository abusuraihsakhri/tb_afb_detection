from pathlib import Path
from typing import Any, Dict, List

import torch


class YOLOAFBDetector:
    """Thin wrapper around Ultralytics YOLO for AFB research workflows."""

    VALID_MODEL_SIZES = {"n", "s", "m", "l", "x"}

    def __init__(self, model_size: str = "m", num_classes: int = 5, pretrained: bool = True):
        if model_size not in self.VALID_MODEL_SIZES:
            raise ValueError(f"Invalid model size identifier: {model_size}")
        self.model_size = model_size
        self.num_classes = int(num_classes)
        self.pretrained = bool(pretrained)
        self.model = None

    @staticmethod
    def _yolo_class():
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is required for model operations.") from exc
        return YOLO

    def build_model(self) -> Any:
        YOLO = self._yolo_class()
        model_source = f"yolov8{self.model_size}.pt" if self.pretrained else f"yolov8{self.model_size}.yaml"
        self.model = YOLO(model_source)
        return self.model

    def load_weights(self, checkpoint_path: Path) -> Any:
        """Load a trusted local Ultralytics checkpoint.

        Model checkpoints are executable/deserialization inputs. Do not load checkpoint
        files from untrusted sources merely because they have a ``.pt`` extension.
        """
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        if path.suffix.lower() != ".pt":
            raise ValueError("Ultralytics checkpoints must use the .pt extension.")

        YOLO = self._yolo_class()
        self.model = YOLO(str(path))
        return self.model

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
        batch_size: int = 16,
        device: str | None = None,
    ) -> Path:
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")

        data_yaml = Path(data_yaml).resolve()
        if not data_yaml.is_file():
            raise FileNotFoundError("Data YAML missing.")

        if device is None:
            if torch.cuda.is_available():
                device = "0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if self.model is None:
            self.build_model()

        result = self.model.train(
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
            mosaic=1.0,
        )

        save_dir = Path(getattr(result, "save_dir", "runs/detect/train"))
        return save_dir / "weights" / "best.pt"

    def predict(
        self,
        image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        if getattr(image, "size", 0) > 50_000_000:
            raise ValueError("Input image array exceeds safe inference limits.")

        conf = max(0.01, min(1.0, float(conf_threshold)))
        iou = max(0.01, min(1.0, float(iou_threshold)))
        max_det = max(1, min(5000, int(max_det)))

        results = self.model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
        detections: List[Dict] = []
        for result in results:
            for box in result.boxes:
                detections.append(
                    {
                        "bbox": box.xywh[0].tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    }
                )
        return detections
