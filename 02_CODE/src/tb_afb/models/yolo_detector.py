from pathlib import Path
from typing import Any, Dict, List


class YOLOAFBDetector:
    """Small wrapper around Ultralytics YOLO used by training and inference."""

    ANCHORS = [
        [4, 16],
        [6, 24],
        [8, 32],
        [4, 12],
        [3, 20],
    ]

    def __init__(self, model_size: str = "m", num_classes: int = 5, pretrained: bool = True):
        self.model_size = model_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None

    @staticmethod
    def _yolo_class():
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed.") from exc
        return YOLO

    def build_model(self) -> Any:
        if self.model_size not in {"n", "s", "m", "l", "x"}:
            raise ValueError(f"Invalid model size identifier: {self.model_size}")
        YOLO = self._yolo_class()
        self.model = YOLO(f"yolov8{self.model_size}.pt")
        return self.model

    def load_model(self, checkpoint_path: Path) -> Any:
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if checkpoint.suffix.lower() != ".pt":
            raise ValueError("YOLO checkpoint must use the .pt extension.")
        YOLO = self._yolo_class()
        self.model = YOLO(str(checkpoint))
        return self.model

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
        batch_size: int = 16,
        device: str = None,
        **kwargs,
    ) -> Path:
        if self.model is None:
            raise RuntimeError("Build or load a model before training.")

        if device is None:
            import torch

            if torch.cuda.is_available():
                device = "0"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        data_yaml = Path(data_yaml).resolve()
        if not data_yaml.is_file():
            raise FileNotFoundError("Data YAML missing.")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")

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
            mosaic=1.0,
            **kwargs,
        )

        trainer = getattr(self.model, "trainer", None)
        best = getattr(trainer, "best", None)
        if best:
            return Path(best)
        return Path("runs/detect/train/weights/best.pt")

    def predict(
        self,
        image: "np.ndarray",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ) -> List[Dict]:
        if self.model is None:
            raise RuntimeError("No model is loaded.")

        if image.size > 50_000_000:
            raise ValueError("Input image array exceeds safe inference limits.")

        conf = max(0.01, min(1.0, float(conf_threshold)))
        iou = max(0.01, min(1.0, float(iou_threshold)))
        max_det = max(1, min(5000, int(max_det)))

        results = self.model(image, conf=conf, iou=iou, max_det=max_det, verbose=False)
        detections = []
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
