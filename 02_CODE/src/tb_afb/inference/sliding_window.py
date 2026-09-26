import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .postprocessor import DetectionPostprocessor
from ..models.yolo_detector import YOLOAFBDetector

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    openslide = None
    OPENSLIDE_AVAILABLE = False


class SlidingWindowInference:
    """Run tiled inference across standard images or OpenSlide-compatible WSIs."""

    WSI_EXTENSIONS = {".svs", ".ndpi", ".vms", ".vmu", ".scn", ".bif", ".mrxs"}

    def __init__(
        self,
        model: YOLOAFBDetector,
        tile_size: int = 512,
        overlap: int = 128,
        batch_size: int = 16,
        confidence_threshold: float = 0.25,
    ):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must satisfy 0 <= overlap < tile_size.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.postprocessor = DetectionPostprocessor(min_confidence=confidence_threshold)

    @staticmethod
    def _axis_positions(length: int, tile_size: int, step: int) -> List[int]:
        if length <= tile_size:
            return [0]
        positions = list(range(0, length - tile_size + 1, step))
        last = length - tile_size
        if positions[-1] != last:
            positions.append(last)
        return positions

    @staticmethod
    def _is_tissue(img: np.ndarray) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return bool(np.mean(gray) < 235)

    def process_slide(self, wsi_path: Path) -> Dict[str, Any]:
        start_time = time.time()
        path = Path(wsi_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing valid image payload: {path.name}")

        suffix = path.suffix.lower()
        if suffix in self.WSI_EXTENSIONS:
            if not OPENSLIDE_AVAILABLE:
                raise RuntimeError("OpenSlide is required for this WSI format.")
            detections, tiles_processed = self._process_openslide(path)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image: {path.name}")
            detections, tiles_processed = self._process_standard_image(image)

        final_detections = self.postprocessor.filter(detections)
        class_counts: Dict[int, int] = {}
        for det in final_detections:
            class_id = int(det.get("class_id", -1))
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        return {
            "total_detections": len(final_detections),
            "detections": final_detections,
            "class_counts": class_counts,
            "processing_time": time.time() - start_time,
            "tiles_processed": tiles_processed,
        }

    def _process_openslide(self, path: Path) -> Tuple[List[Dict], int]:
        detections: List[Dict] = []
        tiles_processed = 0
        slide = openslide.OpenSlide(str(path))
        try:
            width, height = slide.dimensions
            step = self.tile_size - self.overlap
            xs = self._axis_positions(width, self.tile_size, step)
            ys = self._axis_positions(height, self.tile_size, step)

            for y in ys:
                for x in xs:
                    region = slide.read_region((x, y), 0, (self.tile_size, self.tile_size))
                    img = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGBA2BGR)
                    if not self._is_tissue(img):
                        continue
                    tile_detections = self.model.predict(img, conf_threshold=self.confidence_threshold)
                    for det in tile_detections:
                        det["bbox"] = [
                            det["bbox"][0] + x,
                            det["bbox"][1] + y,
                            det["bbox"][2],
                            det["bbox"][3],
                        ]
                    detections.extend(tile_detections)
                    tiles_processed += 1
        finally:
            slide.close()
        return detections, tiles_processed

    def _process_standard_image(self, image: np.ndarray) -> Tuple[List[Dict], int]:
        height, width = image.shape[:2]
        step = self.tile_size - self.overlap
        xs = self._axis_positions(width, self.tile_size, step)
        ys = self._axis_positions(height, self.tile_size, step)

        detections: List[Dict] = []
        tiles_processed = 0
        for y in ys:
            for x in xs:
                tile = image[y : min(y + self.tile_size, height), x : min(x + self.tile_size, width)]
                if tile.size == 0 or not self._is_tissue(tile):
                    continue
                tile_detections = self.model.predict(tile, conf_threshold=self.confidence_threshold)
                for det in tile_detections:
                    det["bbox"] = [
                        det["bbox"][0] + x,
                        det["bbox"][1] + y,
                        det["bbox"][2],
                        det["bbox"][3],
                    ]
                detections.extend(tile_detections)
                tiles_processed += 1
        return detections, tiles_processed
