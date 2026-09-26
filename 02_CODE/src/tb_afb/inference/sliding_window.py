import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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

WSI_EXTENSIONS = {".svs", ".ndpi", ".vms", ".vmu", ".scn", ".bif", ".mrxs"}


class SlidingWindowInference:
    """Run tiled inference on OpenSlide WSIs or standard raster images."""

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
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")

        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.postprocessor = DetectionPostprocessor(
            min_confidence=confidence_threshold,
        )

    @property
    def step(self) -> int:
        return self.tile_size - self.overlap

    def _axis_positions(self, length: int) -> List[int]:
        if length <= self.tile_size:
            return [0]
        positions = list(range(0, length - self.tile_size + 1, self.step))
        final = length - self.tile_size
        if positions[-1] != final:
            positions.append(final)
        return positions

    def _coords(self, width: int, height: int) -> List[Tuple[int, int]]:
        return [
            (x, y)
            for y in self._axis_positions(height)
            for x in self._axis_positions(width)
        ]

    @staticmethod
    def _contains_tissue(image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray)) < 235.0

    def _translate(
        self,
        detections: Iterable[Dict],
        offset_x: int,
        offset_y: int,
    ) -> List[Dict]:
        translated = []
        for detection in detections:
            item = dict(detection)
            bbox = list(item.get("bbox", []))
            if len(bbox) != 4:
                continue
            bbox[0] = float(bbox[0]) + offset_x
            bbox[1] = float(bbox[1]) + offset_y
            item["bbox"] = bbox
            translated.append(item)
        return translated

    def _predict_tile(self, image: np.ndarray, x: int, y: int) -> List[Dict]:
        detections = self.model.predict(
            image,
            conf_threshold=self.confidence_threshold,
        )
        return self._translate(detections, x, y)

    def _process_raster(
        self,
        image: np.ndarray,
    ) -> Tuple[List[Dict], int, None]:
        height, width = image.shape[:2]
        detections: List[Dict] = []
        processed = 0

        for x, y in self._coords(width, height):
            tile = image[
                y : min(y + self.tile_size, height),
                x : min(x + self.tile_size, width),
            ]
            if tile.size == 0 or not self._contains_tissue(tile):
                continue
            detections.extend(self._predict_tile(tile, x, y))
            processed += 1
        return detections, processed, None

    @staticmethod
    def _slide_mpp(slide) -> float | None:
        values = []
        for key in (openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y):
            raw = slide.properties.get(key)
            if raw is None:
                return None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(value) or value <= 0:
                return None
            values.append(value)
        return float(sum(values) / len(values))

    def _process_wsi(
        self,
        path: Path,
    ) -> Tuple[List[Dict], int, float | None]:
        if not OPENSLIDE_AVAILABLE:
            raise RuntimeError("OpenSlide is required for this WSI format.")

        slide = openslide.OpenSlide(str(path))
        detections: List[Dict] = []
        processed = 0
        try:
            width, height = slide.dimensions
            pixel_size_microns = self._slide_mpp(slide)
            coords = self._coords(width, height)
            for start in range(0, len(coords), self.batch_size):
                for x, y in coords[start : start + self.batch_size]:
                    region = slide.read_region(
                        (x, y),
                        0,
                        (
                            min(self.tile_size, width - x),
                            min(self.tile_size, height - y),
                        ),
                    )
                    tile = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGBA2BGR)
                    if not self._contains_tissue(tile):
                        continue
                    detections.extend(self._predict_tile(tile, x, y))
                    processed += 1
        finally:
            slide.close()
        return detections, processed, pixel_size_microns

    def process_slide(self, wsi_path: Path) -> Dict[str, Any]:
        start_time = time.time()
        path = Path(wsi_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        if path.suffix.lower() in WSI_EXTENSIONS:
            detections, processed, pixel_size_microns = self._process_wsi(path)
        else:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"OpenCV could not decode image: {path}")
            detections, processed, pixel_size_microns = self._process_raster(image)

        final_detections = self.postprocessor.filter(
            detections,
            pixel_size_microns=pixel_size_microns,
        )
        return {
            "total_detections": len(final_detections),
            "detections": final_detections,
            "processing_time": time.time() - start_time,
            "tiles_processed": processed,
            "pixel_size_microns": pixel_size_microns,
        }
