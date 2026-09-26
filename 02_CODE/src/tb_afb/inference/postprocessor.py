import numpy as np
from typing import Dict, Iterable, List, Set


class DetectionPostprocessor:
    """Filter candidate AFB detections and perform global NMS."""

    MIN_ASPECT_RATIO = 2.0
    MAX_ASPECT_RATIO = 10.0
    MIN_AREA_SQUARE_MICRONS = 2.0
    MAX_AREA_SQUARE_MICRONS = 20.0

    def __init__(
        self,
        min_confidence: float = 0.3,
        nms_iou_threshold: float = 0.5,
        allowed_class_ids: Iterable[int] = (0, 1, 2),
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")
        if not 0.0 <= nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1].")
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.allowed_class_ids: Set[int] = {int(value) for value in allowed_class_ids}

    def filter(\n        self,\n        detections: List[Dict],\n        pixel_size_microns: float | None = None,\n    ) -> List[Dict]:\n        if pixel_size_microns is not None and pixel_size_microns <= 0:\n            raise ValueError("pixel_size_microns must be positive when provided.")\n\n        filtered = []\n        for detection in detections:
            if int(detection.get("class_id", -1)) not in self.allowed_class_ids:
                continue

            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                cx, cy, width, height = (float(value) for value in bbox)
                confidence = float(detection.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue

            if not np.isfinite([cx, cy, width, height, confidence]).all():
                continue
            if width <= 0 or height <= 0 or confidence < self.min_confidence:
                continue

            aspect_ratio = max(width, height) / min(width, height)
            if not self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                continue

            if pixel_size_microns is not None:
                area_square_microns = (
                    width * pixel_size_microns
                ) * (
                    height * pixel_size_microns
                )
                if not (
                    self.MIN_AREA_SQUARE_MICRONS
                    <= area_square_microns
                    <= self.MAX_AREA_SQUARE_MICRONS
                ):
                    continue

            filtered.append(detection)

        if not filtered:
            return []

        boxes = np.asarray([item["bbox"] for item in filtered], dtype=np.float64)
        x1 = boxes[:, 0] - boxes[:, 2] / 2.0
        y1 = boxes[:, 1] - boxes[:, 3] / 2.0
        x2 = boxes[:, 0] + boxes[:, 2] / 2.0
        y2 = boxes[:, 1] + boxes[:, 3] / 2.0
        scores = np.asarray([item["confidence"] for item in filtered], dtype=np.float64)
        keep = self.apply_nms(
            np.stack([x1, y1, x2, y2], axis=1),
            scores,
            self.nms_iou_threshold,
        )
        return [filtered[index] for index in keep]

    @staticmethod
    def apply_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.5,
    ) -> List[int]:
        if len(boxes) == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(scores) != len(boxes):
            raise ValueError("boxes must be (N, 4) and scores must have length N.")

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size:
            current = int(order[0])
            keep.append(current)
            remaining = order[1:]
            if remaining.size == 0:
                break

            xx1 = np.maximum(x1[current], x1[remaining])
            yy1 = np.maximum(y1[current], y1[remaining])
            xx2 = np.minimum(x2[current], x2[remaining])
            yy2 = np.minimum(y2[current], y2[remaining])
            intersection = (
                np.maximum(0.0, xx2 - xx1)
                * np.maximum(0.0, yy2 - yy1)
            )
            union = areas[current] + areas[remaining] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]

        return keep
