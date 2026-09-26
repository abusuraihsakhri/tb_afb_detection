import numpy as np
from typing import Dict, Iterable, List, Optional, Set


class DetectionPostprocessor:
    """Post-process AFB-candidate detections and suppress overlapping boxes."""

    MIN_ASPECT_RATIO = 2.0
    MAX_ASPECT_RATIO = 10.0
    MIN_AREA_MICRONS2 = 2.0
    MAX_AREA_MICRONS2 = 20.0

    def __init__(
        self,
        min_confidence: float = 0.3,
        nms_iou_threshold: float = 0.5,
        pixel_size_microns: float = 0.25,
        allowed_class_ids: Optional[Iterable[int]] = (0, 1, 2),
    ):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1.")
        if not 0 <= nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold must be between 0 and 1.")
        if pixel_size_microns <= 0:
            raise ValueError("pixel_size_microns must be positive.")

        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.pixel_size_microns = pixel_size_microns
        self.allowed_class_ids: Optional[Set[int]] = (
            None if allowed_class_ids is None else {int(v) for v in allowed_class_ids}
        )

    def filter(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []

        filtered: List[Dict] = []
        for det in detections:
            class_id = int(det.get("class_id", -1))
            if self.allowed_class_ids is not None and class_id not in self.allowed_class_ids:
                continue

            bbox = det.get("bbox", [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            _, _, w, h = map(float, bbox)
            if w <= 0 or h <= 0:
                continue

            aspect_ratio = max(w, h) / min(w, h)
            area_microns2 = (w * self.pixel_size_microns) * (h * self.pixel_size_microns)
            confidence = float(det.get("confidence", 0.0))

            if (
                confidence >= self.min_confidence
                and self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO
                and self.MIN_AREA_MICRONS2 <= area_microns2 <= self.MAX_AREA_MICRONS2
            ):
                filtered.append(det)

        if not filtered:
            return []

        boxes = np.asarray([d["bbox"] for d in filtered], dtype=float)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        scores = np.asarray([d["confidence"] for d in filtered], dtype=float)

        keep = self.apply_nms(
            np.stack([x1, y1, x2, y2], axis=1),
            scores,
            self.nms_iou_threshold,
        )
        return [filtered[i] for i in keep]

    @staticmethod
    def apply_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h
            union = areas[i] + areas[order[1:]] - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

            remaining = np.where(iou <= iou_threshold)[0]
            order = order[remaining + 1]

        return keep
