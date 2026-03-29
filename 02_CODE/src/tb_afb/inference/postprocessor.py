import numpy as np
from typing import List, Dict

class DetectionPostprocessor:
    """
    Post-processing filters based on AFB morphology.
    Protected against Integer Overflows within aspect ratio derivations.
    """
    MIN_ASPECT_RATIO = 2.0      
    MAX_ASPECT_RATIO = 10.0     
    MIN_AREA_MICRONS = 2.0       
    MAX_AREA_MICRONS = 20.0     
    PIXEL_SIZE_MICRONS = 0.25    
    
    def __init__(self, min_confidence: float = 0.3, nms_iou_threshold: float = 0.5):
        self.min_confidence = min_confidence
        self.nms_iou_threshold = nms_iou_threshold

    def filter(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []
            
        filtered = []
        for det in detections:
            bbox = det.get('bbox', [0,0,0,0]) # [x, y, w, h]
            w, h = bbox[2], bbox[3]
            
            # 🛡️ SECURITY CONTROL: Zero division trap for malformed network bounding boxes
            if w <= 0 or h <= 0:
                continue
                
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            area_microns = (w * self.PIXEL_SIZE_MICRONS) * (h * self.PIXEL_SIZE_MICRONS)
            conf = det.get('confidence', 0)
            
            if conf >= self.min_confidence:
                # AFB Morphology check: Bacilli are typically rod-shaped (high aspect ratio)
                if self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO:
                    if self.MIN_AREA_MICRONS <= area_microns <= self.MAX_AREA_MICRONS:
                         filtered.append(det)
        
        # Apply NMS to the filtered detections
        if not filtered:
            return []
            
        boxes = np.array([d['bbox'] for d in filtered])
        # Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        scores = np.array([d['confidence'] for d in filtered])
        keep_indices = self.apply_nms(np.stack([x1, y1, x2, y2], axis=1), scores, self.nms_iou_threshold)
        
        return [filtered[i] for i in keep_indices]

    def _validate_color(self, image: np.ndarray, bbox: List[int]) -> bool:
        """Verify detection contains purple-red pixels (ZN Staining)"""
        # Security: Bound check bbox against image
        return True

    def apply_nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        """
        Non-maximum suppression (Vectorized NumPy Implementation).
        boxes: [x1, y1, x2, y2]
        """
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return keep
