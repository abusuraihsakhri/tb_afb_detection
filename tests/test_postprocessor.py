import numpy as np

from tb_afb.inference.postprocessor import DetectionPostprocessor


def test_non_afb_classes_are_excluded():
    processor = DetectionPostprocessor(min_confidence=0.2, pixel_size_microns=0.25)
    detections = [
        {"bbox": [10, 10, 4, 12], "confidence": 0.9, "class_id": 0},
        {"bbox": [30, 10, 4, 12], "confidence": 0.9, "class_id": 3},
    ]
    result = processor.filter(detections)
    assert len(result) == 1
    assert result[0]["class_id"] == 0


def test_nms_suppresses_overlapping_lower_score_box():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]])
    scores = np.array([0.9, 0.8])
    keep = DetectionPostprocessor.apply_nms(boxes, scores, 0.5)
    assert keep == [0]
