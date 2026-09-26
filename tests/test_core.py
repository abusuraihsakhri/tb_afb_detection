from pathlib import Path

import pytest

from tb_afb.inference.who_grader import WHOGrader
from tb_afb.utils.paths import resolve_within


def test_resolve_within_allows_child(tmp_path: Path):
    child = tmp_path / "data" / "slide.svs"
    child.parent.mkdir()
    child.write_bytes(b"x")
    assert resolve_within(tmp_path, "data/slide.svs", must_exist=True) == child.resolve()


def test_resolve_within_rejects_sibling_prefix(tmp_path: Path):
    base = tmp_path / "data"
    sibling = tmp_path / "data2" / "slide.svs"
    base.mkdir()
    sibling.parent.mkdir()
    sibling.write_bytes(b"x")
    with pytest.raises(PermissionError):
        resolve_within(base, sibling, must_exist=True)


@pytest.mark.parametrize(
    ("count", "fields", "grade"),
    [
        (0, 100, "Negative"),
        (1, 100, "Scanty"),
        (9, 100, "Scanty"),
        (10, 100, "1+"),
        (99, 100, "1+"),
        (50, 50, "2+"),
        (500, 50, "2+"),
        (221, 20, "3+"),
    ],
)
def test_who_grading_thresholds(count, fields, grade):
    assert WHOGrader().calculate_grade(count, fields)["grade"] == grade


def test_who_grading_requires_sufficient_fields():
    result = WHOGrader().calculate_grade(2, 20)
    assert result["grade"] == "Not reportable"
    assert result["reportable"] is False


def test_who_grading_rejects_invalid_values():
    with pytest.raises(ValueError):
        WHOGrader().calculate_grade(-1, 100)
    with pytest.raises(ValueError):
        WHOGrader().calculate_grade(1, 0)


from tb_afb.data.stain_normalizer import MacenkoNormalizer
from tb_afb.inference.postprocessor import DetectionPostprocessor


def test_postprocessor_excludes_non_afb_classes():
    processor = DetectionPostprocessor(min_confidence=0.1)
    detections = [
        {"bbox": [10, 10, 4, 12], "confidence": 0.9, "class_id": 0},
        {"bbox": [20, 20, 4, 12], "confidence": 0.9, "class_id": 3},
        {"bbox": [30, 30, 4, 12], "confidence": 0.9, "class_id": 4},
    ]
    result = processor.filter(detections)
    assert [item["class_id"] for item in result] == [0]


def test_macenko_normalizer_requires_fit():
    normalizer = MacenkoNormalizer()
    image = np.full((8, 8, 3), 128, dtype=np.uint8)
    with pytest.raises(RuntimeError):
        normalizer.transform(image)


def test_macenko_normalizer_preserves_shape_and_dtype():
    rng = np.random.default_rng(42)
    reference = rng.integers(20, 235, size=(64, 64, 3), dtype=np.uint8)
    source = rng.integers(20, 235, size=(64, 64, 3), dtype=np.uint8)
    normalizer = MacenkoNormalizer(reference)
    normalized = normalizer.transform(source)
    assert normalized.shape == source.shape
    assert normalized.dtype == np.uint8
