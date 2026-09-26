import math
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def validate_yolo_label_line(line: str, num_classes: int = 5) -> str | None:
    """Validate one normalized YOLO label line.

    Returns None when valid, otherwise a short issue code.
    """
    parts = line.split()
    if len(parts) != 5:
        return "malformed"

    try:
        class_value = float(parts[0])
        x, y, width, height = (float(value) for value in parts[1:])
    except ValueError:
        return "malformed"

    values = (class_value, x, y, width, height)
    if not all(math.isfinite(value) for value in values):
        return "malformed"

    if not class_value.is_integer():
        return "invalid_class"
    class_id = int(class_value)
    if not 0 <= class_id < num_classes:
        return "invalid_class"

    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return "out_of_bounds"
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return "invalid_size"
    if (
        x - width / 2.0 < 0.0
        or x + width / 2.0 > 1.0
        or y - height / 2.0 < 0.0
        or y + height / 2.0 > 1.0
    ):
        return "out_of_bounds"

    return None


def inspect_yolo_split(split_root: Path, num_classes: int = 5) -> dict[str, Any]:
    """Inspect one YOLO split without decoding image pixels."""
    split_root = Path(split_root)
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    structure_ok = image_dir.is_dir() and label_dir.is_dir()

    result: dict[str, Any] = {
        "structure_ok": structure_ok,
        "images": 0,
        "labels": 0,
        "orphaned_images": 0,
        "orphaned_labels": 0,
        "zero_byte_images": 0,
        "total_boxes": 0,
        "malformed": 0,
        "invalid_class": 0,
        "invalid_size": 0,
        "out_of_bounds": 0,
    }
    if not structure_ok:
        return result

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    labels = [path for path in label_dir.glob("*.txt") if path.is_file()]

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    result["images"] = len(images)
    result["labels"] = len(labels)
    result["orphaned_images"] = len(image_stems - label_stems)
    result["orphaned_labels"] = len(label_stems - image_stems)
    result["zero_byte_images"] = sum(path.stat().st_size == 0 for path in images)

    for label_path in labels:
        with label_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                result["total_boxes"] += 1
                issue = validate_yolo_label_line(line, num_classes=num_classes)
                if issue is not None:
                    result[issue] += 1

    return result


def split_is_valid(result: dict[str, Any]) -> bool:
    """Return whether a split passes structural and label-level checks."""
    return bool(
        result["structure_ok"]
        and result["orphaned_images"] == 0
        and result["orphaned_labels"] == 0
        and result["zero_byte_images"] == 0
        and result["malformed"] == 0
        and result["invalid_class"] == 0
        and result["invalid_size"] == 0
        and result["out_of_bounds"] == 0
    )
