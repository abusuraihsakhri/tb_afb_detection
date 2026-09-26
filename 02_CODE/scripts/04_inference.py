#!/usr/bin/env python3
"""Run local AFB-candidate inference on a trusted image or WSI."""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tb_afb.inference.sliding_window import SlidingWindowInference
from tb_afb.inference.who_grader import WHOGrader
from tb_afb.models.yolo_detector import YOLOAFBDetector


def secure_file_resolution(base_dir: Path, user_input_path: str) -> Path:
    base_dir = base_dir.resolve()
    requested = Path(user_input_path).expanduser()
    if not requested.is_absolute():
        requested = base_dir / requested
    requested = requested.resolve()

    try:
        requested.relative_to(base_dir)
    except ValueError as exc:
        raise PermissionError(f"Input must remain inside {base_dir}") from exc

    if not requested.is_file():
        raise FileNotFoundError(f"Input image not found: {requested}")
    return requested


def load_trusted_model(checkpoint_path: Path) -> YOLOAFBDetector:
    detector = YOLOAFBDetector(model_size="n")
    detector.load_weights(checkpoint_path)
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description="TB-AFB research inference")
    parser.add_argument("--model", type=Path, required=True, help="Trusted local .pt checkpoint")
    parser.add_argument("--wsi", required=True, help="Path relative to 01_DATA (for example raw_wsi/slide.svs)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--fields-examined",
        type=int,
        default=None,
        help="Actual microscope fields examined; required to report WHO/IUATLD ZN smear grade",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    safe_root = project_root / "01_DATA"

    if not 0 < args.conf <= 1:
        parser.error("--conf must be > 0 and <= 1")
    if args.fields_examined is not None and args.fields_examined <= 0:
        parser.error("--fields-examined must be positive")

    try:
        image_path = secure_file_resolution(safe_root, args.wsi)
        detector = load_trusted_model(args.model)
        engine = SlidingWindowInference(model=detector, confidence_threshold=args.conf, batch_size=16)
        results = engine.process_slide(image_path)

        print(f"AFB candidate detections: {results['total_detections']}")
        print(f"Tissue tiles processed: {results['tiles_processed']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")

        if args.fields_examined is not None:
            report = WHOGrader().calculate_grade(results["total_detections"], args.fields_examined)
            print(f"WHO/IUATLD ZN smear grade: {report['report_string']}")
        else:
            print("WHO/IUATLD ZN smear grade: not calculated (no microscope field count supplied)")
    except Exception as exc:
        print(f"[INFERENCE ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
