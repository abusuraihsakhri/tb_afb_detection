#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from tb_afb.inference.sliding_window import SlidingWindowInference
from tb_afb.inference.who_grader import WHOGrader
from tb_afb.models.yolo_detector import YOLOAFBDetector
from tb_afb.utils.paths import resolve_within


def load_model(checkpoint_path: Path) -> YOLOAFBDetector:
    detector = YOLOAFBDetector(model_size="n")
    detector.load_model(Path(checkpoint_path))
    return detector


def main():
    parser = argparse.ArgumentParser(description="TB-AFB research inference")
    parser.add_argument("--model", type=Path, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--wsi", required=True, help="Slide path within 01_DATA.")
    parser.add_argument("--conf", type=float, default=0.25, help="Model score threshold.")
    parser.add_argument(
        "--fields-examined",
        type=int,
        default=None,
        help="Optional actual microscopy field count for smear grading; detector tiles are not HPFs.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    data_root = root_dir / "01_DATA"

    try:
        wsi_path = resolve_within(data_root, args.wsi, must_exist=True)
        detector = load_model(args.model.resolve())

        inference_engine = SlidingWindowInference(
            model=detector,
            confidence_threshold=args.conf,
            batch_size=16,
        )
        results = inference_engine.process_slide(wsi_path)

        print("\n" + "-" * 50)
        print("TB-AFB RESEARCH SCREENING SUMMARY")
        print("-" * 50)
        print(f"Candidate detections : {results['total_detections']}")
        print(f"Processing time      : {results['processing_time']:.2f} seconds")
        print(f"Tiles processed      : {results['tiles_processed']}")

        if args.fields_examined is not None:
            clinical_report = WHOGrader().calculate_grade(
                results["total_detections"],
                fields_examined=args.fields_examined,
            )
            print(f"Smear grade          : {clinical_report['report_string']}")
            print(f"Grading basis        : {clinical_report['reason']}")
        else:
            print("Smear grade          : Not calculated (actual examined-field count required)")
        print("-" * 50)
        print("Research-use output; candidate detections require human review.\n")

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
