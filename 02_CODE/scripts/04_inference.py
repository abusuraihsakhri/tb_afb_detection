#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from tb_afb.inference.sliding_window import SlidingWindowInference
from tb_afb.models.yolo_detector import YOLOAFBDetector
from tb_afb.utils.paths import resolve_within


def load_model(checkpoint_path: Path) -> YOLOAFBDetector:
    detector = YOLOAFBDetector(model_size="n")
    detector.load_model(Path(checkpoint_path))
    return detector


def main():
    parser = argparse.ArgumentParser(description="TB-AFB research inference")
    parser.add_argument("--model", type=Path, required=True, help="Path to an AFB-trained .pt checkpoint.")
    parser.add_argument("--wsi", required=True, help="Slide path within 01_DATA.")
    parser.add_argument("--conf", type=float, default=0.25, help="Model score threshold.")
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
        print("Smear grade          : Not calculated")
        print("-" * 50)
        print(
            "Candidate detections are model outputs, not confirmed AFB counts. "
            "WHO/IUATLD smear grading requires an appropriate microscopy field-count "
            "protocol and confirmed counts.\n"
        )

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
