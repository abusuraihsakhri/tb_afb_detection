#!/usr/bin/env python3
"""YOLO training entry point for the local TB-AFB research dataset."""

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tb_afb.models.yolo_detector import YOLOAFBDetector
from tb_afb.utils.logger import AuditLogger


def secure_yaml_resolution(base_dir: Path, yaml_path: str) -> Path:
    base_dir = base_dir.resolve()
    requested = Path(yaml_path).expanduser()
    if not requested.is_absolute():
        requested = base_dir / requested
    requested = requested.resolve()

    try:
        requested.relative_to(base_dir)
    except ValueError as exc:
        raise PermissionError("Training YAML must remain inside 02_CODE.") from exc

    if requested.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Training configuration must be a YAML file.")
    if not requested.is_file():
        raise FileNotFoundError(f"Training YAML not found: {requested}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO training orchestrator")
    parser.add_argument("--data", required=True, help="Path to data.yaml inside 02_CODE")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch <= 0:
        parser.error("--epochs and --batch must be positive integers")

    project_root = Path(__file__).resolve().parents[2]
    code_root = project_root / "02_CODE"

    try:
        yaml_path = secure_yaml_resolution(code_root, args.data)
        audit = AuditLogger(log_dir=project_root / "06_LOGS" / "training", user_id="CLI_AUTO")
        audit.log_training_start(config_hash="UNSET", data_version="local", git_commit="unknown")

        detector = YOLOAFBDetector(model_size="n", num_classes=5)
        detector.build_model()
        output = detector.train(data_yaml=yaml_path, epochs=args.epochs, batch_size=args.batch)
        print(f"Training completed. Expected best checkpoint: {output}")
    except Exception as exc:
        print(f"[TRAINING FAILURE] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
