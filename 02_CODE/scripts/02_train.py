#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tb_afb.models.yolo_detector import YOLOAFBDetector
from tb_afb.utils.logger import AuditLogger
from tb_afb.utils.paths import resolve_within


def secure_yaml_resolution(base_dir: Path, yaml_path: str) -> Path:
    path = resolve_within(base_dir, yaml_path, must_exist=True)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Dataset configuration must be a YAML file.")
    return path


def main():
    parser = argparse.ArgumentParser(description="YOLO training orchestrator")
    parser.add_argument("--data", required=True, help="Path to data.yaml inside 02_CODE.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch <= 0:
        parser.error("--epochs and --batch must be positive integers")

    root_dir = Path(__file__).resolve().parents[2]
    code_root = root_dir / "02_CODE"

    try:
        yaml_path = secure_yaml_resolution(code_root, args.data)
        print(f"Training dataset configuration: {yaml_path}")

        audit = AuditLogger(log_dir=root_dir / "06_LOGS" / "training", user_id="CLI_AUTO")
        audit.log_training_start(config_hash="UNSET", data_version="UNSET", git_commit="UNSET")

        detector = YOLOAFBDetector(model_size="n", num_classes=5)
        detector.build_model()
        best = detector.train(
            data_yaml=yaml_path,
            epochs=args.epochs,
            batch_size=args.batch,
        )
        print(f"Training complete. Best checkpoint: {best}")
    except Exception as exc:
        print(f"[TRAINING FAILURE] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
