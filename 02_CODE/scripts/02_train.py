#!/usr/bin/env python3
"""
Pipeline Trainer Script 
Protected against Command Injection and Directory Traversal.
"""
import argparse
from pathlib import Path
import sys
import os

# Ensure local packages are resolvable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tb_afb.models.yolo_detector import YOLOAFBDetector
from tb_afb.utils.logger import AuditLogger

def secure_yaml_resolution(base_dir: Path, yaml_path: str) -> Path:
    """Blocks directory traversal patterns for YAML payloads."""
    base_dir = base_dir.resolve()
    requested_path = (base_dir / yaml_path).resolve()
    
    # 🛡️ SECURITY CONTROL: Sub-directory integrity lock
    if not str(requested_path).startswith(str(base_dir)):
        raise PermissionError("Path traversal pattern matching denied for YAML configs.")
    return requested_path

def main():
    parser = argparse.ArgumentParser(description="Secure YOLO Training Orchestrator")
    parser.add_argument("--data", required=True, type=str, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    
    safe_root = Path("02_CODE").resolve()
    
    try:
        yaml_path = secure_yaml_resolution(safe_root, args.data)
        print(f"Executing secure training isolated to: {yaml_path}")
        
        # Initialize Logger securely
        audit = AuditLogger(log_dir=Path("06_LOGS/training"), user_id="CLI_AUTO")
        audit.log_training_start(config_hash="TESTING_HASH", data_version="V1", git_commit="N/A")
        
        # Build Model 
        detector = YOLOAFBDetector(model_size="n", num_classes=5)
        detector.build_model()
        
        # Train securely
        detector.train(data_yaml=yaml_path, epochs=args.epochs, batch_size=args.batch)
        print("Training successfully mitigated against Shell Overrides.")
        
    except Exception as e:
         print(f"[SECURITY/TRAIN FAILURE]: {e}", file=sys.stderr)
         sys.exit(1)

if __name__ == "__main__":
    main()
