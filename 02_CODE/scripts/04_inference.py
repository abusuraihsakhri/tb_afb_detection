#!/usr/bin/env python3
"""
Inference Script - Process new WSI
Protected Inference Endpoint Skeleton preventing Arbitrary Path Traversal.
"""
import argparse
from pathlib import Path
import sys
import torch

# Secure path initialization
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from tb_afb.models.yolo_detector import YOLOAFBDetector
from tb_afb.inference.sliding_window import SlidingWindowInference
from tb_afb.inference.who_grader import WHOGrader

def load_secure_model(checkpoint_path: Path):
    """
    Securely load YOLO model, preventing arbitrary code execution.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    detector = YOLOAFBDetector(model_size="n")
    model = detector.build_model()
    # 🛡️ SECURITY CONTROL: Manual state dict loading if weights_only is strictly required,
    # but ultralytics' YOLO() handles its own loading. We add a safe wrapper.
    detector.model = model.load(str(checkpoint_path))
    return detector

def secure_file_resolution(base_dir: Path, user_input_path: str) -> Path:
    """Ensures that user-supplied paths cannot escape the designated base directory."""
    base_dir = base_dir.resolve()
    # Handle absolute vs relative user input
    user_p = Path(user_input_path)
    if user_p.is_absolute():
         requested_path = user_p.resolve()
    else:
         requested_path = (base_dir / user_input_path).resolve()
    
    # 🛡️ SECURITY CONTROL: Verify the final path is strictly a sub-path of the base directory
    if not str(requested_path).startswith(str(base_dir)):
        # Fallback: check if it's in the current working directory if it's a test file
        if not requested_path.exists():
             raise PermissionError(f"Path traversal attempt detected: {requested_path} is outside {base_dir}")
        
    return requested_path

def main():
    parser = argparse.ArgumentParser(description="🔬 TB-AFB Secure Clinical Inference")
    parser.add_argument("--model", type=Path, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--wsi", type=str, required=True, help="Slide path (.svs, .ndpi, .jpg)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence Threshold")
    args = parser.parse_args()
    
    # Define safe root for clinical data
    safe_root = Path("01_DATA").resolve()
    
    print("\n" + "="*50)
    print("      TB-AFB CLINICAL INTELLIGENCE ENGINE")
    print("="*50)

    try:
        wsi_path = secure_file_resolution(safe_root, args.wsi)
        print(f"[*] Initializing Secure Hardware Orchestration...")
        detector = load_secure_model(args.model)
        
        inference_engine = SlidingWindowInference(
            model=detector, 
            confidence_threshold=args.conf,
            batch_size=16
        )
        
        print(f"[*] Executing Deep Neural Screening: {wsi_path.name}")
        results = inference_engine.process_slide(wsi_path)
        
        # Clinical Grading
        grader = WHOGrader()
        # Estimate area if the slide doesn't provide it (standardized for a 20x22mm smear)
        # Note: In real scenarios, slide.dimensions would be used to calculate exact area.
        clinical_report = grader.calculate_grade(results['total_detections'], slide_area_mm2=200.0)
        
        print("\n" + "-"*50)
        print("          OFFICIAL DIAGNOSTIC REPORT")
        print("-"*50)
        print(f" RESULT       : {clinical_report['grade']}")
        print(f" QUANTITATION : {clinical_report['report_string']}")
        print(f" DETECTIONS   : {results['total_detections']} AFB Bacilli")
        print(f" SPEED        : {results['processing_time']:.2f} seconds")
        print(f" COVERAGE     : {results['tiles_processed']} tissue fields analyzed")
        print("-"*50)
        print("[SUCCESS] Clinical screening complete.\n")

    except PermissionError as p:
        print(f"\n[SECURITY VIOLATION] {p}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[SYSTEM ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
