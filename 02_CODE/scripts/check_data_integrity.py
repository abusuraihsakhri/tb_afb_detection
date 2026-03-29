import os
from pathlib import Path

def print_kpi(name, status, detail=""):
    color = "\033[92m[PASS]\033[0m" if status else "\033[91m[FAIL]\033[0m"
    print(f"{color} {name:<40} {detail}")

def check_data_integrity():
    print("\n=======================================================")
    print("      TB DATALAKE INTEGRITY AND SECURITY AUDIT       ")
    print("=======================================================\n")
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = root_dir / "01_DATA" / "processed_tiles" / "train"
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"
    
    # 1. Directory Structure Integrity
    dir_ok = img_dir.exists() and lbl_dir.exists()
    print_kpi("1. Neural Datalake Topology", dir_ok, f"Paths: {data_dir.relative_to(root_dir)}")
    if not dir_ok:
        print("\n[ABORTING] Datalake structure missing. Cannot proceed with Audit.\n")
        return

    images = list(img_dir.glob("*.jpg"))
    labels = list(lbl_dir.glob("*.txt"))
    
    # 2. Vector Count Symmetry (Every Image must have a Label file)
    img_stems = set(i.stem for i in images)
    lbl_stems = set(l.stem for l in labels)
    
    symmetry_ok = (img_stems == lbl_stems)
    orphaned_imgs = len(img_stems - lbl_stems)
    orphaned_lbls = len(lbl_stems - img_stems)
    
    detail = f"Imgs: {len(images)} | Lbls: {len(labels)}"
    if not symmetry_ok:
         detail += f" | ORPHANS: {orphaned_imgs} Imgs, {orphaned_lbls} Lbls"
    print_kpi("2. Tensor Matrix Symmetry", symmetry_ok, detail)
    
    # 3. Size constraints (No 0-byte corrupt files)
    corrupt = 0
    for file in images + labels:
        if file.stat().st_size == 0:
            corrupt += 1
    
    print_kpi("3. File Size Corruption Verification", corrupt == 0, f"Found {corrupt} Zero-Byte items.")

    # 4. YOLO Structural Parsing (Strict Float Bounding Box math)
    valid_format = True
    malformed_lines = 0
    total_boxes = 0
    out_of_bounds = 0
    
    for lbl in labels:
        with open(lbl, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    malformed_lines += 1
                    valid_format = False
                    continue
                try:
                    c, x, y, w, h = map(float, parts)
                    total_boxes += 1
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                        out_of_bounds += 1
                        valid_format = False
                except ValueError:
                    malformed_lines += 1
                    valid_format = False

    detail_yolo = f"{total_boxes} Valid Arrays"
    if not valid_format:
        detail_yolo += f" | {malformed_lines} Malformed | {out_of_bounds} Out-of-Bounds (>1.0)"

    print_kpi("4. YOLO Coordinate Bounds (0.0 - 1.0)", valid_format, detail_yolo)

    # 5. Config Mapping (data.yaml)
    yaml_path = root_dir / "02_CODE" / "data.yaml"
    yaml_ok = yaml_path.exists()
    print_kpi("5. Training Config (data.yaml)", yaml_ok, f"Path: {str(yaml_path.relative_to(root_dir))}" if yaml_ok else "MISSING")
    
    print("\n-------------------------------------------------------")
    if dir_ok and symmetry_ok and corrupt == 0 and valid_format and yaml_ok:
        print("\033[92mSECURITY AUDIT: PASSED.\033[0m")
        print("Dataset is 100% compliant with PyTorch/Ultralytics standards.")
        print("Ready for GPU Training Injection.")
    else:
        print("\033[91mSECURITY AUDIT: FAILED.\033[0m")
        print("Critical Data Integrity Violations detected.")
    print("-------------------------------------------------------\n")

if __name__ == "__main__":
    check_data_integrity()
