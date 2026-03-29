import os
import cv2
import numpy as np
import yaml
from pathlib import Path

# Security Control: Explicit path assignments
DATA_ROOT = Path("01_DATA/processed_tiles").resolve()
YAML_PATH = Path("02_CODE/data.yaml").resolve()

def create_synthetic_data():
    """Generates localized data explicitly restricted to known folders"""
    for split in ["train", "val"]:
        img_dir = DATA_ROOT / split / "images"
        lbl_dir = DATA_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(5): # Generate 5 dummy synthetic images
            img_path = img_dir / f"synthetic_{i}.jpg"
            lbl_path = lbl_dir / f"synthetic_{i}.txt"
            
            # Clinical ZN background (blue)
            img = np.full((512, 512, 3), (180, 100, 50), dtype=np.uint8) 
            # Add synthetic AFB rod (red)
            cv2.rectangle(img, (250, 250), (258, 280), (50, 50, 200), -1)
            cv2.imwrite(str(img_path), img)
            
            # YOLO label
            with open(lbl_path, "w") as f:
                f.write("0 0.5 0.5 0.05 0.1\n") 
                
    # Writing YAML config strictly
    data_dict = {
        "path": str(DATA_ROOT),
        "train": "train/images",
        "val": "val/images",
        "nc": 5,
        "names": {0: "AFB_Definite", 1: "AFB_Probable", 2: "AFB_Possible", 3: "Debris", 4: "RBC"}
    }
    
    with open(YAML_PATH, "w") as f:
        yaml.safe_dump(data_dict, f)
        
    print(f"Synthetic pipeline instantiated safely under {DATA_ROOT}")

if __name__ == "__main__":
    create_synthetic_data()
