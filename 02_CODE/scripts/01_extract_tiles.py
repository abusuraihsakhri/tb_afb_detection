import os
import argparse
from pathlib import Path
import sys
import numpy as np
import cv2

# Secure path initialization
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from tb_afb.utils.logger import AuditLogger

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("[WARNING] OpenSlide binaries missing. Only small TIFF/JPG WSIs will process via Fallback.")

from concurrent.futures import ProcessPoolExecutor

def process_single_tile(slide_path, x, y, patch_size, output_dir, basename):
    """Worker function for multi-processed tile extraction."""
    import openslide
    import numpy as np
    import cv2
    
    slide = openslide.OpenSlide(slide_path)
    rgba_img = slide.read_region((x, y), 0, (patch_size, patch_size))
    img = cv2.cvtColor(np.array(rgba_img), cv2.COLOR_RGBA2BGR)
    
    # Tissue check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) > 0 and np.mean(gray) < 220:
        tile_name = output_dir / f"{basename}_{x}_{y}.jpg"
        cv2.imwrite(str(tile_name), img)
        return 1
    return 0

def extract_wsi_patches(wsi_path: Path, output_dir: Path, patch_size: int = 512, overlap: int = 0):
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = wsi_path.stem
    
    if OPENSLIDE_AVAILABLE and wsi_path.suffix.lower() in [".svs", ".ndpi", ".vms", ".scn"]:
        slide = openslide.OpenSlide(str(wsi_path))
        w, h = slide.dimensions
        step = patch_size - overlap
        
        # 🛡️ SYSTEM OPTIMIZATION: Multi-core Process Orchestration
        tasks = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                if (x + patch_size <= w) and (y + patch_size <= h):
                    tasks.append((str(wsi_path), x, y, patch_size, output_dir, basename))
        
        print(f"[{basename}] Dispatching {len(tasks)} extraction tasks to Multi-core Engine...")
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(lambda p: process_single_tile(*p), tasks))
            
        print(f"Extraction successful: {sum(results)} tissue tiles securely exported.")
    else:
        # Standard Fallback (Single-threaded for safety on small TIFFs)
        img = cv2.imread(str(wsi_path))
        if img is None: return
        h, w = img.shape[:2]
        step = patch_size - overlap
        count = 0
        for y in range(0, h, step):
            for x in range(0, w, step):
                if (x + patch_size <= w) and (y + patch_size <= h):
                    patch = img[y:y+patch_size, x:x+patch_size]
                    if np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)) < 230:
                        cv2.imwrite(str(output_dir / f"{basename}_{x}_{y}.jpg"), patch)
                        count += 1
        print(f"Fallback complete: {count} tiles exported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", type=str, required=True, help="Path to absolute WSI file (.svs/.ndpi)")
    parser.add_argument("--out", type=str, default="01_DATA/raw_tiles", help="Output directory safely routed")
    parser.add_argument("--size", type=int, default=512, help="Patch Dimension (e.g. 512)")
    args = parser.parse_args()
    
    safe_root = Path().resolve()
    wsi_target = Path(args.wsi).resolve()
    
    extract_wsi_patches(wsi_target, Path(args.out).resolve(), args.size)
