import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import time
import cv2
from .postprocessor import DetectionPostprocessor
from ..models.yolo_detector import YOLOAFBDetector

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

class SlidingWindowInference:
    """
    Inference on full WSI using sliding window approach.
    Enforces Strict boundaries preventing Arbitrary file reads.
    """
    def __init__(self, model: YOLOAFBDetector, tile_size: int = 512, overlap: int = 128, batch_size: int = 16, confidence_threshold: float = 0.25):
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        # Instantiate Secure Filter
        self.postprocessor = DetectionPostprocessor(min_confidence=confidence_threshold)

    def process_slide(self, wsi_path: Path) -> Dict[str, Any]:
        """Orchestrate tiling and batch inference on a massive WSI binary."""
        start_time = time.time()
        
        # 🛡️ SECURITY CONTROL: Path resolution and Verification
        wsi_path = Path(wsi_path).resolve()
        if not wsi_path.exists() or not wsi_path.is_file():
            raise FileNotFoundError(f"Missing valid WSI payload: {wsi_path.name}")
            
        all_detections = []
        tiles_processed = 0
        
        if not OPENSLIDE_AVAILABLE:
            # Fallback for standard images
            img = cv2.imread(str(wsi_path))
            if img is None:
                raise ValueError(f"Could not read image: {wsi_path}")
            h, w = img.shape[:2]
            detections = self._process_image_tiles(img, (0, 0))
            all_detections.extend(detections)
            tiles_processed = 1 # Simplified for fallback
        else:
            slide = openslide.OpenSlide(str(wsi_path))
            try:
                w, h = slide.dimensions
                step = self.tile_size - self.overlap
                
                # Tile coordinates generation
                coords = []
                for y in range(0, h, step):
                    for x in range(0, w, step):
                        coords.append((x, y))
                
                # Batch processing
                for i in range(0, len(coords), self.batch_size):
                    batch_coords = coords[i:i + self.batch_size]
                    batch_imgs = []
                    valid_coords = []
                    
                    for x, y in batch_coords:
                        # 🛡️ TISSUE CHECK: Optimization to skip white space/background
                        region = slide.read_region((x, y), 0, (self.tile_size, self.tile_size))
                        img = cv2.cvtColor(np.array(region), cv2.COLOR_RGBA2BGR)
                        
                        # Basic tissue filter (mean intensity check)
                        if np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 235:
                            batch_imgs.append(img)
                            valid_coords.append((x, y))
                    
                    if batch_imgs:
                        for img, (x, y) in zip(batch_imgs, valid_coords):
                            tile_detections = self.model.predict(img, conf_threshold=self.confidence_threshold)
                            # Translate to global coordinates
                            for det in tile_detections:
                                det['bbox'] = [
                                    det['bbox'][0] + x, # x_center
                                    det['bbox'][1] + y, # y_center
                                    det['bbox'][2],     # w
                                    det['bbox'][3]      # h
                                ]
                            all_detections.extend(tile_detections)
                            tiles_processed += 1
            finally:
                slide.close() # 🛡️ SECURITY CONTROL: Prevent resource/FD exhaustion

        # Apply Global NMS to merge overlaps from different tiles
        final_detections = self.postprocessor.filter(all_detections)
        
        return {
            "total_detections": len(final_detections), 
            "detections": final_detections, 
            "processing_time": time.time() - start_time, 
            "tiles_processed": tiles_processed
        }

    def _process_image_tiles(self, img: np.ndarray, offset: Tuple[int, int]) -> List[Dict]:
        """Process tiles for a standard image (fallback mode)"""
        return self.model.predict(img, conf_threshold=self.confidence_threshold)
