import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from pathlib import Path
from typing import Tuple, Union, Any

class WSI_DenialOfService_Error(Exception):
    """Exception raised when WSI parsing violates security constraints."""
    pass

class WSILoader:
    """
    Whole Slide Image loader with format abstraction.
    Includes memory exhaustion (DoS) protections.
    """
    
    # 🛡️ SECURITY CONTROL: Maximum dimension to prevent decompression bomb attacks
    MAX_ALLOWED_DIMENSION = 200000 
    
    def __init__(self, file_path: Union[str, Path]):
        """Initialize and validate WSI file securely."""
        self.file_path = Path(file_path).resolve()
        
        # Verify file exists before passing to native librarires
        if not self.file_path.exists():
            raise FileNotFoundError(f"WSI file not found: {self.file_path}")
            
        try:
            self.slide = openslide.OpenSlide(str(self.file_path))
        except openslide.OpenSlideError as e:
            raise ValueError(f"Failed to open WSI or corrupt file header: {e}")
            
        self.level_count = self.slide.level_count
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples
        self.properties = dict(self.slide.properties)
        
        # 🛡️ SECURITY CONTROL: Validate logical dimensions against integer overflow & extreme memory allocation loops 
        width, height = self.level_dimensions[0]
        if width > self.MAX_ALLOWED_DIMENSION or height > self.MAX_ALLOWED_DIMENSION:
            self.close()
            raise WSI_DenialOfService_Error(
                f"WSI dimensions ({width}x{height}) exceed the security constraint ({self.MAX_ALLOWED_DIMENSION})"
            )
            
    def get_level_for_magnification(self, target_mag: float) -> int:
        """Find pyramid level closest to target magnification."""
        base_mag_str = self.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, None)
        if base_mag_str is None:
            base_mag_str = self.properties.get('aperio.AppMag', '40.0') # Fallback
            
        try:
            base_mag = float(base_mag_str)
        except ValueError:
            raise ValueError(f"Invalid objective power metadata: {base_mag_str}")
            
        target_downsample = base_mag / target_mag
        level = self.slide.get_best_level_for_downsample(target_downsample)
        return level
        
    def read_region(self, level: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Read region from WSI at specified level with strict layout bounds.
        """
        # 🛡️ SECURITY CONTROL: Prevent excessive read allocations (max 10,000x10,000 memory buffer per call)
        if w > 10000 or h > 10000:
             raise ValueError("Requested tile extraction exceeds safe memory bounds (10,000px limit).")
             
        if level >= self.level_count or level < 0:
            raise ValueError("Requested pyramid level out of bounds.")
            
        # Execute underlying read operation on valid bounds
        image_pil = self.slide.read_region((x, y), level, (w, h)).convert("RGB")
        return np.array(image_pil, dtype=np.uint8)
        
    def get_pixel_size_microns(self) -> Tuple[float, float]:
        """Return (pixel_width, pixel_height) in microns."""
        mpp_x = float(self.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        mpp_y = float(self.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))
        return (mpp_x, mpp_y)
        
    def get_mpp_at_level(self, level: int) -> float:
        """Microns per pixel at specified level."""
        mpp_x, _ = self.get_pixel_size_microns()
        downsample = self.level_downsamples[level]
        return mpp_x * downsample
        
    def close(self) -> None:
        """Release underlying system file handles to prevent FD exhaustion."""
        if hasattr(self, 'slide') and self.slide is not None:
            self.slide.close()
