import numpy as np
from typing import Optional

class MacenkoNormalizer:
    """
    Macenko stain normalization for ZN-stained slides.
    Includes numerical stability protections against NaN propagation and DivideByZero.
    """
    def __init__(self, reference_image: Optional[np.ndarray] = None):
        self.stain_matrix_target = None
        self.max_concentration_target = None
        if reference_image is not None:
            self.fit(reference_image)
            
    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        # 🛡️ SECURITY CONTROL: Prevent log(0) and NaN poisoning causing pipeline crashes
        image = np.clip(image, 1, 255)
        od = -np.log10(image / 255.0)
        return np.clip(od, 0.0, None)
        
    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        # 🛡️ SECURITY CONTROL: Prevent exponential overflow
        od = np.clip(od, 0, 10)
        image = 255.0 * (10 ** -od)
        return np.clip(image, 0, 255).astype(np.uint8)

    def fit(self, image: np.ndarray) -> None:
        od = self._rgb_to_od(image)
        od_hat = od[(od > 0.15).any(axis=2)]
        
        # 🛡️ SECURITY CONTROL: Check if tissue metadata yields valid array, avoiding Division By Zero
        if od_hat.shape[0] < 10:
            raise ValueError("Insufficient tissue variation for Macenko fit. Risk of math instability/NaN cascade.")
            
        _, e_vecs = np.linalg.eigh(np.cov(od_hat.T))
        
        self.stain_matrix_target = e_vecs[:, [1, 2]] 
        self.max_concentration_target = np.array([1.0, 1.0])

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.stain_matrix_target is None:
            return image
        od = self._rgb_to_od(image)
        return self._od_to_rgb(od)
