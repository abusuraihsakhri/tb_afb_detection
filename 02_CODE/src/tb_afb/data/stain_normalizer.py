from typing import Optional, Tuple

import numpy as np


class MacenkoNormalizer:
    """Two-stain optical-density normalization using the Macenko method."""

    def __init__(
        self,
        reference_image: Optional[np.ndarray] = None,
        od_threshold: float = 0.15,
        angular_percentile: float = 1.0,
        concentration_percentile: float = 99.0,
    ):
        if od_threshold <= 0:
            raise ValueError("od_threshold must be positive.")
        if not 0 < angular_percentile < 50:
            raise ValueError("angular_percentile must be between 0 and 50.")
        if not 50 < concentration_percentile <= 100:
            raise ValueError("concentration_percentile must be in (50, 100].")

        self.od_threshold = od_threshold
        self.angular_percentile = angular_percentile
        self.concentration_percentile = concentration_percentile
        self.stain_matrix_target: Optional[np.ndarray] = None
        self.max_concentration_target: Optional[np.ndarray] = None

        if reference_image is not None:
            self.fit(reference_image)

    @staticmethod
    def _validate_rgb(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected an HxWx3 RGB image.")
        if array.size == 0:
            raise ValueError("Image cannot be empty.")
        if not np.isfinite(array).all():
            raise ValueError("Image contains NaN or infinite values.")
        return np.clip(array, 0, 255).astype(np.float64)

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        rgb = self._validate_rgb(image)
        return -np.log((rgb + 1.0) / 256.0)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        rgb = 256.0 * np.exp(-np.clip(od, 0.0, 20.0)) - 1.0
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    def _estimate_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        od = self._rgb_to_od(image)
        pixels = od.reshape(-1, 3)
        tissue = pixels[np.any(pixels > self.od_threshold, axis=1)]
        if tissue.shape[0] < 10:
            raise ValueError("Insufficient optical-density variation for stain estimation.")

        covariance = np.cov(tissue, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        plane = eigenvectors[:, order[:2]]

        projection = tissue @ plane
        angles = np.arctan2(projection[:, 1], projection[:, 0])
        low = np.percentile(angles, self.angular_percentile)
        high = np.percentile(angles, 100.0 - self.angular_percentile)

        vectors = np.column_stack(
            (
                plane @ np.array([np.cos(low), np.sin(low)]),
                plane @ np.array([np.cos(high), np.sin(high)]),
            )
        )
        for column in range(2):
            if vectors[:, column].sum() < 0:
                vectors[:, column] *= -1.0
            norm = np.linalg.norm(vectors[:, column])
            if norm <= 1e-12:
                raise ValueError("Degenerate stain vector.")
            vectors[:, column] /= norm

        concentrations = np.linalg.lstsq(vectors, pixels.T, rcond=None)[0]
        maxima = np.percentile(
            np.maximum(concentrations, 0.0),
            self.concentration_percentile,
            axis=1,
        )
        if np.any(maxima <= 1e-8) or not np.isfinite(maxima).all():
            raise ValueError("Degenerate stain concentrations.")
        return vectors, maxima

    @staticmethod
    def _align_source(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        direct = abs(float(source[:, 0] @ target[:, 0])) + abs(float(source[:, 1] @ target[:, 1]))
        swapped = abs(float(source[:, 0] @ target[:, 1])) + abs(float(source[:, 1] @ target[:, 0]))
        return source[:, ::-1] if swapped > direct else source

    def fit(self, image: np.ndarray) -> None:
        matrix, maxima = self._estimate_stain_matrix(image)
        self.stain_matrix_target = matrix
        self.max_concentration_target = maxima

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.stain_matrix_target is None or self.max_concentration_target is None:
            raise RuntimeError("Fit the normalizer on a reference image before transform().")

        rgb = self._validate_rgb(image)
        source_matrix, source_maxima = self._estimate_stain_matrix(rgb)
        source_matrix = self._align_source(source_matrix, self.stain_matrix_target)

        od = self._rgb_to_od(rgb).reshape(-1, 3)
        concentrations = np.linalg.lstsq(source_matrix, od.T, rcond=None)[0]
        concentrations = np.maximum(concentrations, 0.0)

        source_maxima = np.percentile(
            concentrations,
            self.concentration_percentile,
            axis=1,
        )
        source_maxima = np.maximum(source_maxima, 1e-8)
        normalized_concentrations = concentrations * (
            self.max_concentration_target / source_maxima
        )[:, None]

        normalized_od = (self.stain_matrix_target @ normalized_concentrations).T
        return self._od_to_rgb(normalized_od.reshape(rgb.shape))
