from pathlib import Path
from typing import Tuple, Union

import numpy as np
import openslide


class WSIResourceLimitError(Exception):
    """Raised when an individual WSI operation exceeds configured limits."""


WSI_DenialOfService_Error = WSIResourceLimitError


class WSILoader:
    """Whole-slide image loader with bounded region reads."""

    MAX_READ_DIMENSION = 10_000
    MAX_READ_PIXELS = 25_000_000

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path).resolve()
        if not self.file_path.is_file():
            raise FileNotFoundError(f"WSI file not found: {self.file_path}")

        try:
            self.slide = openslide.OpenSlide(str(self.file_path))
        except openslide.OpenSlideError as exc:
            raise ValueError(f"OpenSlide could not open the file: {exc}") from exc

        self.level_count = self.slide.level_count
        self.level_dimensions = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples
        self.properties = dict(self.slide.properties)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _validate_level(self, level: int) -> None:
        if not isinstance(level, int) or not 0 <= level < self.level_count:
            raise ValueError("Requested pyramid level is out of bounds.")

    def get_level_for_magnification(self, target_mag: float) -> int:
        if target_mag <= 0:
            raise ValueError("target_mag must be positive.")

        raw = self.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if raw is None:
            raw = self.properties.get("aperio.AppMag")
        if raw is None:
            raise ValueError("Objective magnification metadata is unavailable.")

        try:
            base_mag = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid objective magnification metadata: {raw}") from exc
        if not np.isfinite(base_mag) or base_mag <= 0:
            raise ValueError(f"Invalid objective magnification metadata: {raw}")

        return self.slide.get_best_level_for_downsample(base_mag / target_mag)

    def read_region(self, level: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        self._validate_level(level)
        if x < 0 or y < 0:
            raise ValueError("Region coordinates must be non-negative.")
        if w <= 0 or h <= 0:
            raise ValueError("Region dimensions must be positive.")
        if w > self.MAX_READ_DIMENSION or h > self.MAX_READ_DIMENSION:
            raise WSIResourceLimitError(
                f"Region dimensions exceed {self.MAX_READ_DIMENSION}px per axis."
            )
        if w * h > self.MAX_READ_PIXELS:
            raise WSIResourceLimitError(
                f"Region exceeds the {self.MAX_READ_PIXELS:,}-pixel read limit."
            )

        level0_width, level0_height = self.level_dimensions[0]
        if x >= level0_width or y >= level0_height:
            raise ValueError("Region origin is outside the slide.")

        image = self.slide.read_region((x, y), level, (w, h)).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def get_pixel_size_microns(self) -> Tuple[float, float]:
        values = []
        for key in (openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y):
            raw = self.properties.get(key)
            if raw is None:
                raise ValueError(f"Required pixel-size metadata is unavailable: {key}")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid pixel-size metadata for {key}: {raw}") from exc
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid pixel-size metadata for {key}: {raw}")
            values.append(value)
        return values[0], values[1]

    def get_mpp_at_level(self, level: int) -> float:
        self._validate_level(level)
        mpp_x, _ = self.get_pixel_size_microns()
        return mpp_x * float(self.level_downsamples[level])

    def close(self) -> None:
        if getattr(self, "slide", None) is not None:
            self.slide.close()
            self.slide = None
