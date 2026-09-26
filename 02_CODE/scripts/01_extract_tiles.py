import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    openslide = None
    OPENSLIDE_AVAILABLE = False


def process_single_tile(slide_path: str, x: int, y: int, patch_size: int, output_dir: str, basename: str) -> int:
    if not OPENSLIDE_AVAILABLE:
        raise RuntimeError("OpenSlide is not available in the worker process.")

    slide = openslide.OpenSlide(slide_path)
    try:
        rgba = slide.read_region((x, y), 0, (patch_size, patch_size))
        img = cv2.cvtColor(np.asarray(rgba), cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(gray) > 0 and np.mean(gray) < 220:
            tile_name = Path(output_dir) / f"{basename}_{x}_{y}.jpg"
            if not cv2.imwrite(str(tile_name), img):
                raise OSError(f"Failed to write tile: {tile_name}")
            return 1
        return 0
    finally:
        slide.close()


def _process_single_tile_args(args) -> int:
    return process_single_tile(*args)


def extract_wsi_patches(wsi_path: Path, output_dir: Path, patch_size: int = 512, overlap: int = 0) -> int:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size.")
    if not wsi_path.is_file():
        raise FileNotFoundError(wsi_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = wsi_path.stem
    step = patch_size - overlap

    if OPENSLIDE_AVAILABLE and wsi_path.suffix.lower() in {".svs", ".ndpi", ".vms", ".vmu", ".scn", ".bif", ".mrxs"}:
        slide = openslide.OpenSlide(str(wsi_path))
        try:
            width, height = slide.dimensions
        finally:
            slide.close()

        tasks = [
            (str(wsi_path), x, y, patch_size, str(output_dir), basename)
            for y in range(0, height - patch_size + 1, step)
            for x in range(0, width - patch_size + 1, step)
        ]
        workers = max(1, min(os.cpu_count() or 1, 8))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            written = sum(executor.map(_process_single_tile_args, tasks, chunksize=8))
        return written

    image = cv2.imread(str(wsi_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {wsi_path.name}")
    height, width = image.shape[:2]
    written = 0
    for y in range(0, height - patch_size + 1, step):
        for x in range(0, width - patch_size + 1, step):
            patch = image[y : y + patch_size, x : x + patch_size]
            if np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)) < 230:
                tile_name = output_dir / f"{basename}_{x}_{y}.jpg"
                if not cv2.imwrite(str(tile_name), patch):
                    raise OSError(f"Failed to write tile: {tile_name}")
                written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", required=True, help="Path to an input slide or image")
    parser.add_argument("--out", default="01_DATA/raw_tiles", help="Output tile directory")
    parser.add_argument("--size", type=int, default=512, help="Patch dimension")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap in pixels")
    args = parser.parse_args()

    written = extract_wsi_patches(
        Path(args.wsi).expanduser().resolve(),
        Path(args.out).expanduser().resolve(),
        args.size,
        args.overlap,
    )
    print(f"Extraction complete: {written} tissue tiles written.")


if __name__ == "__main__":
    main()
