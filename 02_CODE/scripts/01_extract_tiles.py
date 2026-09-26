import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

try:
    import openslide

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

WSI_EXTENSIONS = {".svs", ".ndpi", ".vms", ".vmu", ".scn", ".bif", ".mrxs"}


def process_single_tile(slide_path, x, y, patch_size, output_dir, basename):
    """Extract one OpenSlide tile in a worker process."""
    import cv2
    import numpy as np
    import openslide

    slide = openslide.OpenSlide(slide_path)
    try:
        rgba_img = slide.read_region((x, y), 0, (patch_size, patch_size))
        img = cv2.cvtColor(np.array(rgba_img), cv2.COLOR_RGBA2BGR)
    finally:
        slide.close()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) > 0 and np.mean(gray) < 220:
        tile_name = Path(output_dir) / f"{basename}_{x}_{y}.jpg"
        if not cv2.imwrite(str(tile_name), img):
            raise OSError(f"Failed to write tile: {tile_name}")
        return 1
    return 0


def extract_wsi_patches(
    wsi_path: Path,
    output_dir: Path,
    patch_size: int = 512,
    overlap: int = 0,
) -> int:
    wsi_path = Path(wsi_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not wsi_path.is_file():
        raise FileNotFoundError(wsi_path)
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size.")

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = wsi_path.stem
    step = patch_size - overlap

    if OPENSLIDE_AVAILABLE and wsi_path.suffix.lower() in WSI_EXTENSIONS:
        slide = openslide.OpenSlide(str(wsi_path))
        try:
            width, height = slide.dimensions
        finally:
            slide.close()

        coords = [
            (x, y)
            for y in range(0, max(0, height - patch_size + 1), step)
            for x in range(0, max(0, width - patch_size + 1), step)
        ]
        print(f"[{basename}] Dispatching {len(coords)} extraction tasks...")

        if not coords:
            return 0

        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        workers = max(1, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(
                process_single_tile,
                repeat(str(wsi_path)),
                xs,
                ys,
                repeat(patch_size),
                repeat(str(output_dir)),
                repeat(basename),
            )
            count = sum(results)
        print(f"Extraction complete: {count} tissue tiles exported.")
        return count

    img = cv2.imread(str(wsi_path))
    if img is None:
        if not OPENSLIDE_AVAILABLE and wsi_path.suffix.lower() in WSI_EXTENSIONS:
            raise RuntimeError("OpenSlide is required for this WSI format.")
        raise ValueError(f"OpenCV could not decode image: {wsi_path}")

    height, width = img.shape[:2]
    count = 0
    for y in range(0, max(0, height - patch_size + 1), step):
        for x in range(0, max(0, width - patch_size + 1), step):
            patch = img[y : y + patch_size, x : x + patch_size]
            if np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)) < 230:
                target = output_dir / f"{basename}_{x}_{y}.jpg"
                if not cv2.imwrite(str(target), patch):
                    raise OSError(f"Failed to write tile: {target}")
                count += 1

    print(f"Fallback extraction complete: {count} tissue tiles exported.")
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract fixed-size image tiles from a slide.")
    parser.add_argument("--wsi", required=True, help="Path to WSI or raster image.")
    parser.add_argument("--out", default="01_DATA/raw_tiles", help="Output directory.")
    parser.add_argument("--size", type=int, default=512, help="Tile width/height in pixels.")
    parser.add_argument("--overlap", type=int, default=0, help="Overlap in pixels.")
    args = parser.parse_args()

    extract_wsi_patches(Path(args.wsi), Path(args.out), args.size, args.overlap)


if __name__ == "__main__":
    main()
