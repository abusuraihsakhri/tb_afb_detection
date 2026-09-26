from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT_DIR / "01_DATA" / "processed_tiles"


def create_synthetic_data(samples_per_split: int = 5) -> None:
    """Create small synthetic train/validation fixtures without editing data.yaml."""
    if samples_per_split <= 0:
        raise ValueError("samples_per_split must be positive.")

    for split in ("train", "val"):
        img_dir = DATA_ROOT / split / "images"
        lbl_dir = DATA_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for index in range(samples_per_split):
            img_path = img_dir / f"synthetic_{index}.jpg"
            lbl_path = lbl_dir / f"synthetic_{index}.txt"

            image = np.full((512, 512, 3), (180, 100, 50), dtype=np.uint8)
            cv2.rectangle(image, (250, 250), (258, 280), (50, 50, 200), -1)
            if not cv2.imwrite(str(img_path), image):
                raise OSError(f"Could not write synthetic image: {img_path}")

            with lbl_path.open("w", encoding="utf-8") as handle:
                handle.write("0 0.5 0.5 0.05 0.1\n")

    print(f"Synthetic fixtures created under {DATA_ROOT}")
    print("The tracked 02_CODE/data.yaml configuration was left unchanged.")


if __name__ == "__main__":
    create_synthetic_data()
