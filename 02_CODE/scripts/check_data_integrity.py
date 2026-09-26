import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from tb_afb.data.integrity import dataset_is_valid, inspect_yolo_dataset


def print_kpi(name: str, status: bool, detail: str = "") -> None:
    color = "\033[92m[PASS]\033[0m" if status else "\033[91m[FAIL]\033[0m"
    print(f"{color} {name:<36} {detail}")


def _print_split_result(split: str, result: dict) -> None:
    prefix = split.upper()
    print_kpi(
        f"{prefix} directory structure",
        result["structure_ok"],
        f"images={result['images']} labels={result['labels']}",
    )

    pairing_ok = result["orphaned_images"] == 0 and result["orphaned_labels"] == 0
    print_kpi(
        f"{prefix} image-label pairing",
        pairing_ok,
        (
            f"orphaned_images={result['orphaned_images']} "
            f"orphaned_labels={result['orphaned_labels']}"
        ),
    )

    print_kpi(
        f"{prefix} image file size",
        result["zero_byte_images"] == 0,
        f"zero_byte_images={result['zero_byte_images']}",
    )

    label_issues = (
        result["malformed"]
        + result["invalid_class"]
        + result["invalid_size"]
        + result["out_of_bounds"]
    )
    print_kpi(
        f"{prefix} YOLO labels",
        label_issues == 0,
        (
            f"boxes={result['total_boxes']} malformed={result['malformed']} "
            f"invalid_class={result['invalid_class']} "
            f"invalid_size={result['invalid_size']} "
            f"out_of_bounds={result['out_of_bounds']}"
        ),
    )


def check_data_integrity() -> bool:
    print("\n=======================================================")
    print("              TB DATASET INTEGRITY CHECK")
    print("=======================================================\n")

    root_dir = Path(__file__).resolve().parents[2]
    processed_root = root_dir / "01_DATA" / "processed_tiles"
    results = inspect_yolo_dataset(processed_root, splits=("train", "val"), num_classes=5)

    for split, result in results.items():
        _print_split_result(split, result)

    yaml_path = root_dir / "02_CODE" / "data.yaml"
    yaml_ok = yaml_path.is_file()
    print_kpi(
        "Training config",
        yaml_ok,
        str(yaml_path.relative_to(root_dir)) if yaml_ok else "02_CODE/data.yaml is missing",
    )

    passed = dataset_is_valid(results) and yaml_ok
    print("\n-------------------------------------------------------")
    if passed:
        print("\033[92mDATA INTEGRITY CHECK: PASSED.\033[0m")
        print("Required splits, image-label pairing, and YOLO label bounds are valid.")
    else:
        print("\033[91mDATA INTEGRITY CHECK: FAILED.\033[0m")
        print("Review the failed checks above before training.")
    print("-------------------------------------------------------\n")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if check_data_integrity() else 1)
