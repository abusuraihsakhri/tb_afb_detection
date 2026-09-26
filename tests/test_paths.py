import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_yaml_cannot_escape_code_directory(tmp_path):
    module = load_module(ROOT / "02_CODE" / "scripts" / "02_train.py", "train_script")
    base = tmp_path / "02_CODE"
    base.mkdir()
    (base / "data.yaml").write_text("train: x\n", encoding="utf-8")
    assert module.secure_yaml_resolution(base, "data.yaml") == (base / "data.yaml").resolve()

    outside = tmp_path / "02_CODE_evil" / "data.yaml"
    outside.parent.mkdir()
    outside.write_text("train: x\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        module.secure_yaml_resolution(base, str(outside))


def test_inference_input_cannot_escape_data_directory(tmp_path):
    module = load_module(ROOT / "02_CODE" / "scripts" / "04_inference.py", "inference_script")
    base = tmp_path / "01_DATA"
    base.mkdir()
    inside = base / "image.jpg"
    inside.write_bytes(b"x")
    assert module.secure_file_resolution(base, "image.jpg") == inside.resolve()

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")
    with pytest.raises(PermissionError):
        module.secure_file_resolution(base, str(outside))
