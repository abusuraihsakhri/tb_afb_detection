import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def load_server():
    path = ROOT / "05_DEPLOYMENT" / "api" / "server.py"
    spec = importlib.util.spec_from_file_location("tb_afb_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bounding_boxes_must_be_normalized_and_in_bounds():
    server = load_server()
    server.BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2, label=0)
    with pytest.raises(ValidationError):
        server.BoundingBox(x=0.05, y=0.5, width=0.2, height=0.2, label=0)
    with pytest.raises(ValidationError):
        server.BoundingBox(x=0.5, y=0.5, width=0.2, height=0.2, label=8)


def test_undecodable_payload_is_rejected():
    server = load_server()
    with pytest.raises(server.HTTPException) as exc:
        server.decode_image(b"not-an-image")
    assert exc.value.status_code == 415


def test_research_summary_does_not_emit_clinical_grade():
    server = load_server()
    assert server.research_summary(12) == "Research candidate count: 12"
