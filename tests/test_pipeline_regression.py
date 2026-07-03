import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    GVParams,
    analyze_components,
    encode_png,
    metrics_to_json,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_KEYS = {
    "gray",
    "enhanced",
    "raw_mask",
    "clean_mask",
    "metrics",
    "labels",
    "distance",
    "overlay",
    "heatmap",
}

METRIC_COLUMNS = [
    "id",
    "area_px",
    "centroid_x",
    "centroid_y",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_aspect_ratio",
    "length_px",
    "mean_width_px",
    "max_width_px",
    "orientation_deg",
    "visual_severity",
]


def make_synthetic_rgb(case: str) -> np.ndarray:
    if case == "white":
        return np.full((96, 128, 3), 255, dtype=np.uint8)
    if case == "dark_line":
        rgb = np.full((96, 128, 3), 255, dtype=np.uint8)
        cv2.line(rgb, (12, 48), (115, 48), (20, 20, 20), thickness=5)
        return rgb
    if case == "small":
        return np.full((32, 32, 3), 180, dtype=np.uint8)
    raise ValueError(f"Caso sintético desconocido: {case}")


@pytest.mark.parametrize("case", ["white", "dark_line", "small"])
def test_run_pipeline_contract_with_synthetic_images(case: str) -> None:
    rgb = make_synthetic_rgb(case)
    height, width = rgb.shape[:2]

    result = run_pipeline(rgb, GVParams())

    assert set(result) == EXPECTED_KEYS

    for key in ("gray", "enhanced", "raw_mask", "clean_mask", "labels", "distance"):
        assert result[key].shape == (height, width)

    for key in ("gray", "enhanced", "raw_mask", "clean_mask"):
        assert result[key].ndim == 2

    assert result["overlay"].shape == (height, width, 3)
    assert result["heatmap"].shape == (height, width, 3)
    assert result["overlay"].dtype == np.uint8
    assert result["heatmap"].dtype == np.uint8
    assert result["raw_mask"].dtype == np.uint8
    assert result["clean_mask"].dtype == np.uint8
    assert np.issubdtype(result["distance"].dtype, np.floating)
    assert isinstance(result["metrics"], pd.DataFrame)
    assert list(result["metrics"].columns) == METRIC_COLUMNS


@pytest.mark.parametrize(
    ("relative_path", "components", "total_area", "total_length"),
    [
        ("data/input/grieta_clara/1.png", 2, 1481, 336),
        ("data/input/sin_grietas/9.png", 0, 0, 0),
        ("data/input/rugosa/7.png", 18, 20888, 3581),
        ("data/input/sombras/12.png", 6, 7229, 1039),
    ],
)
def test_documented_real_image_regression(
    relative_path: str,
    components: int,
    total_area: int,
    total_length: int,
) -> None:
    image_path = ROOT / relative_path
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"No se pudo leer la imagen: {image_path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    metrics = run_pipeline(rgb, GVParams())["metrics"]

    assert len(metrics) == components
    assert int(metrics["area_px"].sum()) == total_area
    assert int(metrics["length_px"].sum()) == total_length


def test_analyze_components_with_empty_mask() -> None:
    mask = np.zeros((48, 64), dtype=np.uint8)

    metrics, labels, distance = analyze_components(mask)

    assert metrics.empty
    assert list(metrics.columns) == METRIC_COLUMNS
    assert labels.shape == mask.shape
    assert distance.shape == mask.shape


def test_encode_png_returns_png_signature() -> None:
    rgb = make_synthetic_rgb("dark_line")

    encoded = encode_png(rgb)

    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_metrics_to_json_contract() -> None:
    metrics, _, _ = analyze_components(np.zeros((24, 24), dtype=np.uint8))

    payload = json.loads(metrics_to_json(metrics, "synthetic.png", GVParams()))

    assert payload["project"] == "GretaVision"
    assert payload["unit"] == "px"
    assert payload["parameters"]["min_area"] == 250
    assert isinstance(payload["cracks"], list)
