from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.pipeline import GVParams, PipelineStages, run_pipeline, run_pipeline_with_stages


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

INTEGER_ARRAY_KEYS = {
    "gray",
    "enhanced",
    "raw_mask",
    "clean_mask",
    "labels",
    "overlay",
    "heatmap",
}

REAL_IMAGE_PATHS = [
    "data/input/grieta_clara/1.png",
    "data/input/sin_grietas/9.png",
    "data/input/rugosa/7.png",
    "data/input/sombras/12.png",
]


def make_synthetic_rgb() -> np.ndarray:
    rgb = np.full((96, 128, 3), 220, dtype=np.uint8)
    cv2.line(rgb, (10, 48), (118, 48), (15, 15, 15), thickness=5)
    return rgb


def load_rgb(relative_path: str) -> np.ndarray:
    image_path = ROOT / relative_path
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert bgr is not None, f"No se pudo leer la imagen: {image_path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def assert_results_equal(legacy: dict, staged: dict) -> None:
    assert set(legacy) == set(staged) == EXPECTED_KEYS
    for key in INTEGER_ARRAY_KEYS:
        assert np.array_equal(legacy[key], staged[key]), key
    assert np.allclose(legacy["distance"], staged["distance"])
    pd.testing.assert_frame_equal(legacy["metrics"], staged["metrics"])


def test_stages_container_dimensions_and_types() -> None:
    rgb = make_synthetic_rgb()
    height, width = rgb.shape[:2]

    output = run_pipeline_with_stages(rgb, GVParams())

    assert isinstance(output, tuple)
    assert len(output) == 2
    result, stages = output
    assert set(result) == EXPECTED_KEYS
    assert isinstance(stages, PipelineStages)
    assert stages.denoised.shape == (height, width)
    assert stages.opened.shape == (height, width)
    assert stages.closed.shape == (height, width)
    assert stages.skeleton.shape == (height, width)
    assert stages.denoised.dtype == np.uint8
    assert stages.opened.dtype == np.uint8
    assert stages.closed.dtype == np.uint8
    assert np.issubdtype(stages.skeleton.dtype, np.bool_)


def test_run_pipeline_legacy_contract_remains_exact() -> None:
    result = run_pipeline(make_synthetic_rgb(), GVParams())

    assert set(result) == EXPECTED_KEYS
    assert {"denoised", "opened", "closed", "skeleton"}.isdisjoint(result)


@pytest.mark.parametrize(
    "relative_path",
    [None, *REAL_IMAGE_PATHS],
    ids=["synthetic", "grieta_clara", "sin_grietas", "rugosa", "sombras"],
)
def test_public_entry_points_return_identical_results(
    relative_path: str | None,
) -> None:
    rgb = make_synthetic_rgb() if relative_path is None else load_rgb(relative_path)

    legacy = run_pipeline(rgb, GVParams())
    staged, _ = run_pipeline_with_stages(rgb, GVParams())

    assert_results_equal(legacy, staged)


def test_stages_are_coherent_with_pipeline_results() -> None:
    result, stages = run_pipeline_with_stages(make_synthetic_rgb(), GVParams())

    assert stages.denoised.shape == result["gray"].shape
    assert stages.opened.shape == result["raw_mask"].shape
    assert stages.closed.shape == result["raw_mask"].shape
    assert stages.skeleton.shape == result["clean_mask"].shape
    assert not np.any(stages.skeleton & (result["clean_mask"] == 0))


def test_empty_clean_mask_produces_empty_skeleton() -> None:
    rgb = np.full((64, 80, 3), 255, dtype=np.uint8)

    result, stages = run_pipeline_with_stages(rgb, GVParams())

    assert not np.any(result["clean_mask"])
    assert stages.skeleton.shape == result["clean_mask"].shape
    assert np.issubdtype(stages.skeleton.dtype, np.bool_)
    assert not np.any(stages.skeleton)
