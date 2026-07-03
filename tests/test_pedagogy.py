from dataclasses import fields

import numpy as np
import pandas as pd

from src.pedagogy import (
    STAGE_EXPLANATIONS,
    StageExplanation,
    make_components_visualization,
    make_skeleton_visualization,
)


EXPECTED_STAGE_NAMES = [
    "Escala de grises",
    "Filtro de mediana",
    "CLAHE",
    "Umbral adaptativo",
    "Morfología",
    "Componentes conectados",
    "Esqueleto y grosor",
]


def make_metrics(
    *,
    x: int = 7,
    y: int = 5,
    width: int = 23,
    height: int = 15,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": width,
                "bbox_h": height,
            }
        ]
    )


def test_stage_explanations_have_exact_order_and_required_fields() -> None:
    assert list(STAGE_EXPLANATIONS) == EXPECTED_STAGE_NAMES
    assert len(STAGE_EXPLANATIONS) == 7

    for name, explanation in STAGE_EXPLANATIONS.items():
        assert isinstance(explanation, StageExplanation)
        assert explanation.title == name
        for field in fields(StageExplanation):
            value = getattr(explanation, field.name)
            assert isinstance(value, str)
            assert value.strip()


def test_components_visualization_is_deterministic_and_non_mutating() -> None:
    rgb = np.full((40, 50, 3), 120, dtype=np.uint8)
    labels = np.zeros((40, 50), dtype=np.int32)
    labels[5:20, 7:30] = 1
    metrics = make_metrics()

    rgb_before = rgb.copy()
    labels_before = labels.copy()
    metrics_before = metrics.copy(deep=True)

    first = make_components_visualization(rgb, labels, metrics)
    second = make_components_visualization(rgb, labels, metrics)

    assert first.shape == rgb.shape
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert not np.array_equal(first, rgb)
    assert np.array_equal(rgb, rgb_before)
    assert np.array_equal(labels, labels_before)
    pd.testing.assert_frame_equal(metrics, metrics_before)


def test_components_visualization_handles_empty_metrics() -> None:
    rgb = np.full((24, 32, 3), 80, dtype=np.uint8)
    labels = np.zeros((24, 32), dtype=np.int32)

    visualization = make_components_visualization(rgb, labels, pd.DataFrame())

    assert visualization.shape == rgb.shape
    assert visualization.dtype == np.uint8
    assert np.array_equal(visualization, rgb)


def test_component_bounding_boxes_are_clamped_to_image() -> None:
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    labels = np.zeros((20, 30), dtype=np.int32)
    metrics = make_metrics(x=-5, y=-4, width=50, height=40)

    visualization = make_components_visualization(
        rgb,
        labels,
        metrics,
        draw_ids=False,
    )

    assert np.any(visualization[0, 0] != rgb[0, 0])
    assert np.any(visualization[-1, -1] != rgb[-1, -1])


def test_skeleton_visualization_is_rgb_uint8_and_non_mutating() -> None:
    rgb = np.full((30, 36, 3), 100, dtype=np.uint8)
    skeleton = np.zeros((30, 36), dtype=bool)
    skeleton[5:25, 18] = True

    rgb_before = rgb.copy()
    skeleton_before = skeleton.copy()

    visualization = make_skeleton_visualization(rgb, skeleton)

    assert visualization.shape == rgb.shape
    assert visualization.dtype == np.uint8
    assert not np.array_equal(visualization, rgb)
    assert np.array_equal(rgb, rgb_before)
    assert np.array_equal(skeleton, skeleton_before)


def test_skeleton_visualization_handles_empty_skeleton() -> None:
    rgb = np.full((18, 22, 3), 160, dtype=np.uint8)
    skeleton = np.zeros((18, 22), dtype=bool)

    visualization = make_skeleton_visualization(rgb, skeleton)

    assert visualization.shape == rgb.shape
    assert visualization.dtype == np.uint8
    assert np.array_equal(visualization, rgb)
