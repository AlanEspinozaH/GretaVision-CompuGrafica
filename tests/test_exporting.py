import io
import zipfile

import numpy as np
import pandas as pd

from src.exporting import build_export_zip
from src.pipeline import GVParams, encode_png, metrics_to_json


def test_build_results_zip_contains_expected_files_and_contents() -> None:
    inputs = {
        "overlay_png": b"overlay-png-bytes",
        "mask_png": b"mask-png-bytes",
        "heatmap_png": b"heatmap-png-bytes",
        "csv_data": b"id,area_px\n1,250\n",
        "json_data": b'{"components": 1}',
    }

    zip_data, zip_name = build_export_zip("puente_01.jpg", **inputs)

    assert zip_name == "puente_01_resultados_gretavision.zip"
    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        expected_contents = {
            "puente_01_overlay.png": inputs["overlay_png"],
            "puente_01_mascara.png": inputs["mask_png"],
            "puente_01_heatmap.png": inputs["heatmap_png"],
            "puente_01_metricas.csv": inputs["csv_data"],
            "puente_01_reporte.json": inputs["json_data"],
        }

        assert archive.namelist() == list(expected_contents)
        assert len(archive.infolist()) == 5
        for filename, expected_content in expected_contents.items():
            assert archive.read(filename) == expected_content


def test_build_results_zip_preserves_dots_in_image_stem() -> None:
    zip_data, zip_name = build_export_zip(
        "muro.prueba.jpg",
        b"overlay",
        b"mask",
        b"heatmap",
        b"csv",
        b"json",
    )

    assert zip_name == "muro.prueba_resultados_gretavision.zip"
    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        assert archive.namelist() == [
            "muro.prueba_overlay.png",
            "muro.prueba_mascara.png",
            "muro.prueba_heatmap.png",
            "muro.prueba_metricas.csv",
            "muro.prueba_reporte.json",
        ]


def test_build_export_zip_is_byte_for_byte_deterministic() -> None:
    args = (
        "superficie.jpg",
        b"overlay",
        b"mask",
        b"heatmap",
        b"csv",
        b"json",
    )

    first_data, first_name = build_export_zip(*args)
    second_data, second_name = build_export_zip(*args)

    assert first_name == second_name
    assert first_data == second_data
    with zipfile.ZipFile(io.BytesIO(first_data)) as archive:
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_build_export_zip_sanitizes_untrusted_image_name() -> None:
    zip_data, zip_name = build_export_zip(
        r"..\..\muro prueba?.jpg",
        b"overlay",
        b"mask",
        b"heatmap",
        b"csv",
        b"json",
    )

    assert zip_name == "muro_prueba_resultados_gretavision.zip"
    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        for member_name in archive.namelist():
            assert "/" not in member_name
            assert "\\" not in member_name
            assert ".." not in member_name


def test_build_export_zip_does_not_write_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    files_before = list(tmp_path.iterdir())

    build_export_zip("pavimento.png", b"1", b"2", b"3", b"4", b"5")

    assert list(tmp_path.iterdir()) == files_before


def test_empty_metrics_flow_does_not_mutate_inputs() -> None:
    overlay = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=np.uint8)
    heatmap = np.full((4, 4, 3), 127, dtype=np.uint8)
    original_arrays = (overlay.copy(), mask.copy(), heatmap.copy())
    metrics = pd.DataFrame()
    original_metrics = metrics.copy(deep=True)

    overlay_png = encode_png(overlay)
    mask_png = encode_png(mask)
    heatmap_png = encode_png(heatmap)
    csv_data = metrics.to_csv(index=False).encode("utf-8")
    json_data = metrics_to_json(
        metrics,
        "vacia.png",
        GVParams(),
    ).encode("utf-8")

    zip_data, _ = build_export_zip(
        "vacia.png",
        overlay_png,
        mask_png,
        heatmap_png,
        csv_data,
        json_data,
    )

    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        assert archive.read("vacia_overlay.png") == overlay_png
        assert archive.read("vacia_mascara.png") == mask_png
        assert archive.read("vacia_heatmap.png") == heatmap_png
        assert archive.read("vacia_metricas.csv") == csv_data
        assert archive.read("vacia_reporte.json") == json_data

    assert np.array_equal(overlay, original_arrays[0])
    assert np.array_equal(mask, original_arrays[1])
    assert np.array_equal(heatmap, original_arrays[2])
    pd.testing.assert_frame_equal(metrics, original_metrics)
