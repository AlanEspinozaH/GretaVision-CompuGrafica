"""
GretaVision - pipeline mínimo corregido.

Procesa una imagen 2D, segmenta regiones candidatas a grietas,
calcula métricas aproximadas en píxeles y genera visualizaciones.
No realiza diagnóstico estructural.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize


@dataclass
class GVParams:
    block_size: int = 31
    C: int = 5
    blur_ksize: int = 5
    morph_kernel: int = 3
    min_area: int = 250
    min_width: int = 40          # eje mayor mínimo del bounding box
    min_height: int = 3          # eje menor mínimo del bounding box
    min_aspect_ratio: float = 2.5
    overlay_alpha: float = 0.45


@dataclass(frozen=True)
class PipelineStages:
    denoised: np.ndarray
    opened: np.ndarray
    closed: np.ndarray
    skeleton: np.ndarray


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(int(value), minimum)
    return value if value % 2 == 1 else value + 1


def validate_params(params: GVParams) -> None:
    if params.block_size < 3:
        raise ValueError("block_size debe ser mayor o igual que 3.")
    if params.blur_ksize < 1:
        raise ValueError("blur_ksize debe ser mayor o igual que 1.")
    if params.morph_kernel < 1:
        raise ValueError("morph_kernel debe ser mayor o igual que 1.")
    if params.min_area < 0:
        raise ValueError("min_area debe ser mayor o igual que 0.")
    if params.min_width < 1:
        raise ValueError("min_width debe ser mayor o igual que 1.")
    if params.min_height < 1:
        raise ValueError("min_height debe ser mayor o igual que 1.")
    if params.min_aspect_ratio < 1.0:
        raise ValueError("min_aspect_ratio debe ser mayor o igual que 1.0.")
    if not 0.0 <= params.overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha debe estar entre 0.0 y 1.0.")


def validate_rgb_image(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise ValueError("La imagen debe ser un array de NumPy.")
    if rgb.size == 0:
        raise ValueError("La imagen no puede estar vacía.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("La imagen debe tener forma H x W x 3 (RGB).")
    if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ValueError("El alto y el ancho de la imagen deben ser mayores que cero.")
    if rgb.dtype != np.uint8:
        raise ValueError("La imagen debe tener tipo uint8.")


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    if file_bytes is None:
        raise ValueError("No se recibió contenido de imagen.")
    if not file_bytes:
        raise ValueError("El archivo de imagen está vacío.")
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("No se pudo leer la imagen. Use PNG, JPG o JPEG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _preprocess_with_stages(
    rgb: np.ndarray, params: GVParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    k = ensure_odd(params.blur_ksize)
    denoised = cv2.medianBlur(gray, k)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    return gray, denoised, enhanced


def preprocess(rgb: np.ndarray, params: GVParams) -> Tuple[np.ndarray, np.ndarray]:
    gray, _, enhanced = _preprocess_with_stages(rgb, params)
    return gray, enhanced


def segment_adaptive(enhanced: np.ndarray, params: GVParams) -> np.ndarray:
    block = ensure_odd(params.block_size, minimum=3)
    return cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        params.C,
    )


def _postprocess_with_stages(
    mask: np.ndarray, params: GVParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Limpia la máscara y filtra componentes.

    La relación de aspecto se calcula como eje_mayor/eje_menor para no
    descartar grietas verticales. Usar w/h directamente sesga el detector
    hacia grietas horizontales.
    """
    k = max(1, int(params.morph_kernel))
    kernel = np.ones((k, k), np.uint8)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)

    clean = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        major_axis = max(w, h)
        minor_axis = max(min(w, h), 1)
        aspect_ratio = major_axis / minor_axis

        keep = (
            area >= params.min_area
            and major_axis >= params.min_width
            and minor_axis >= params.min_height
            and aspect_ratio >= params.min_aspect_ratio
        )
        if keep:
            clean[labels == label_id] = 255

    return opened, closed, clean


def postprocess(mask: np.ndarray, params: GVParams) -> np.ndarray:
    """Limpia la máscara y filtra componentes."""
    _, _, clean = _postprocess_with_stages(mask, params)
    return clean


def estimate_orientation(coords_xy: np.ndarray) -> float:
    if len(coords_xy) < 2:
        return 0.0
    centered = coords_xy - coords_xy.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
    if angle_deg > 90:
        angle_deg -= 180
    if angle_deg < -90:
        angle_deg += 180
    return angle_deg


def severity_rule(area_px: int, length_px: int, max_width_px: float) -> str:
    # Regla visual heurística; no equivale a severidad estructural.
    if area_px < 300 or length_px < 50:
        return "baja"
    if area_px < 1500 and max_width_px < 8:
        return "media"
    return "alta"


def _analyze_components_with_stages(
    mask: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = skeletonize(binary.astype(bool))

    rows = []
    for label_id in range(1, num_labels):
        component = labels == label_id
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label_id]

        component_skel = skeleton & component
        length_px = int(np.count_nonzero(component_skel))
        local_widths = 2.0 * distance[component_skel]
        mean_width = float(local_widths.mean()) if local_widths.size else 0.0
        max_width = float(local_widths.max()) if local_widths.size else 0.0

        ys, xs = np.where(component)
        coords_xy = np.column_stack([xs, ys])
        orientation = estimate_orientation(coords_xy)

        major_axis = max(w, h)
        minor_axis = max(min(w, h), 1)
        bbox_aspect_ratio = major_axis / minor_axis

        rows.append({
            "id": int(label_id),
            "area_px": area,
            "centroid_x": round(float(cx), 2),
            "centroid_y": round(float(cy), 2),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "bbox_aspect_ratio": round(float(bbox_aspect_ratio), 2),
            "length_px": length_px,
            "mean_width_px": round(mean_width, 2),
            "max_width_px": round(max_width, 2),
            "orientation_deg": round(orientation, 2),
            "visual_severity": severity_rule(area, length_px, max_width),
        })

    columns = [
        "id", "area_px", "centroid_x", "centroid_y", "bbox_x", "bbox_y",
        "bbox_w", "bbox_h", "bbox_aspect_ratio", "length_px",
        "mean_width_px", "max_width_px", "orientation_deg", "visual_severity"
    ]
    return pd.DataFrame(rows, columns=columns), labels, distance, skeleton


def analyze_components(mask: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataframe, labels, distance, _ = _analyze_components_with_stages(mask)
    return dataframe, labels, distance


def make_overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = rgb.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = ((1 - alpha) * rgb[mask_bool] + alpha * red[mask_bool]).astype(np.uint8)
    return overlay


def make_heatmap(rgb: np.ndarray, mask: np.ndarray, distance: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heat = np.zeros_like(distance, dtype=np.uint8)
    if np.any(mask > 0):
        values = distance[mask > 0]
        max_val = values.max() if values.size else 0
        if max_val > 0:
            heat = np.uint8(np.clip((distance / max_val) * 255, 0, 255))

    color_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    result = rgb.copy()
    mask_bool = mask > 0
    result[mask_bool] = ((1 - alpha) * rgb[mask_bool] + alpha * color_rgb[mask_bool]).astype(np.uint8)
    return result


def encode_png(rgb_or_gray: np.ndarray) -> bytes:
    if rgb_or_gray.ndim == 2:
        image = rgb_or_gray
    else:
        image = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("No se pudo codificar la imagen como PNG.")
    return buffer.tobytes()


def metrics_to_json(df: pd.DataFrame, image_name: str, params: GVParams) -> str:
    payload: Dict = {
        "project": "GretaVision",
        "image_name": image_name,
        "unit": "px",
        "note": "Las métricas son visuales y aproximadas; no son diagnóstico estructural.",
        "parameters": params.__dict__,
        "cracks": df.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _run_pipeline_core(rgb: np.ndarray, params: GVParams):
    validate_params(params)
    validate_rgb_image(rgb)
    gray, denoised, enhanced = _preprocess_with_stages(rgb, params)
    raw_mask = segment_adaptive(enhanced, params)
    opened, closed, clean_mask = _postprocess_with_stages(raw_mask, params)
    df, labels, distance, skeleton = _analyze_components_with_stages(clean_mask)
    overlay = make_overlay(rgb, clean_mask, params.overlay_alpha)
    heatmap = make_heatmap(rgb, clean_mask, distance)
    result = {
        "gray": gray,
        "enhanced": enhanced,
        "raw_mask": raw_mask,
        "clean_mask": clean_mask,
        "metrics": df,
        "labels": labels,
        "distance": distance,
        "overlay": overlay,
        "heatmap": heatmap,
    }
    stages = PipelineStages(
        denoised=denoised,
        opened=opened,
        closed=closed,
        skeleton=skeleton,
    )
    return result, stages


def run_pipeline(rgb: np.ndarray, params: GVParams):
    result, _ = _run_pipeline_core(rgb, params)
    return result


def run_pipeline_with_stages(rgb: np.ndarray, params: GVParams):
    return _run_pipeline_core(rgb, params)
