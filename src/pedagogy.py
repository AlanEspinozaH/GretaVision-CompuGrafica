"""Contenido y visualizaciones para la vitrina pedagógica de GretaVision."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StageExplanation:
    title: str
    concept: str
    what_happens: str
    why_used: str
    parameter: str
    what_to_observe: str
    limitation: str


STAGE_EXPLANATIONS: dict[str, StageExplanation] = {
    "Escala de grises": StageExplanation(
        title="Escala de grises",
        concept="Canales RGB, intensidad y representación matricial.",
        what_happens=(
            "Los tres canales RGB se transforman en una única matriz de intensidad."
        ),
        why_used=(
            "Simplifica el procesamiento posterior al trabajar con variaciones de "
            "intensidad en lugar de tres canales de color."
        ),
        parameter="Ninguno; es una conversión fija.",
        what_to_observe=(
            "Compara qué diferencias de color se conservan o desaparecen al representar "
            "cada píxel mediante una sola intensidad."
        ),
        limitation="La conversión descarta información cromática.",
    ),
    "Filtro de mediana": StageExplanation(
        title="Filtro de mediana",
        concept="Vecindad, kernel y mediana local.",
        what_happens=(
            "Cada píxel se sustituye por la mediana de las intensidades de su vecindad."
        ),
        why_used=(
            "Reduce ruido impulsivo y variaciones aisladas mientras preserva "
            "aproximadamente los bordes."
        ),
        parameter="blur_ksize controla el tamaño impar del kernel.",
        what_to_observe=(
            "Observa la reducción de puntos aislados y la conservación aproximada de "
            "los contornos oscuros."
        ),
        limitation="Un kernel grande puede eliminar detalles finos de regiones candidatas.",
    ),
    "CLAHE": StageExplanation(
        title="CLAHE",
        concept="Contraste local, histogramas y división de la imagen en regiones.",
        what_happens=(
            "Se ajusta el histograma de cada región y se limita la amplificación del "
            "contraste antes de combinar las regiones."
        ),
        why_used=(
            "Mejora la visibilidad local de detalles cuando la iluminación no es uniforme."
        ),
        parameter="clipLimit = 2.0 y tileGridSize = 8 × 8; valores fijos.",
        what_to_observe=(
            "Compara la separación local entre zonas oscuras y claras después del realce."
        ),
        limitation="El contraste local también puede amplificar textura o ruido.",
    ),
    "Umbral adaptativo": StageExplanation(
        title="Umbral adaptativo",
        concept="Segmentación binaria mediante un umbral local e inversión.",
        what_happens=(
            "Cada vecindad obtiene un umbral local; el foreground candidato queda blanco "
            "en la máscara binaria."
        ),
        why_used=(
            "Permite segmentar regiones oscuras bajo variaciones locales de iluminación."
        ),
        parameter="block_size define la vecindad y C ajusta el umbral local.",
        what_to_observe=(
            "Observa qué zonas oscuras pasan a foreground blanco y cuáles permanecen "
            "como background."
        ),
        limitation="Sombras y texturas oscuras pueden producir falsos positivos.",
    ),
    "Morfología": StageExplanation(
        title="Morfología",
        concept="Erosión, dilatación, apertura, cierre y elemento estructurante.",
        what_happens=(
            "La apertura aplica erosión seguida de dilatación; el cierre aplica "
            "dilatación seguida de erosión."
        ),
        why_used=(
            "La apertura reduce ruido pequeño y el cierre conecta discontinuidades "
            "locales de la máscara."
        ),
        parameter="morph_kernel define el tamaño del elemento estructurante.",
        what_to_observe=(
            "Compara el ruido eliminado por la apertura y las discontinuidades tratadas "
            "por el cierre."
        ),
        limitation="La morfología puede eliminar trazos finos o unir regiones próximas.",
    ),
    "Componentes conectados": StageExplanation(
        title="Componentes conectados",
        concept=(
            "Conectividad 8, etiquetado de regiones, área, bounding box y relación "
            "de aspecto."
        ),
        what_happens=(
            "Las regiones de la máscara morfológica se etiquetan y se conservan solo "
            "cuando cumplen los filtros geométricos actuales."
        ),
        why_used=(
            "Permite separar regiones candidatas a grietas y descartar componentes que "
            "no cumplen los criterios geométricos."
        ),
        parameter=(
            "min_area, min_width, min_height y min_aspect_ratio controlan el filtrado."
        ),
        what_to_observe=(
            "Compara la máscara posterior a morfología con la máscara limpia, los "
            "bounding boxes y la clasificación visual heurística."
        ),
        limitation=(
            "Una forma geométricamente compatible no garantiza una grieta; se requiere "
            "validación cualitativa."
        ),
    ),
    "Esqueleto y grosor": StageExplanation(
        title="Esqueleto y grosor",
        concept=(
            "Eje central aproximado, conteo de píxeles del esqueleto y distancia "
            "a la frontera."
        ),
        what_happens=(
            "El esqueleto aproxima el eje central y la transformada de distancia "
            "representa la separación relativa respecto de la frontera."
        ),
        why_used=(
            "Permite describir la longitud aproximada del esqueleto y visualizar "
            "variaciones de grosor estimado."
        ),
        parameter="No tiene un parámetro interactivo propio.",
        what_to_observe=(
            "Observa el eje central aproximado y las zonas relativamente más alejadas "
            "del borde de cada región."
        ),
        limitation=(
            "length_px no es longitud euclidiana exacta; el heatmap no usa unidades "
            "físicas y sus colores se normalizan dentro de cada imagen."
        ),
    ),
}


_COMPONENT_PALETTE = np.array(
    [
        (230, 57, 70),
        (29, 120, 180),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 195, 0),
        (0, 187, 249),
        (106, 153, 78),
    ],
    dtype=np.uint8,
)


def _validate_rgb(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray):
        raise ValueError("rgb debe ser un array de NumPy.")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb debe tener forma H x W x 3.")
    if rgb.dtype != np.uint8:
        raise ValueError("rgb debe tener tipo uint8.")


def make_components_visualization(
    rgb: np.ndarray,
    labels: np.ndarray,
    metrics: pd.DataFrame,
    draw_ids: bool = True,
) -> np.ndarray:
    """Colorea y delimita de forma determinista las componentes retenidas."""
    _validate_rgb(rgb)
    if labels.ndim != 2 or labels.shape != rgb.shape[:2]:
        raise ValueError("labels debe tener las mismas dimensiones espaciales que rgb.")

    visualization = rgb.copy()
    if metrics.empty:
        return visualization

    required_columns = {"id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"}
    missing = required_columns.difference(metrics.columns)
    if missing:
        raise ValueError(f"Faltan columnas de métricas: {sorted(missing)}")

    image_height, image_width = rgb.shape[:2]
    for row in metrics.itertuples(index=False):
        component_id = int(row.id)
        color_array = _COMPONENT_PALETTE[(component_id - 1) % len(_COMPONENT_PALETTE)]
        color = tuple(int(channel) for channel in color_array)

        component = labels == component_id
        if np.any(component):
            pixels = visualization[component].astype(np.uint16)
            visualization[component] = ((pixels + color_array) // 2).astype(np.uint8)

        x = int(row.bbox_x)
        y = int(row.bbox_y)
        box_width = int(row.bbox_w)
        box_height = int(row.bbox_h)
        if box_width <= 0 or box_height <= 0:
            continue

        left = max(0, x)
        top = max(0, y)
        right = min(image_width - 1, x + box_width - 1)
        bottom = min(image_height - 1, y + box_height - 1)
        if left > right or top > bottom:
            continue

        cv2.rectangle(visualization, (left, top), (right, bottom), color, 2)
        if draw_ids:
            text_y = top - 5 if top >= 15 else min(image_height - 1, top + 15)
            cv2.putText(
                visualization,
                str(component_id),
                (left, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    return visualization


def make_skeleton_visualization(
    rgb: np.ndarray,
    skeleton: np.ndarray,
) -> np.ndarray:
    """Superpone un esqueleto ya calculado sin modificar las entradas."""
    _validate_rgb(rgb)
    if skeleton.ndim != 2 or skeleton.shape != rgb.shape[:2]:
        raise ValueError("skeleton debe tener las mismas dimensiones espaciales que rgb.")
    if skeleton.dtype != np.bool_:
        raise ValueError("skeleton debe tener tipo booleano.")

    visualization = rgb.copy()
    visualization[skeleton] = np.array((255, 0, 255), dtype=np.uint8)
    return visualization
