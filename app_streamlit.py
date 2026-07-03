import logging
from pathlib import Path

import cv2
import streamlit as st

from src.pipeline import (
    GVParams,
    decode_uploaded_image,
    run_pipeline,
    encode_png,
    metrics_to_json,
)

logger = logging.getLogger(__name__)
defaults = GVParams()

st.set_page_config(
    page_title="GretaVision | MVP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GretaVision: Segmentación visual de grietas candidatas")
st.markdown(
    """
    MVP académico para cargar imágenes 2D de pavimento o concreto, segmentar regiones
    candidatas a grietas, calcular métricas aproximadas en píxeles y visualizar resultados
    mediante máscara, overlay y mapa de calor.
    """
)

st.warning(
    "GretaVision no realiza diagnóstico estructural. "
    "Las métricas son visuales y aproximadas; si no existe calibración física, se reportan en píxeles."
)

with st.sidebar:
    st.header("Entrada")
    uploaded = st.file_uploader(
        "Cargar imagen",
        type=["png", "jpg", "jpeg"],
        help="Use una imagen 2D de pavimento, concreto o superficie con grietas visibles.",
    )

    st.header("Preprocesamiento")
    blur_ksize = st.slider("Filtro mediana", 3, 15, defaults.blur_ksize, step=2)

    st.header("Segmentación")
    block_size = st.slider("Block size", 3, 99, defaults.block_size, step=2)
    C = st.slider("Constante C", -15, 25, defaults.C, step=1)
    morph_kernel = st.slider(
        "Kernel morfológico",
        1,
        9,
        defaults.morph_kernel,
        step=2,
        help=(
            "Valores mayores eliminan más ruido, pero también pueden borrar "
            "regiones candidatas muy finas."
        ),
    )

    st.header("Filtros geométricos")
    min_area = st.slider("Área mínima (px²)", 0, 5000, defaults.min_area, step=25)
    min_width = st.slider(
        "Eje mayor mínimo del bounding box (px)",
        min_value=1,
        max_value=300,
        value=defaults.min_width,
        step=1,
        help="Filtra componentes demasiado cortas, sin favorecer solo grietas horizontales.",
    )
    min_height = st.slider(
        "Eje menor mínimo del bounding box (px)",
        min_value=1,
        max_value=100,
        value=defaults.min_height,
        step=1,
        help="Evita aceptar componentes extremadamente delgadas o ruido aislado.",
    )
    min_aspect_ratio = st.slider(
        "Relación eje mayor/eje menor mínima",
        min_value=1.0,
        max_value=15.0,
        value=defaults.min_aspect_ratio,
        step=0.1,
        help="Favorece regiones alargadas, típicas de grietas candidatas.",
    )

    st.header("Visualización")
    overlay_alpha = st.slider(
        "Opacidad del overlay", 0.05, 0.95, defaults.overlay_alpha, step=0.05
    )

params = GVParams(
    block_size=block_size,
    C=C,
    blur_ksize=blur_ksize,
    morph_kernel=morph_kernel,
    min_area=min_area,
    min_width=min_width,
    min_height=min_height,
    min_aspect_ratio=min_aspect_ratio,
    overlay_alpha=overlay_alpha,
)

if uploaded is None:
    st.info("Sube una imagen PNG, JPG o JPEG para ejecutar el pipeline.")
    st.stop()

try:
    rgb = decode_uploaded_image(uploaded.getvalue())
    result = run_pipeline(rgb, params)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except cv2.error:
    logger.exception("OpenCV no pudo procesar la imagen")
    st.error("OpenCV no pudo procesar la imagen con los parámetros actuales.")
    st.stop()
except MemoryError:
    logger.exception("Memoria insuficiente al procesar la imagen")
    st.error("No hay memoria suficiente para procesar esta imagen.")
    st.stop()
except Exception:
    logger.exception("Error inesperado al procesar la imagen")
    st.error("Ocurrió un error inesperado al procesar la imagen.")
    st.stop()

df = result["metrics"]
height, width, channels = rgb.shape

st.caption(
    f"Dimensiones: {width} × {height} px · Canales: {channels} (RGB) · "
    f"Tipo de dato: {rgb.dtype} · Métricas: píxeles; áreas expresadas en px²"
)

image_stem = Path(uploaded.name).stem

tab_pipeline, tab_metrics, tab_export, tab_limits = st.tabs(
    ["Pipeline visual", "Métricas", "Exportación", "Limitaciones"]
)

with tab_pipeline:
    st.subheader("Flujo visual del procesamiento")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Imagen original")
        st.image(rgb, use_container_width=True)
    with c2:
        st.caption("Imagen preprocesada")
        st.image(result["enhanced"], clamp=True, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.caption("Máscara refinada")
        st.image(result["clean_mask"], clamp=True, use_container_width=True)
    with c4:
        st.caption("Overlay sobre imagen original")
        st.image(result["overlay"], use_container_width=True)

    st.caption("Mapa relativo de grosor estimado")
    st.image(result["heatmap"], use_container_width=True)
    st.caption(
        "Los colores representan valores relativos de distancia al borde dentro de la imagen. "
        "No corresponden a una medida física."
    )

with tab_metrics:
    st.subheader("Regiones candidatas a grietas")

    if df.empty:
        st.warning(
            "No se detectaron componentes con los parámetros actuales. "
            "Reduce área mínima, eje mayor mínimo o relación de aspecto si la máscara quedó demasiado estricta."
        )
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Componentes", len(df))
        col_b.metric("Área total candidata (px²)", int(df["area_px"].sum()))
        col_c.metric(
            "Longitud aproximada total del esqueleto (px)",
            int(df["length_px"].sum()),
        )
        st.dataframe(
            df,
            column_config={
                "mean_width_px": "Grosor medio estimado (px)",
                "max_width_px": "Grosor máximo estimado (px)",
                "visual_severity": "Clasificación visual heurística",
            },
            use_container_width=True,
        )

        st.subheader("Clasificación visual heurística")
        st.dataframe(
            df["visual_severity"].value_counts().rename_axis("severidad").reset_index(name="cantidad"),
            use_container_width=True,
        )

with tab_export:
    st.subheader("Exportar resultados")

    overlay_png = encode_png(result["overlay"])
    mask_png = encode_png(result["clean_mask"])
    heatmap_png = encode_png(result["heatmap"])
    csv_data = df.to_csv(index=False).encode("utf-8")
    json_data = metrics_to_json(df, uploaded.name, params).encode("utf-8")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Descargar overlay PNG",
            overlay_png,
            f"{image_stem}_overlay.png",
            "image/png",
        )

        st.download_button(
            "Descargar máscara PNG",
            mask_png,
            f"{image_stem}_mask.png",
            "image/png",
        )

    with c2:
        st.download_button(
            "Descargar heatmap PNG",
            heatmap_png,
            f"{image_stem}_heatmap.png",
            "image/png",
        )

    with c3:
        st.download_button(
            "Descargar métricas CSV",
            csv_data,
            f"{image_stem}_metricas.csv",
            "text/csv",
        )

        st.download_button(
            "Descargar reporte JSON",
            json_data,
            f"{image_stem}_reporte.json",
            "application/json",
        )

with tab_limits:
    st.subheader("Alcance y limitaciones del MVP")
    st.markdown(
        """
        - El sistema detecta **regiones candidatas a grietas**, no grietas garantizadas.
        - La severidad reportada es **visual**, no estructural.
        - Las medidas se reportan en **píxeles** si no existe calibración física.
        - Sombras, manchas, juntas del pavimento y textura granular pueden generar falsos positivos.
        - La visualización corresponde a una inspección 2D/2.5D, no a reconstrucción 3D real.
        - Sin máscaras manuales o ground truth, la validación será principalmente cualitativa.
        """
    )
