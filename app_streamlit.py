import streamlit as st

from src.pipeline import (
    GVParams,
    decode_uploaded_image,
    run_pipeline,
    encode_png,
    metrics_to_json,
)

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
    blur_ksize = st.slider(
        "Filtro mediana",
        min_value=3,
        max_value=15,
        value=5,
        step=2,
        help="Reduce ruido fino antes de segmentar.",
    )

    st.header("Segmentación")
    block_size = st.slider(
        "Block size",
        min_value=3,
        max_value=99,
        value=31,
        step=2,
        help="Tamaño de vecindad para umbral adaptativo. Debe ser impar.",
    )

    C = st.slider(
        "Constante C",
        min_value=-15,
        max_value=25,
        value=5,
        step=1,
        help="Ajusta la sensibilidad del umbral adaptativo.",
    )

    morph_kernel = st.slider(
        "Kernel morfológico",
        min_value=1,
        max_value=9,
        value=3,
        step=1,
        help="Controla opening/closing para limpiar la máscara.",
    )

    st.header("Filtros geométricos")
    min_area = st.slider(
        "Área mínima (px)",
        min_value=0,
        max_value=5000,
        value=250,
        step=25,
        help="Elimina componentes pequeñas que suelen ser ruido.",
    )

    min_width = st.slider(
        "Ancho mínimo de bounding box (px)",
        min_value=1,
        max_value=300,
        value=40,
        step=1,
    )

    min_height = st.slider(
        "Alto mínimo de bounding box (px)",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
    )

    min_aspect_ratio = st.slider(
        "Relación de aspecto mínima",
        min_value=1.0,
        max_value=15.0,
        value=2.5,
        step=0.1,
        help="Favorece regiones alargadas, típicas de grietas candidatas.",
    )

    st.header("Visualización")
    overlay_alpha = st.slider(
        "Opacidad del overlay",
        min_value=0.05,
        max_value=0.95,
        value=0.45,
        step=0.05,
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

rgb = decode_uploaded_image(uploaded.getvalue())
result = run_pipeline(rgb, params)
df = result["metrics"]

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

    st.caption("Mapa de calor por grosor estimado")
    st.image(result["heatmap"], use_container_width=True)

with tab_metrics:
    st.subheader("Métricas de regiones candidatas")

    if df.empty:
        st.warning(
            "No se detectaron componentes con los parámetros actuales. "
            "Reduce área mínima o relación de aspecto si la máscara quedó demasiado estricta."
        )
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Componentes", len(df))
        col_b.metric("Área total detectada (px)", int(df["area_px"].sum()))
        col_c.metric("Longitud total estimada (px)", int(df["length_px"].sum()))

        st.dataframe(df, use_container_width=True)

        if "visual_severity" in df.columns:
            st.subheader("Resumen por severidad visual")
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
            "gretavision_overlay.png",
            "image/png",
        )

        st.download_button(
            "Descargar máscara PNG",
            mask_png,
            "gretavision_mask.png",
            "image/png",
        )

    with c2:
        st.download_button(
            "Descargar heatmap PNG",
            heatmap_png,
            "gretavision_heatmap.png",
            "image/png",
        )

    with c3:
        st.download_button(
            "Descargar métricas CSV",
            csv_data,
            "gretavision_metrics.csv",
            "text/csv",
        )

        st.download_button(
            "Descargar reporte JSON",
            json_data,
            "gretavision_report.json",
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