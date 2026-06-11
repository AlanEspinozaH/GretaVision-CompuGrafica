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
    blur_ksize = st.slider("Filtro mediana", 3, 15, 5, step=2)

    st.header("Segmentación")
    block_size = st.slider("Block size", 3, 99, 31, step=2)
    C = st.slider("Constante C", -15, 25, 5, step=1)
    morph_kernel = st.slider("Kernel morfológico", 1, 9, 3, step=1)

    st.header("Filtros geométricos")
    min_area = st.slider("Área mínima (px)", 0, 5000, 250, step=25)
    min_width = st.slider(
        "Eje mayor mínimo del bounding box (px)",
        min_value=1,
        max_value=300,
        value=40,
        step=1,
        help="Filtra componentes demasiado cortas, sin favorecer solo grietas horizontales.",
    )
    min_height = st.slider(
        "Eje menor mínimo del bounding box (px)",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
        help="Evita aceptar componentes extremadamente delgadas o ruido aislado.",
    )
    min_aspect_ratio = st.slider(
        "Relación eje mayor/eje menor mínima",
        min_value=1.0,
        max_value=15.0,
        value=2.5,
        step=0.1,
        help="Favorece regiones alargadas, típicas de grietas candidatas.",
    )

    st.header("Visualización")
    overlay_alpha = st.slider("Opacidad del overlay", 0.05, 0.95, 0.45, step=0.05)

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
            "Reduce área mínima, eje mayor mínimo o relación de aspecto si la máscara quedó demasiado estricta."
        )
    else:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Componentes", len(df))
        col_b.metric("Área total detectada (px)", int(df["area_px"].sum()))
        col_c.metric("Longitud total estimada (px)", int(df["length_px"].sum()))
        st.dataframe(df, use_container_width=True)

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
        st.download_button("Descargar overlay PNG", overlay_png, "gretavision_overlay.png", "image/png")
        st.download_button("Descargar máscara PNG", mask_png, "gretavision_mask.png", "image/png")
    with c2:
        st.download_button("Descargar heatmap PNG", heatmap_png, "gretavision_heatmap.png", "image/png")
    with c3:
        st.download_button("Descargar métricas CSV", csv_data, "gretavision_metrics.csv", "text/csv")
        st.download_button("Descargar reporte JSON", json_data, "gretavision_report.json", "application/json")

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
