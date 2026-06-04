import streamlit as st
import pandas as pd

from src.pipeline import (
    GVParams,
    decode_uploaded_image,
    run_pipeline,
    encode_png,
    metrics_to_json,
)

st.set_page_config(page_title="GretaVision MVP", layout="wide")

st.title("GretaVision - MVP de inspección visual de grietas")
st.caption(
    "Sistema académico para segmentación, medición aproximada y visualización de grietas candidatas. "
    "No realiza diagnóstico estructural."
)

uploaded = st.file_uploader("Cargar imagen de pavimento/concreto", type=["png", "jpg", "jpeg"])

with st.sidebar:
    st.header("Parámetros")
    block_size = st.slider("Block size - umbral adaptativo", 3, 99, 31, step=2)
    C = st.slider("Constante C", -15, 25, 5, step=1)
    blur_ksize = st.slider("Filtro mediana", 3, 15, 5, step=2)
    morph_kernel = st.slider("Kernel morfológico", 1, 9, 3, step=1)
    min_area = st.slider("Área mínima de componente (px)", 0, 2000, 80, step=10)
    overlay_alpha = st.slider("Opacidad overlay", 0.05, 0.95, 0.45, step=0.05)

params = GVParams(
    block_size=block_size,
    C=C,
    blur_ksize=blur_ksize,
    morph_kernel=morph_kernel,
    min_area=min_area,
    overlay_alpha=overlay_alpha,
)

if uploaded is None:
    st.info("Sube una imagen PNG, JPG o JPEG para ejecutar el pipeline.")
    st.stop()

rgb = decode_uploaded_image(uploaded.getvalue())
result = run_pipeline(rgb, params)

tab1, tab2, tab3, tab4 = st.tabs(["Pipeline", "Métricas", "Exportación", "Límites"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Imagen original")
        st.image(rgb, use_container_width=True)
    with c2:
        st.subheader("Preprocesamiento")
        st.image(result["enhanced"], clamp=True, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Máscara refinada")
        st.image(result["clean_mask"], clamp=True, use_container_width=True)
    with c4:
        st.subheader("Overlay")
        st.image(result["overlay"], use_container_width=True)

    st.subheader("Mapa de calor por grosor estimado")
    st.image(result["heatmap"], use_container_width=True)

with tab2:
    st.subheader("Tabla de grietas candidatas")
    df = result["metrics"]
    if len(df) == 0:
        st.warning("No se detectaron componentes con el área mínima configurada.")
    else:
        st.dataframe(df, use_container_width=True)
        st.write("Resumen:")
        st.write({
            "componentes_detectados": len(df),
            "area_total_px": int(df["area_px"].sum()),
            "longitud_total_px": int(df["length_px"].sum()),
        })

with tab3:
    st.subheader("Descargas")
    overlay_png = encode_png(result["overlay"])
    mask_png = encode_png(result["clean_mask"])
    heatmap_png = encode_png(result["heatmap"])
    csv_data = result["metrics"].to_csv(index=False).encode("utf-8")
    json_data = metrics_to_json(result["metrics"], uploaded.name, params).encode("utf-8")

    st.download_button("Descargar overlay PNG", overlay_png, "gretavision_overlay.png", "image/png")
    st.download_button("Descargar máscara PNG", mask_png, "gretavision_mask.png", "image/png")
    st.download_button("Descargar heatmap PNG", heatmap_png, "gretavision_heatmap.png", "image/png")
    st.download_button("Descargar métricas CSV", csv_data, "gretavision_metrics.csv", "text/csv")
    st.download_button("Descargar reporte JSON", json_data, "gretavision_report.json", "application/json")

with tab4:
    st.markdown("""
    **Límites del MVP**

    - Las medidas están en píxeles si no existe calibración física.
    - La severidad es visual, no estructural.
    - Sombras, manchas, juntas y texturas rugosas pueden generar falsos positivos.
    - La visualización es 2D/2.5D; no hay reconstrucción 3D real.
    """)
