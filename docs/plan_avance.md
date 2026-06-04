# Plan mínimo para avance de GretaVision

## Para mañana

### Dataset trabajado
- 10 a 20 imágenes.
- Separar en:
  - grietas claras;
  - textura rugosa;
  - iluminación irregular;
  - sin grietas.
- Guardar en `data/input/`.

### Scripts avanzados
Mostrar:
- `src/pipeline.py`
- `app_streamlit.py`
- notebook de Colab

### Propuesta final lista
Debe decir:
- entrada de imagen;
- preprocesamiento;
- segmentación;
- postprocesamiento;
- análisis geométrico;
- visualización;
- exportación;
- límites: no diagnóstico estructural, métricas en píxeles.

## Reparto recomendado

### Integrante 1
Dataset, problema, justificación, limitaciones.

### Integrante 2
Segmentación, morfología, componentes conectadas, métricas.

### Integrante 3
Streamlit/interfaz, visualización, exportación, demo.
