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




## Dataset inicial

Se preparó un conjunto inicial de imágenes para probar el pipeline de GretaVision. Las imágenes se agruparon en:

| Categoría | Cantidad | Propósito |
|---|---:|---|
| Grietas claras | 10 | validar la detección en casos simples y muy bien definidos |
| Textura rugosa | 6 | Evaluar la presencia de falsos positivos en superficies complejas |
| Sombras/iluminación irregular | 6 | Analizar la robustez del sistema frente a cambios de iluminación|
| Sin grietas | 4 | Evaluar falsas detecciones en superficies carentes de  fisuras |


Las imágenes serán utilizadas para comparar las distintas etapas del procesamiento: imagen original, máscara binaria, overlay de detección, heatmap (cuando esté disponible) y tabla de métricas generadas por el sistema.

## Enlace al drive del dataset

https://drive.google.com/drive/folders/1N2xJLkMAl62NQDYl9EHbPE0ar0GBXSJO?usp=sharing 