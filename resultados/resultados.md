# Resultados cualitativos - GretaVision

Este directorio contiene las salidas generadas por GretaVision a partir de las imágenes originales ubicadas en `data/input/`.

Las imágenes fueron clasificadas en cuatro casos:

- `grieta_clara`: imágenes con grietas visibles.
- `rugosa`: superficies con textura granular o irregular.
- `sin_grietas`: superficies sin grietas evidentes.
- `sombras`: imágenes con sombras o iluminación irregular.

Para cada imagen procesada se guardarán:

- máscara binaria;
- overlay;
- heatmap;
- métricas CSV;
- reporte JSON.

La interpretación final se completará después de ejecutar las pruebas con el dataset local.