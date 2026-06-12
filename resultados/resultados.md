# Resultados cualitativos - GretaVision

Este directorio contiene las salidas generadas por GretaVision a partir de las imágenes originales ubicadas en `data/input/`.

Las imágenes fueron evaluadas en cuatro categorías:

* `grieta_clara`: imágenes con grietas visibles.
* `rugosa`: superficies con textura granular o irregular.
* `sin_grietas`: superficies sin grietas evidentes.
* `sombras`: imágenes con sombras, manchas oscuras o iluminación irregular.

Para cada imagen procesada se guardaron los siguientes archivos:

* `*_mask.png`: máscara binaria refinada.
* `*_overlay.png`: superposición de la máscara sobre la imagen original.
* `*_heatmap.png`: mapa de calor asociado al grosor estimado.
* `*_metricas.csv`: métricas geométricas por componente detectada.
* `*_reporte.json`: reporte con parámetros, métricas y resultados.
* `*_metricas.png`: captura de la tabla de métricas mostrada en la app.

## Parámetros usados en las pruebas

Las pruebas cualitativas se realizaron con una configuración base del pipeline:

| Parámetro          |  Valor |
| ------------------ | -----: |
| `block_size`       |     31 |
| `C`                |      5 |
| `blur_ksize`       |      5 |
| `morph_kernel`     |      3 |
| `min_area`         | 250 px |
| `min_width`        |  40 px |
| `min_height`       |   3 px |
| `min_aspect_ratio` |    2.5 |
| `overlay_alpha`    |   0.45 |

Estos parámetros representan un punto medio entre sensibilidad y limpieza de ruido. No son valores universales; pueden modificarse desde la interfaz según el contraste, textura e iluminación de cada imagen.

## Resumen de resultados

| Categoría    |   Imagen | Componentes detectados | Área total detectada (px) | Longitud total estimada (px) | Interpretación                                                                               |
| ------------ | -------: | ---------------------: | ------------------------: | ---------------------------: | -------------------------------------------------------------------------------------------- |
| Grieta clara |  `1.png` |                      2 |                      1481 |                          336 | Detección concentrada en regiones alargadas compatibles con la grieta principal.             |
| Grieta clara |  `5.png` |                      3 |                      2159 |                          403 | Detección de una grieta dominante con algunas regiones secundarias.                          |
| Rugosa       |  `7.png` |                     18 |                     20888 |                         3581 | Alta cantidad de regiones candidatas por textura irregular.                                  |
| Rugosa       |  `8.png` |                     16 |                     32403 |                         6318 | Caso problemático: la textura genera una región grande y múltiples falsos positivos.         |
| Sin grietas  |  `9.png` |                      0 |                         0 |                            0 | Caso negativo exitoso: no se detectaron regiones candidatas.                                 |
| Sin grietas  | `10.png` |                      0 |                         0 |                            0 | Caso negativo exitoso: no se detectaron regiones candidatas.                                 |
| Sombras      | `12.png` |                      6 |                      7229 |                         1039 | Las zonas oscuras generan regiones candidatas; evidencia sensibilidad a iluminación.         |
| Sombras      | `13.png` |                     17 |                      8643 |                         1680 | Caso límite: múltiples detecciones asociadas a sombras, manchas o variaciones de intensidad. |

## Análisis por categoría

### 1. Grieta clara

En las imágenes `1.png` y `5.png`, el sistema logró detectar pocas componentes candidatas y asociarlas a regiones alargadas compatibles con grietas visibles.

En `1.png`, GretaVision detectó 2 componentes. La componente principal presentó un área de 1004 px y una longitud estimada de 213 px. Esto muestra que el pipeline puede aislar una grieta visible cuando existe buen contraste con el fondo.

En `5.png`, el sistema detectó 3 componentes. La región dominante presentó un área de 1519 px y una longitud estimada de 272 px, lo que indica una detección relevante de una grieta más extendida.

Estos casos son los más favorables para el método, porque las grietas son oscuras, alargadas y relativamente distinguibles respecto al pavimento.

### 2. Superficie rugosa

Las imágenes `7.png` y `8.png` evidencian una limitación importante del método.

En `7.png`, se detectaron 18 componentes, incluyendo una región de gran tamaño con área de 11220 px y longitud estimada de 1561 px. En `8.png`, se detectaron 16 componentes, incluyendo una región muy extensa con área de 23190 px y longitud estimada de 4408 px.

Estos resultados muestran que las superficies rugosas pueden generar falsos positivos. La textura granular del pavimento produce zonas oscuras y alargadas que el algoritmo puede interpretar como grietas candidatas.

Este comportamiento es esperable porque GretaVision usa procesamiento clásico basado en intensidad, contraste, umbralización y geometría. No cuenta con un modelo semántico que distinga automáticamente entre textura superficial y grieta real.

### 3. Imágenes sin grietas

Las imágenes `9.png` y `10.png` funcionan como casos negativos.

En ambos casos, el sistema no detectó componentes candidatas. Esto es positivo porque indica que, bajo los parámetros utilizados, GretaVision no marcó grietas donde no existían regiones oscuras alargadas suficientemente relevantes.

Estos resultados ayudan a validar que los filtros geométricos y el área mínima pueden reducir falsos positivos en imágenes limpias o sin grietas evidentes.

### 4. Sombras e iluminación irregular

Las imágenes `12.png` y `13.png` permiten evaluar la sensibilidad del sistema ante zonas oscuras que no necesariamente corresponden a grietas.

En `12.png`, se detectaron 6 componentes, incluyendo una región de 4682 px de área y 579 px de longitud estimada. En `13.png`, se detectaron 17 componentes, lo que evidencia una mayor fragmentación y presencia de regiones oscuras candidatas.

Estos casos muestran que sombras, manchas o iluminación irregular pueden confundirse con grietas. Esto ocurre porque la segmentación se basa en diferencias de intensidad: si una sombra tiene forma alargada o contraste fuerte, puede superar los filtros geométricos.

## Observaciones técnicas

GretaVision produce mejores resultados cuando:

* la grieta es oscura;
* la grieta tiene forma alargada;
* existe buen contraste entre grieta y fondo;
* la textura del pavimento no es demasiado granular;
* la iluminación es relativamente uniforme.

El sistema presenta más falsos positivos cuando:

* hay textura rugosa;
* existen sombras marcadas;
* hay manchas oscuras;
* aparecen bordes o cambios bruscos de iluminación;
* la superficie tiene patrones parecidos a grietas.

## Limitaciones

Los resultados son cualitativos. No se reportan métricas como precisión, recall, Dice o IoU porque no se cuenta con máscaras manuales de referencia o ground truth pixel a pixel.

Las métricas generadas por GretaVision son aproximadas y se reportan en píxeles. No representan medidas físicas en milímetros o centímetros, ya que no se aplicó calibración espacial.

La severidad visual calculada por el sistema es heurística. No debe interpretarse como severidad estructural ni como diagnóstico profesional.

## Conclusión

GretaVision cumple como MVP académico para segmentar, visualizar y medir aproximadamente regiones candidatas a grietas en imágenes 2D de pavimento o concreto.

El sistema funciona mejor en imágenes con grietas claras y buen contraste. También mostró buen comportamiento en los casos negativos `9.png` y `10.png`, donde no se detectaron componentes candidatas.

Las principales limitaciones aparecen en superficies rugosas y en imágenes con sombras o iluminación irregular, donde se generan falsos positivos. Por ello, GretaVision debe entenderse como una herramienta de apoyo visual y no como un detector definitivo ni como un sistema de diagnóstico estructural.
