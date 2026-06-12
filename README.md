# GretaVision - MVP de Computación Gráfica

GretaVision es un MVP académico para la **segmentación, medición aproximada y visualización de regiones candidatas a grietas** en imágenes 2D de pavimento o concreto.

El sistema permite cargar una imagen, aplicar preprocesamiento, segmentar posibles grietas, limpiar la máscara binaria, calcular métricas geométricas en píxeles y visualizar los resultados mediante máscara, overlay y mapa de calor.

> **Advertencia académica:** GretaVision no realiza diagnóstico estructural profesional. Las métricas generadas son visuales y aproximadas. Sin calibración física, todas las medidas se reportan en píxeles.

## Objetivo del proyecto

El objetivo principal es construir una herramienta académica de apoyo visual para analizar imágenes de superficies con posibles grietas, aplicando técnicas de computación gráfica y procesamiento digital de imágenes.

El sistema busca demostrar:

1. Carga de imágenes de pavimento o concreto.
2. Preprocesamiento para reducción de ruido y mejora de contraste.
3. Segmentación de regiones oscuras candidatas a grietas.
4. Limpieza de máscara mediante operaciones morfológicas.
5. Filtrado de componentes conectados.
6. Cálculo de métricas geométricas aproximadas.
7. Visualización mediante overlay y mapa de calor.
8. Exportación de métricas en CSV/JSON.
9. Demo funcional mediante Google Colab y Streamlit.



## 1. Funcionalidades actuales

* Carga de imágenes en formato JPG, JPEG o PNG.
* Preprocesamiento:

  * conversión a escala de grises;
  * reducción de ruido;
  * mejora de contraste.
* Segmentación por umbral adaptativo inverso.
* Limpieza morfológica de la máscara.
* Filtrado de componentes conectados.
* Cálculo de métricas aproximadas:

  * área en píxeles;
  * centroide;
  * bounding box;
  * longitud estimada;
  * grosor promedio;
  * grosor máximo;
  * orientación aproximada;
  * severidad visual.
* Visualización:

  * imagen original;
  * imagen preprocesada;
  * máscara refinada;
  * overlay;
  * mapa de calor.
* Exportación:

  * overlay PNG;
  * máscara PNG;
  * heatmap PNG;
  * métricas CSV;
  * reporte JSON.


## 2. Tecnologías utilizadas

* Python
* OpenCV
* NumPy
* Pandas
* Matplotlib
* scikit-image
* Streamlit
* Google Colab
* Git/GitHub


## 3. Instalación local

Clonar el repositorio:

```bash
git clone https://github.com/AlanEspinozaH/GretaVision-CompuGrafica.git
cd GretaVision-CompuGrafica
```

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno virtual en Linux/macOS:

```bash
source .venv/bin/activate
```

Activar entorno virtual en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la aplicación:

```bash
streamlit run app_streamlit.py
```



## 4. Ejecución rápida en Google Colab

Abrir el notebook:

```text
notebooks/GretaVision_MVP_Colab.ipynb
```

Luego ejecutar las celdas y subir una imagen de prueba.

Esta opción se recomienda para validar rápidamente el pipeline sin instalar dependencias localmente.



## 5. Estructura del proyecto

```text
GretaVision-CompuGrafica/
├── app_streamlit.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   └── pipeline.py
├── notebooks/
│   └── GretaVision_MVP_Colab.ipynb
├── data/
│   └── input/
│       ├── grieta_clara/
│       ├── rugosa/
│       ├── sin_grietas/
│       └── sombras/
├── resultados/
│   ├── grieta_clara/
│   ├── rugosa/
│   ├── sin_grietas/
│   ├── sombras/
│   └── resultados.md
└── docs/
    ├── plan_avance.md
    └── validacion_cualitativa.md
```

Descripción general:

* `data/input/`: contiene las imágenes originales clasificadas por tipo de caso.
* `resultados/`: contiene las salidas generadas por GretaVision: máscaras, overlays, heatmaps, métricas CSV, reportes JSON y capturas de métricas.
* `src/pipeline.py`: contiene el pipeline principal de procesamiento, segmentación, postprocesamiento, métricas y visualización.
* `app_streamlit.py`: contiene la interfaz interactiva de la demo.
* `docs/`: contiene documentación complementaria del avance y validación cualitativa.

## 6. Flujo general del pipeline

```text
Imagen RGB
→ Escala de grises
→ Reducción de ruido con filtro de mediana
→ Mejora de contraste local con CLAHE
→ Umbral adaptativo inverso
→ Operaciones morfológicas
→ Componentes conectados
→ Filtrado geométrico
→ Skeletonization
→ Distance Transform
→ Métricas aproximadas
→ Overlay / Heatmap / Exportación
```

El filtrado geométrico considera:

* área mínima de componente;
* eje mayor mínimo del bounding box;
* eje menor mínimo del bounding box;
* relación eje mayor/eje menor mínima.

La relación eje mayor/eje menor se usa para favorecer regiones alargadas sin limitar el sistema únicamente a grietas horizontales. Esto permite conservar grietas verticales o diagonales.


## 7. Distribución de trabajo

| Integrante   | Rama sugerida               | Responsabilidad principal                            | Archivos principales                                       |
| ------------ | --------------------------- | ---------------------------------------------------- | ---------------------------------------------------------- |
| Integrante 1: Merchan Bianca | `feature/dataset-docs`      | Dataset, pruebas, capturas y documentación breve     | `data/input/`, `docs/plan_avance.md`, `README.md`          |
| Integrante 2: Guerrero Jhiens| `feature/pipeline-metricas` | Pipeline, segmentación, postprocesamiento y métricas | `src/pipeline.py`, `notebooks/GretaVision_MVP_Colab.ipynb` |
| Integrante 3: Espinoza Alan | `feature/streamlit-demo`    | Interfaz, visualización, GitHub e integración        | `app_streamlit.py`, `requirements.txt`, `README.md`        |



## 8. Ramas de trabajo

El repositorio usa el siguiente flujo:

```text
main
dev
feature/streamlit-demo
feature/pipeline-metricas
feature/dataset-docs
docs/presentacion
```

Descripción:

* `main`: versión estable del proyecto.
* `dev`: rama de integración.
* `feature/streamlit-demo`: desarrollo de la interfaz y demo.
* `feature/pipeline-metricas`: mejoras del pipeline y métricas.
* `feature/dataset-docs`: dataset, pruebas, capturas y documentación.
* `docs/presentacion`: presentación e informe final.

Flujo recomendado:

```text
feature/... → dev → main
```

No se debe trabajar directamente sobre `main`.


## 9. Comandos básicos para colaboradores

Clonar repositorio:

```bash
git clone https://github.com/AlanEspinozaH/GretaVision-CompuGrafica.git
cd GretaVision-CompuGrafica
```

Cambiar a una rama de trabajo:

```bash
git checkout nombre-de-rama
```

Actualizar rama:

```bash
git pull origin nombre-de-rama
```

Guardar cambios:

```bash
git status
git add .
git commit -m "descripcion breve del cambio"
git push origin nombre-de-rama
```

Ejemplo:

```bash
git checkout feature/dataset-docs
git pull origin feature/dataset-docs
git add .
git commit -m "docs: agrega descripcion del dataset inicial"
git push origin feature/dataset-docs
```


## 10. Validación cualitativa realizada

Se realizó una validación cualitativa con imágenes organizadas en cuatro categorías:

| Categoría      | Descripción                                            | Objetivo de prueba                                             |
| -------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| `grieta_clara` | Imágenes con grietas visibles y buen contraste.        | Verificar detección de regiones alargadas asociadas a grietas. |
| `rugosa`       | Superficies con textura granular o irregular.          | Evaluar falsos positivos por textura.                          |
| `sin_grietas`  | Superficies sin grietas evidentes.                     | Verificar que el sistema no detecte componentes innecesarias.  |
| `sombras`      | Imágenes con sombras, manchas o iluminación irregular. | Evaluar sensibilidad a zonas oscuras no estructurales.         |


Los resultados generados se encuentran en:

```text
resultados/
```

La interpretación cualitativa se documenta en:
```text
resultados/resultados.md
```

En las pruebas realizadas, el sistema funcionó mejor en imágenes con grietas claras y buen contraste. En imágenes sin grietas, algunos casos no generaron componentes candidatas, lo cual es deseable. Las principales limitaciones se observaron en superficies rugosas y en imágenes con sombras, donde aparecieron falsos positivos por textura o iluminación.

No se reportan métricas como precisión, recall, Dice o IoU porque no se cuenta con máscaras manuales de referencia o ground truth pixel a pixel.

## 11. Estado final del MVP



Para la entrega final, el sistema permite:



* cargar una imagen de pavimento o concreto;

* ejecutar el pipeline de procesamiento;

* visualizar imagen original, imagen preprocesada, máscara, overlay y heatmap;

* modificar parámetros de preprocesamiento, segmentación, morfología y filtrado geométrico;

* mostrar tabla de métricas por región candidata;

* exportar overlay PNG, máscara PNG, heatmap PNG, métricas CSV y reporte JSON;

* trabajar con un dataset local clasificado en `data/input/`;

* documentar resultados cualitativos y limitaciones;

* ejecutar una demo funcional mediante Streamlit.



## 12. Fuera de alcance



Por restricciones de tiempo y alcance académico, quedan fuera del proyecto:



* diagnóstico estructural profesional;

* medición física en milímetros o centímetros sin calibración;

* reconstrucción 3D real;

* entrenamiento de modelos deep learning;

* segmentación supervisada con ground truth pixel a pixel;

* validación cuantitativa exhaustiva;

* cálculo de precisión, recall, Dice o IoU sin máscaras de referencia.



## 13. Limitaciones observadas



GretaVision segmenta regiones candidatas, no grietas garantizadas.



Las principales limitaciones observadas fueron:



* falsos positivos en superficies rugosas;

* sensibilidad a sombras o iluminación irregular;

* detección de manchas oscuras como posibles grietas;

* pérdida de grietas muy finas si los parámetros son demasiado restrictivos;

* dependencia de parámetros como `block_size`, `C`, filtro de mediana y área mínima.



Las métricas calculadas son aproximadas y se reportan en píxeles. La severidad visual es heurística y no representa severidad estructural.



## 14. Convención de commits



Se recomienda usar commits breves y descriptivos:



```text

feat: nueva funcionalidad

fix: corrección de error

docs: documentación

refactor: reorganización de código

test: pruebas o casos de evaluación

chore: configuración o cambios menores

```



Ejemplos:



```bash

git commit -m "feat: agrega filtrado geometrico de componentes conectados"

git commit -m "docs: documenta limitaciones del MVP"

git commit -m "fix: corrige visualizacion del mapa de calor"

```