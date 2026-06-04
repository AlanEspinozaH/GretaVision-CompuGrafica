# GretaVision Starter

MVP académico para segmentación, medición aproximada y visualización de grietas candidatas en imágenes de pavimento/concreto.

## Objetivo para avance

Mostrar:

1. Dataset inicial trabajado.
2. Pipeline funcional.
3. Scripts organizados.
4. Métricas exportables.
5. Propuesta final lista.

## Instalación local

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Ejecución rápida en Colab

Abrir `notebooks/GretaVision_MVP_Colab.ipynb`, ejecutar las celdas y subir una imagen.

## Estructura

```text
gretavision_starter/
├── app_streamlit.py
├── requirements.txt
├── src/
│   └── pipeline.py
├── notebooks/
│   └── GretaVision_MVP_Colab.ipynb
├── data/
│   └── input/
├── results/
└── docs/
```

## Advertencia académica

GretaVision detecta regiones candidatas a grietas y calcula métricas visuales aproximadas. No realiza diagnóstico estructural profesional.
