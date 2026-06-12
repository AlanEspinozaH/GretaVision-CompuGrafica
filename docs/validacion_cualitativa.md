# Validación cualitativa sugerida

Usa entre 10 y 20 imágenes separadas por caso:

| Caso | Cantidad mínima | Qué observar | Captura requerida |
|---|---:|---|---|
| Grietas claras | 3 | Si la máscara sigue la grieta principal | Original + overlay |
| Textura rugosa | 3 | Falsos positivos por textura | Original + máscara |
| Iluminación irregular | 3 | Sensibilidad a sombras | Original + overlay |
| Manchas/juntas | 3 | Confusión con objetos no grieta | Original + máscara |
| Sin grietas | 2 | Falsos positivos | Original + máscara |

Conclusión sugerida: reportar aciertos, falsos positivos, falsos negativos y limitaciones. No reportar precisión/recall/IoU salvo que tengan máscaras manuales de referencia.
