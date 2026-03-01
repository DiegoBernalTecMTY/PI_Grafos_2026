# Depth Study – InGram (NL-25, WK-25, FB-25)

## Descripción del experimento

En esta carpeta se incluyen los resultados del estudio de profundidad estructural realizado sobre el modelo **InGram** en escenarios inductivos.

El objetivo fue analizar el impacto del número de capas de propagación estructural en:

- el módulo de entidades (`Le`)
- el módulo de relaciones (`Lr`)

En cada bloque experimental se varió uno de los hiperparámetros (`Le` o `Lr`) mientras se mantuvo fijo el otro en el valor por defecto `L = 2`.

Se evaluaron tres datasets en configuración **25% new links**:

- **NL-25** (derivado de NELL)
- **WK-25** (derivado de WordNet)
- **FB-25** (derivado de Freebase)

---

## Configuración experimental

Para cada dataset se ejecutaron dos tipos de experimentos:

### 1. Entity Depth Study
- `Le ∈ {1, 2, 3, 4}`
- `Lr = 2`

### 2. Relation Depth Study
- `Lr ∈ {1, 2, 3, 4}`
- `Le = 2`

---

## Métricas reportadas

Los archivos `.txt` contienen la salida directa del modelo al finalizar entrenamiento y evaluación.

Las métricas incluidas son:

- **MRR** (Mean Reciprocal Rank)
- **Hits@10**
- **Hits@1**

---

## Objetivo del análisis

El propósito del estudio fue determinar si una mayor profundidad estructural mejora sistemáticamente el desempeño inductivo, o si existe un punto óptimo de agregación contextual.

El análisis permite examinar:

- La estabilidad del mejor valor de `Le` entre distintos dominios.
- La variabilidad del mejor valor de `Lr` según la estructura relacional del dataset.
- Posibles efectos de **over-smoothing** al incrementar excesivamente la profundidad.

---

## Interpretación general (resumen)

De forma preliminar, se observa que:

- `Le = 2` tiende a ser el valor más estable entre datasets.
- El valor óptimo de `Lr` depende del dominio y la naturaleza estructural del grafo.
- Profundidades mayores no garantizan mejoras y pueden degradar el desempeño.