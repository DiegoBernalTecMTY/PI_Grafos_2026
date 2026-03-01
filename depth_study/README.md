Depth Study – InGram (NL-25, WK-25, FB-25)
Descripción del experimento

En esta carpeta se incluyen los resultados del estudio de profundidad estructural realizado sobre el modelo InGram en escenarios inductivos.

El objetivo fue analizar el impacto del número de capas de propagación estructural en:

el módulo de entidades (Le)

el módulo de relaciones (Lr)

En cada bloque experimental se varió uno de los hiperparámetros (Le o Lr) mientras se mantuvo fijo el otro en el valor por defecto L = 2.

Se evaluaron tres datasets en configuración 25% new links:

NL-25 (derivado de NELL)

WK-25 (derivado de WordNet)

FB-25 (derivado de Freebase)

Configuración experimental

Para cada dataset se ejecutaron dos tipos de experimentos:

Entity Depth Study

Le ∈ {1, 2, 3, 4}

Lr = 2

Relation Depth Study

Lr ∈ {1, 2, 3, 4}

Le = 2

Las métricas reportadas en los archivos .txt son:

MRR (Mean Reciprocal Rank)

Hits@10

Hits@1

Cada archivo contiene la salida directa del modelo al finalizar entrenamiento y evaluación.

Objetivo del análisis

El propósito del estudio fue determinar si una mayor profundidad estructural mejora sistemáticamente el desempeño inductivo, o si existe un punto óptimo de agregación contextual.

Los resultados permiten analizar:

La estabilidad del mejor valor de Le entre distintos dominios.

La variabilidad del mejor valor de Lr dependiendo de la estructura relacional del dataset.

Posibles efectos de sobre-agregación (over-smoothing) al incrementar excesivamente la profundidad.

Un análisis cualitativo detallado se desarrolla en el documento principal del proyecto.

Interpretación general (resumen)

De manera preliminar, se observa que:

Le = 2 tiende a ser el valor más estable entre datasets.

El valor óptimo de Lr depende del dominio y la naturaleza estructural del grafo.

Profundidades mayores no garantizan mejoras y pueden degradar el desempeño.

Estos resultados sugieren que la profundidad estructural debe calibrarse de manera diferenciada para entidades y relaciones en escenarios inductivos.