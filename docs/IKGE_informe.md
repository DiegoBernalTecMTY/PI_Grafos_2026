Tecnológico de Monterrey

Maestría en Inteligencia Artificial Aplicada 

Análisis de Grafos y Extrapolación de Conocimiento 
para identificar Relaciones No Explícitas de 
información

Informe Técnico Exhaustivo: Implementación, Análisis y Replicación del Modelo IKGE para Open-World KGC

**Fecha:** 7 de Marzo de 2026  
Equipo 32

José Adan Vega Pérez A01796093 

Silvia Xochitl Ibañez Vara A01795200 

Diego Andrés Bernal Díaz A01795975 
 

**Documentos de Referencia Internos:** 
* `[Ref 1: paper_code_correspondence.md]`
* `[Ref 2: dataset_dbpedia_vs_fb20k.md]`
* `[Ref 3: eval_log_explained.md]`
* `[Ref 4: fb20k_train_20260305_033928.log]`
* `[Ref 5: fb20k_eval_20260305_172717.log]`
* `[Ref 6: fb20k_triclf_20260305_180858.log]`
* `[Ref 7: train_20260305_002929.log]` (DBPedia50k+)

---

## 1. Introducción y Contexto del Proyecto

La completitud de grafos de conocimiento (Knowledge Graph Completion, KGC) tradicionalmente asume un entorno cerrado (*closed-world*), donde todas las entidades y relaciones son conocidas durante la fase de entrenamiento (modelos transductivos como TransE o ComplEx). Sin embargo, en escenarios reales, los grafos evolucionan constantemente. 

El modelo **Inductive Knowledge Graph Embedding (IKGE)** (Oh et al., 2022) propone una arquitectura inductiva de mundo abierto (*open-world*). En lugar de aprender un vector estático por entidad, IKGE aprende una función generadora que extrae características a partir de descripciones textuales y metadatos (nombres, restricciones de tipos), y posteriormente enriquece estas características mediante una red neuronal convolucional de grafos (GCN) basada en atención, agregando el contexto topológico a múltiples saltos (K-hops).

Durante esta semana, el equipo implementó IKGE desde cero. El proceso reveló discrepancias fundamentales entre la teoría del artículo y la práctica algorítmica. Este informe documenta la arquitectura implementada, las variaciones críticas aplicadas para evitar el colapso del gradiente, el descubrimiento del problema de dispersión topológica en los datasets originales y los resultados de evaluación sobre la variante FB20k+.

---

## 2. Correspondencia entre Teoría y Código (Implementación de la Arquitectura)

De acuerdo con nuestro análisis de mapeo `[Ref 1: paper_code_correspondence.md]`, la implementación se dividió en dos módulos principales: `FactFeatureExtractor` y `AttentiveAggregator`.

### 2.1. Extracción de Características de los Hechos (Fact Feature Extraction)
El artículo especifica que para cada relación $r$ y entidad $e$, se deben utilizar descripciones ($D_e$), tipos ($T_e$), y nombres ($U_e$). El cruce de atención bidireccional se implementó en `FactFeatureExtractor.py`:

*   **Codificación de Palabras (Word Encoding):** Replicamos la Ecuación 1 del paper utilizando Wikipedia2Vec (300 dimensiones, *enwiki_20180420*). Identificamos un error latente en la formulación teórica: el vocabulario debía ser compartido. En nuestra implementación inicial, las URI de las entidades y relaciones (ej. `dbr:Barack_Obama`) caían en el token `<UNK>`. Lo corregimos pre-procesando las cadenas de nombres para que compartieran el mismo espacio latente que las descripciones textuales `[Ref 1, Sec 5.1.1]`.
*   **Convolución Basada en Atención (Ecuaciones 1-4):** El paper enmascara la descripción de la entidad $h$ con el nombre de la relación $U_r$, la restricción de tipo **rango** $T_{r,r}$ y el nombre de la otra entidad $U_t$. Implementamos esto compartiendo los pesos $W_{c1}, W_{c2}, W_a$ y $W_p$ entre las llamadas del *head* y el *tail*, garantizando simetría matemática.
*   **Corrección del Promedio de Embeddings de Tipo:** Encontramos que el paper define la representación del tipo $w_r \in \mathbb{R}^{d \times 1}$ como un vector único. Al vectorizar esto en PyTorch con tensores acolchados (*padded tensors*), el promedio de los embeddings dividía erróneamente por posiciones de padding (que son ceros). Implementamos una máscara (*masked mean*) para promediar estrictamente los tokens válidos `[Ref 1, Sec 5.1.2]`.

### 2.2. Agregación Atenta de Características (Attentive Feature Aggregation - AFA)
El módulo AFA (Ecuaciones 6 a 11) se encarga de recorrer el "Grafo de Líneas" (*Line Graph*) donde los nodos son los *facts* y las aristas son entidades compartidas.

*   **Subgrafos K-hop BFS:** En `train_ikge_w2v.py`, construimos un muestreo BFS (Breadth-First Search) alrededor de la tripleta objetivo para construir el vecindario $\mathcal{N}(f_u)$. La tripleta a predecir se inserta dinámicamente como un **nodo virtual** durante el entrenamiento para evitar fugas de datos (*data leakage*), acoplándose perfectamente con la formulación de inferencia del paper.
*   **Corrección de la Dimensionalidad de Atención (Ecuación 8 vs Sec 5.2.4):** El texto del artículo incurre en una contradicción. La Ecuación 8 formula la puntuación de atención como una forma bilineal $f_v^T W_a^{k+1} f_u$ (donde $W_a$ debe ser una matriz $d \times d$), pero la Sección 5.2.4 afirma que los pesos son un vector en $\mathbb{R}^{2d}$. Optamos por implementar la matriz $d \times d$ `nn.Linear(d, d, bias=False)`, lo cual es matemáticamente congruente con la Ecuación 8 `[Ref 1, Sec 5.2.1]`.

---

## 3. Discrepancias Arquitectónicas y Soluciones de Ingeniería

Para lograr que el modelo convergiera, tuvimos que desviarnos de la formulación estricta del artículo en tres puntos fundamentales, detallados en nuestros registros `[Ref 1]` y `[Ref 7]`.

### 3.1. Puerta Suave (*Soft Floor Gate*) en el Type Matching
La Ecuación 5 del artículo dicta una compuerta lógica (multiplicación por 0 o 1) para el *Type Matching*: si los tipos de la entidad no contienen la restricción de dominio/rango de la relación exacta, la característica generada se vuelve un vector cero.
*   **El Problema:** La ontología de DBPedia y Freebase es jerárquica. Por ejemplo, `dbo:President` satisface la restricción `dbo:Person`, pero una intersección plana (*flat intersection*) de tensores arroja 0. Nuestros diagnósticos mostraron que **el 87.4% de los gradientes colapsaban a 0** en la primera época debido a esta formulación ingenua `[Ref 7]`.
*   **La Solución:** Implementamos una puerta suave o *soft floor*: `type_gate = 0.1 + 0.9 * type_validity`. Si no hay coincidencia directa, el modelo retiene un 10% de la señal original de la CNN. Esto permite el flujo del gradiente a través de las jerarquías ontológicas no explícitas. Adicionalmente, permitimos una validez de 1.0 si la entidad carece por completo de tipos registrados `[Ref 1, Sec 5.1.3]`.

### 3.2. Función de Pérdida: De BCE a Hinge Ranking Loss
El paper entrena usando *Binary Cross-Entropy* (BCE) (Ecuación 13). Al inicio, observamos que las predicciones colapsaban alrededor de un logit de 0 (sigmoide $\approx$ 0.5), provocando que los gradientes positivos y negativos se cancelaran mutuamente.
*   **La Solución:** Migramos a una **Hinge Margin Ranking Loss** con un margen de $0.5$. La pérdida penaliza al modelo solo si la diferencia entre la puntuación positiva y la negativa no supera el margen: $L = \max(0, margin - score_{pos} + score_{neg})$. Esto reactivó la dinámica de aprendizaje obligando al modelo a crear un "gap" de separabilidad `[Ref 4]`.

### 3.3. Estabilidad Numérica: LayerNorm y ReLU
Debido a las conexiones residuales en el módulo AFA ($\tilde{f}_u = h_{\mathcal{N}(f_u)}^{k+1} + f_u$), la magnitud de los vectores crecía exponencialmente tras K iteraciones, llevando las entradas del MLP a valores mayores a $\pm 15$ y saturando la función sigmoide.
*   Agregamos una capa `nn.LayerNorm(300)` justo antes de las capas totalmente conectadas (MLP).
*   Añadimos un `F.relu()` entre las dos convoluciones 1D de extracción de texto. Sin esto, el apilamiento de dos convoluciones colapsaba matemáticamente en una sola transformación lineal, perdiendo capacidad de representación `[Ref 1]`.

---

## 4. El Problema de Topología: De DBPedia50k+ a FB20k+

El avance más crucial de la semana ocurrió al analizar los resultados deficientes del dataset propuesto como estándar (DBPedia50k+). Nuestro reporte `[Ref 2: dataset_dbpedia_vs_fb20k.md]` documenta este análisis estructural.

### 4.1. La Dispersión de DBPedia50k+
Al replicar DBPedia50k+ a partir de los *dumps* originales de 2016, obtuvimos un MRR global de apenas **0.0589** tras 63 épocas `[Ref 7]`. 
*   **Métricas del Grafo:** El dataset contiene 49,900 entidades pero sólo 32,388 tripletas de entrenamiento.
*   **Diagnóstico:** Esto da un promedio de **2.7 *facts* por entidad**, y un grado de grafo de líneas de 52. Con un hiperparámetro de K=3 capas de agregación (sugerido por el paper), el BFS reciclaba continuamente los mismos 2 nodos. El módulo AFA no aportaba información de contexto rica, sino que inducía al modelo a memorizar atajos topológicos básicos que fallaban estrepitosamente ante entidades OOKG en el set de prueba.

### 4.2. La Densidad de FB20k+
Decidimos migrar el flujo de trabajo al dataset **FB20k+** (basado en Freebase FB15k). Reconstruimos las divisiones OOKG utilizando un optimizador de *Simulated Annealing* para coincidir exactamente con los conteos de evaluación del paper original `[Ref 2, Sec 3.2]`.
*   **Métricas del Grafo:** FB20k+ concentra **459,104 tripletas de entrenamiento** en tan solo 14,904 entidades In-KG.
*   **Diagnóstico:** Esto eleva el promedio a **61.7 *facts* por entidad** y un asombroso grado de grafo de líneas de **1,058.2** `[Ref 4]`. En este entorno hiper-denso, el modelo no puede memorizar estructuras discretas y se ve obligado a utilizar el codificador semántico (CNN sobre descripciones) para desenredar el vecindario. Además, optimizamos la profundidad de la red a **K=2**, lo cual fue suficiente para un grafo tan interconectado, reduciendo el tiempo de inferencia y la propagación de ruido.

El cambio de DBPedia50k+ a FB20k+ justificó inmediatamente los saltos métricos en nuestros experimentos, pasando de un MRR de 0.05 a 0.40.

---

## 5. Cierre de Brechas de Reproducibilidad (Reproducibility Gaps)

El artículo carece de detalles de ingeniería algorítmica. El documento `[Ref 1]` cataloga 16 brechas encontradas. Destacamos las de criticidad alta:

### 5.1. El "Atajo Estructural" en el Muestreo Negativo (Gap #2)
El artículo indica: *"generate negative triples by randomly replacing head or tail entity"*. No especifica la población candidata.
*   **El Error Inicial:** Muestrear aleatoriamente entre todas las 19k entidades (incluidas las Out-of-KG). Dado que las entidades OOKG tienen un subgrafo vacío por definición, el modelo aprendió rápidamente la regla trivial: "Si no tiene vecinos, es una tripleta falsa". El MRR en validación saltó a 0.99 en la época 3, pero el MRR en *full-ranking* real colapsó a 0.017 `[Ref 3]`.
*   **La Solución Implementada:** Restringimos el muestreo negativo **exclusivamente a entidades In-KG**. Además, construimos cubetas por restricciones de tipo (*type-constrained buckets*). Al inyectar como negativos a entidades In-KG que *comparten el mismo tipo* que la respuesta correcta, forzamos a la red a utilizar el texto de Wikipedia para discriminar verdaderos de falsos.

### 5.2. Hiperparámetros No Declarados (Gaps #1, #4)
*   **Batch Size:** No documentado. Asumimos `256` tras ensayos heurísticos de uso de memoria VRAM (aprox. 14GB en GPU).
*   **Duración del Entrenamiento:** Configuramos 200 épocas con *Cosine Annealing LR*, deteniéndonos en *Early Stopping* basado en la perdida de la particion de validacion durante entrenamiento. `[Ref 4]`.

---

## 6. Metodología de Evaluación: Filtrado Multinivel de 4 Capas (4-Tier Filtering)

La evaluación en escenarios *Open-World* es compleja. El artículo menciona que "las entidades candidatas cuyo cruce (relación-entidad) no exista en el conjunto de entrenamiento son filtradas" (Target Filtering). Sin embargo, esto es insuficiente para las tripletas donde todo el contexto es OOKG. En `[Ref 3: eval_log_explained.md]`, detallamos nuestra implementación de **Filtrado en Cascada de 4 Niveles (4-Tier Cascade)**:

| Nivel (Tier) | Identificador en Código | Cobertura | Candidatos Promedio | Uso Principal |
| :--- | :--- | :--- | :---: | :--- |
| **T1** | `pair_*_cands` | Cruce exacto `(h, r)` o `(r, t)` | ~3 a 4 | Transductivo clásico. |
| **T2** | `rel_*_cands` | Filtro por relación `r` únicamente | ~87 a 110 | Entidad OOKG, pero relación conocida. |
| **T3** | `ent_*_cands` | Filtro por entidad contexto | ~25 | Relación OOKG, pero entidad conocida. |
| **T4** | *Full Ranking* | Sin historial (Cold-Start) | 19,890 | Patrones severos donde no hay anclaje. |

Esta estructura nos permitió un **aceleramiento computacional de ~59x** en evaluación. Para el 85% de las entidades (T1-T3), solo ranqueamos un promedio de 336 candidatos en lugar del universo total de 19,890. El 15% restante (T4) se somete a un *Full Ranking* puramente basado en texto.

### 6.1. La Disparidad MRR: Población Filtrada vs OOKG
El análisis `[Ref 3]` explica matemáticamente por qué los promedios globales oscilan alrededor de MRR $\approx 0.35$.
*   Para la **Población T2/T3** (donde hay contexto de vecindario): MRR $\approx$ 0.35 a 0.42. El *Mean Rank* se sitúa en ~45.
*   Para la **Población T4 (OOKG puro)**: MRR $\approx$ 0.024. Aunque parece bajo, ubicar aleatoriamente un objetivo entre 19,890 opciones da un MRR de $\approx 0.0001$. Obtener 0.024 significa que el módulo de texto puro (sin grafo) es **~240 veces mejor que el azar**, demostrando una generalización *Zero-Shot* funcional basada en semántica.

---

## 7. Resultados Experimentales en FB20k+

Evaluamos el modelo a la **Época 29** del ciclo de entrenamiento (archivo de pesos `fb20k_ikge_w2v_best_mrr_20260305_033928.pt`). El análisis se desglosa en la clasificación en los 4 Grupos de Evaluación dictados por el artículo, probando las 85,557 tripletas de test en forma vectorizada sobre GPU `[Ref 5]`.

### 7.1. Link Prediction (Entity & Relation Ranking)

La siguiente tabla sintetiza la capacidad del modelo para adivinar el nodo (o relación) faltante, evaluado mediante *Mean Reciprocal Rank* (MRR) y *Mean Rank* (MR).

| Grupo de Evaluación del Paper | Patrones O/X Evaluados | N° Tripletas | MRR Obtenido | Mean Rank | Objetivo MRR (FB20k+) | Diagnóstico |
| :--- | :--- | :---: | :---: | :---: | :--- |:--- |
| **Group 1 - Head Pred.** | O-O-X, O-X-X, O-X-O | 13,105 | **0.3064** | 552.8 | **0.39** | 🟡 Sólido teniendo la psoibilidad de hacer mas epocas de entrenamiento . |
| **Group 2 - Tail Pred.** | X-O-O, X-X-O, O-X-O | 20,272 | **0.3771** | 165.7 | **0.40** |🟡 Sólido; la predicción de cola con texto rinde bien. |
| **Group 3 - Head+Tail OOK** | O-O-X (H), X-O-O (T) | 26,085 | **0.4053** | 49.0 | **0.42** | 🟡 Sólido; El muestreo negativo permite al modelo generalizar a entidades OOKG. |
| **Group 4 - Relation Pred.** | O-O-X, X-O-O, X-O-X | 26,095 | **0.0396** | 413.1 | **0.36** | 🔴 Subóptimo (Colapso por *Training Bug*). |
| **Promedio Global** | **Total ponderado** | **85,557** | **0.2719** | **264.9** | **0.3925** | El modelo está asimilando la semántica textual. |

*Nota sobre notación del paper original: (Head-Relation-Tail), donde O = In-KG (Conocido) y X = Out-of-KG (Desconocido).*

**El Colapso del Grupo 4 (Relation Prediction):**  
Como se documenta en `[Ref 3, Sec 6]`, el modelo puntúa MRR $\approx$ 0.04 en la predicción de relaciones, muy por debajo del ~0.36 esperado. Descubrimos que nuestro bucle de entrenamiento actual genera ejemplos negativos **únicamente corrompiendo las entidades** (Head/Tail). Al no haber reemplazado nunca la relación durante el cálculo del Hinge Loss, la capa clasificadora final (MLP) es agnóstica a la diferenciación de relaciones, provocando que todas las relaciones candidatas obtengan puntajes estadísticamente idénticos (desviación estándar $\approx$ 0.065). 

### 7.2. Clasificación Binaria de Tripletas (Triple Classification)

Adicional a las tablas del artículo original, diseñamos un experimento de Clasificación Binaria `eval_triple_classification.py` `[Ref 6]`. Tras calibrar el umbral óptimo en el set de validación segun sos puntuaciones durante el entrenamiento ($\tau = 0.38$), probamos un set balanceado (1:1 Verdadero/Falso con negativos difíciles) de 88,142 tripletas:

| Segmento de Inferencia | Precisión (Acc) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **in_KG** (Todo conocido) | 0.8525 | 0.8128 | 0.9159 | **0.8613** |
| **out_T** (Cola desconocida) | 0.7881 | 0.8028 | 0.7637 | **0.7828** |
| **out_H** (Cabeza desconocida) | 0.7694 | 0.7800 | 0.7506 | **0.7650** |
| **out_RT** (Cola y Rel desconocidos) | 0.6891 | 0.8327 | 0.4733 | **0.6035** |
| **Promedio MICRO Global** | **0.8236** | **0.8066** | **0.8514** | **0.8284** |

Este resultado es excepcional y valida nuestra hipótesis semántica. El modelo alcanzó un **Micro F1 de 82.8%** en la clasificación pura utilizando las representaciones de los CNN extrayendo contexto de Wikipedia. Incluso en el escenario extremo de `out_RT` (donde la entidad receptora Y la relación misma son alienígenas para la red), el modelo retiene un F1 superior al 60%, evitando caer en conjeturas completamente aleatorias (50%).

---

## 8. Conclusiones y Plan de Trabajo Futuro

### 8.1. Conclusiones
1. **La Topología de Grafo es Precondición Indispensable:** Modelos de agregación topológica como IKGE fallan inherentemente en grafos dispersos. Replicar el modelo sobre DBPedia50k+ fue un ejercicio instructivo pero inútil en métricas reales. La migración a la hiper-densidad de FB20k+ destrabó las capacidades de la red neuronal convolucional en grafos (GCN).
2. **Robustez Zero-Shot Validada:** Resolviendo las múltiples brechas de reproducibilidad (destacando el muestreo negativo desde `type-constrained buckets` exclusivos de In-KG), conseguimos un modelo de KGC de mundo abierto que retiene la capacidad de clasificar hechos verídicos para nodos fuera de vocabulario (OOKG), multiplicando por ~240 el rendimiento esperado del azar.
3. **Mejoras Arquitectónicas Propias:** Nuestra adopción del Hinge Loss y el Soft Gate para ontologías jerárquicas son aportes originales que estabilizaron sustancialmente las derivadas del modelo frente a la propuesta original del *paper*.

### 8.2. Siguientes Pasos de recreacion del paper y correspondencia
De cara a culminar la experimentación, delineamos los siguientes pasos:

1. **Parchear el "Training Bug" del Grupo 4:** Intervenir la función `generate_neg_indices()` en el ciclo de entrenamiento `train_ikge_w2v.py` para aplicar *Relation Corruption*. Al sustituir la relación original por una relación aleatoria en el $\sim 33\%$ de las tripletas negativas, inyectaremos el gradiente necesario para que el MLP aprenda a rankear relaciones, esperando que el MRR del Grupo 4 salte de 0.04 a $\approx$ 0.36.
2. **Entrenamiento Continuo hasta la Época 200:** Los resultados actuales de MRR $\sim 0.40$ se consiguieron con la ejecución detenida prematuramente en la época 30. Desplegar una ejecución ininterrumpida de ~60 horas en infraestructura GPU dedicada para que la convergencia del vector logre exprimir el potencial del Hinge Loss.
3. **Estudios de Ablación Formales:** Emular las variantes del paper (IKGE$_{No\_AFA}$, IKGE$_{No\_ATT}$) simplemente apagando los tensores de agregación durante el script de evaluación. Esto proporcionará la medición cuantitativa exacta de cuánto rendimiento se extrae del texto y cuánto del mapeo topológico.



## 9. Propuesta de Implementación: Descubrimiento de Nuevas Relaciones en el Grafo Original (In-KG Link Discovery)

**Objetivo:** Utilizar el modelo IKGE entrenado para minar el grafo de conocimiento original (ej. FB20k+) y descubrir tripletas $(h, r, t)$ que son semántica y topológicamente válidas, pero que no estaban explícitamente documentadas en el dataset original.

A diferencia de la predicción *Open-World* (evaluada en nuestros logs como G1, G2 y G3), este es un problema de **Mundo Cerrado (Closed-World)**, ya que tanto la cabeza ($h$) como la cola ($t$) y la relación ($r$) existen en el vocabulario de entrenamiento. IKGE es excepcionalmente bueno para esto porque, al ser In-KG, el módulo AFA (Agregación Atenta) cuenta con vecindarios hiper-densos (1058 vecinos en FB20k+, promedio de facts por entidad de 67) para triangular la información, sumado al contexto rico de las descripciones de Wikipedia.

A continuación, se detalla el *pipeline* de implementación en 4 fases.

---

### 9.1: Prerrequisito Crítico (El Parche de Corrupción de Relaciones)

Como se identificó en nuestros logs (`[Ref 3: eval_log_explained.md]`), el Grupo 4 (Relation Prediction) actualmente sufre de un colapso en la distribución de puntajes (MRR de 0.039) porque el modelo nunca fue entrenado para discriminar relaciones falsas.

**Acción Requerida:** 
Antes de minar nuevas relaciones, debemos parchear la función `generate_neg_indices()` en `train_ikge_w2v.py`. 
1. Durante el cálculo del *Hinge Loss*, en el 33% de los casos (por ejemplo), en lugar de corromper la entidad, mantendremos $(h, t)$ intactos y reemplazaremos $r$ por una relación aleatoria $r_{falsa}$.
2. Entrenar el modelo hasta la convergencia (época 200). Esto forzará al MLP final a aprender las sutilezas que diferencian una relación de otra dados dos nodos correctos, preparando al modelo para puntuar relaciones con alta confianza.

---

### 9.2: Generación del Espacio de Búsqueda (Candidate Generation)

Si intentamos evaluar todas las combinaciones posibles en FB20k+ ($14,904 \text{ entidades} \times 1,341 \text{ relaciones} \times 14,904 \text{ entidades}$), tendríamos casi **300 mil millones de combinaciones**, lo cual es computacionalmente inviable. Necesitamos una heurística de filtrado para generar candidatos plausibles.

**Implementación del Filtrado:**
Aprovecharemos las tablas que ya construimos en memoria y nuestro módulo de *Type Matching*:

1. **Filtrado Topológico (Proximidad):** En un KG, las relaciones ocultas suelen existir entre entidades que ya están conectadas a 2 o 3 saltos de distancia (K-hops). Podemos usar nuestra matriz de adyacencia del grafo de líneas para extraer pares de entidades $(h, t)$ que comparten vecinos pero no tienen una arista directa.
2. **Filtrado por Ontología (Type-Constrained Buckets):** Utilizaremos las cubetas de restricciones de tipo que ya creamos para el muestreo negativo (`rel_head_type_ents` y `rel_tail_type_ents`). 
   * Para una relación $r$ específica (ej. `director_de_cine`), solo generaremos pares donde $h$ pertenezca a los tipos válidos para el dominio (ej. `Persona`) y $t$ a los tipos válidos para el rango (ej. `Película`). 
   * Esto reduce el espacio de búsqueda de $300 \times 10^9$ a solo unos pocos millones de candidatos semánticamente lógicos.

---

### 9.3: Inferencia y Puntuación a Gran Escala (Batch Scoring)

Aprovecharemos la vectorización en GPU que ya desarrollamos para la función `_score_flat_gpu()`. 

1. **Agrupación en Lotes (Batching):** Pasamos los millones de tripletas candidatas generadas en la Fase 2 a través del modelo IKGE.
2. **Procesamiento Dual (Texto + Grafo):**
   * El `FactFeatureExtractor` codificará la compatibilidad semántica leyendo las descripciones de Wikipedia de $h$ y $t$, atendiendo a la relación $r$.
   * El `AttentiveAggregator` recopilará el contexto de los vecinos de $h$ y $t$. Si $h$ (un director) y $t$ (un actor de la película) comparten muchas conexiones de la industria en el grafo, el módulo AFA amplificará enormemente el vector resultante.
3. **Puntuación (Scoring):** El MLP final emitirá un `logit`, al cual le aplicaremos la función Sigmoide para obtener una probabilidad entre $0.0$ y $1.0$.

---

### 9.4: Extracción de Conocimiento y Aplicación de Umbrales

Aquí es donde reutilizamos el trabajo de nuestro script de clasificación binaria (`eval_triple_classification.py`).

1. **Filtrado de Hechos Conocidos:** Cruzamos los resultados contra nuestro diccionario `train_triples` (el KG original). Cualquier tripleta candidata que ya exista en el dataset original se descarta. Solo nos interesan los "nuevos descubrimientos".
2. **Aplicación del Umbral de Alta Precisión (High-Precision Thresholding):**
   * En nuestros logs `[Ref 6]`, el umbral óptimo para maximizar el F1-Score fue $\tau = 0.38$.
   * Sin embargo, para descubrimiento de conocimiento en el mundo real, queremos minimizar los Falsos Positivos. Podemos establecer un umbral mucho más estricto, calibrado en base a la distribución de puntuaciones. Por ejemplo, exigir un $\tau \geq 0.85$ o $\tau \geq 0.90$.
3. **Salida (Output):** El modelo escupe una lista ordenada de tripletas $(h, r, t)$ no vistas previamente, ordenadas por su nivel de confianza (probabilidad). 

---

### Ejemplo Teórico del Flujo

1. **Entidades en el grafo:** `h = Quentin_Tarantino`, `t = Pulp_Fiction`. 
2. **Estado actual:** En nuestro KG de entrenamiento, sabemos que `Tarantino` escribió `Pulp_Fiction`, pero falta la relación de que la *dirigió*.
3. **Fase 2 (Generación):** El algoritmo nota que están a 1 salto de distancia y que cumplen con los tipos `Persona` y `Película` para la relación `dirigida_por`.
4. **Fase 3 (Scoring):** El texto de Wikipedia de Tarantino menciona "Pulp Fiction" y la CNN lo detecta. El módulo AFA detecta que comparten actores en el grafo. El modelo asigna a la tripleta `(Quentin_Tarantino, dirigida_por, Pulp_Fiction)` un puntaje de **0.94**.
5. **Fase 4 (Extracción):** Como 0.94 > 0.85 y la tripleta no estaba en el dataset original de entrenamiento, el script la exporta como un **Nuevo Hecho Descubierto**.

### Conclusión de la Propuesta
Esta implementación transforma a IKGE en un **Motor de Minería de Grafos (Graph Mining Engine)**. Al combinar nuestra *Soft Gate* para tipos, el poder de los embeddings de texto, y la densidad estructural de FB20k+, creemos que el modelo tendra la capacidad real de autocompletar KGs empresariales detectando vínculos que los curadores humanos no encuentren al momento de crear el grafo.

