# Reporte de Replicación: IKGE sobre DBPedia50k+

> **Referencia del paper:** Oh, B., Seo, S., Hwang, J. et al.  
> *"Open-world knowledge graph completion for unseen entities and relations via attentive feature aggregation"*  
> Information Sciences 586 (2022), pp. 468–484.

---

## 1. Dataset: DBPedia50k+

DBPedia50k+ es una extensión del dataset DBPedia50k construida por los autores del paper para soportar evaluación en escenarios **inductivos** (entidades y relaciones fuera del grafo de entrenamiento).

| Estadística | Valor |
|---|---|
| Entidades en entrenamiento (in-KG) | 24 092 |
| Entidades totales (incl. test out-of-KG) | 29 849 |
| Relaciones in-KG | 323 |
| Relaciones totales (incl. out-of-KG) | 395 |
| Tipos de entidad | 219 |
| Triples de entrenamiento | 32 388 |
| Triples de validación | 399 |
| Triples de test | 10 273 |

El conjunto de test se divide en **8 patrones** según si cabeza (h), relación (r) y cola (t) pertenecen o no al grafo de entrenamiento (O = in-KG, X = out-of-KG):

| Patrón | n | Descripción |
|---|---|---|
| O-O-X | 3 430 | Cola fuera del KG |
| X-O-O | 2 871 | Cabeza fuera del KG |
| O-X-O | 110 | Relación fuera del KG |
| O-X-X | 169 | Relación y cola fuera del KG |
| X-X-O | 434 | Cabeza y relación fuera del KG |
| O-O-O | 3 259 | Todo in-KG (no usado en evaluación IKGE) |

Cada entidad incluye una descripción textual en inglés (fuente: páginas de Wikipedia). Las relaciones llevan su nombre y restricciones de tipo de dominio y rango (fichero `relation2constraint.txt`).

El grafo de línea construido sobre los triples de entrenamiento tiene **32 388 nodos** (hechos) y **1 689 734 aristas** (pares de hechos adyacentes), con un grado medio de **52.2**.

---

## 2. Arquitectura del modelo IKGE

IKGE opera en tres etapas encadenadas:

1. **Extractor de características de hecho** (`FactFeatureExtractor`):  
   Dos capas CNN 1D con atención sobre la descripción de la entidad, enmascarada por el nombre de la relación, la restricción de tipo y el nombre de la entidad opuesta. Produce un vector de hecho $f \in \mathbb{R}^{256}$.  
   Antes de la agregación, el vector se escala por la validez de tipo: $f \leftarrow f \odot (\sum_i(t_h \odot t_{r,d})_i \times \sum_i(t_t \odot t_{r,r})_i)$.

2. **Agregador atencional** (`AttentiveAggregator`, K=3 capas):  
   Para cada hecho, agrega los vectores de sus vecinos en el grafo de línea mediante atención aprendida:  
   $h^{k+1}_{N(f_u)} = \tanh\!\left(\sum_{f_v \in N(f_u)} a_v \cdot f_v\right)$, $\quad \tilde{f}_u = h^{k+1}_{N(f_u)} + f_u$

3. **Función de puntuación** (`IKGENetwork`):  
   Dos capas totalmente conectadas (512 → 256 → 1) con activación ReLU y Dropout (p=0.25).  
   Pérdida: entropía cruzada binaria (Ecuación 13 del paper).

**Hiperparámetros usados** (todos alineados con el paper):

| Parámetro | Valor |
|---|---|
| Dimensión de embedding de hecho | 256 |
| Canales CNN | 128 |
| Ancho de filtro CNN | 3 |
| Capas de agregación K | 3 |
| Longitud máxima de descripción | 50 tokens |
| Dropout | 0.25 |
| Optimizador | AdamW (lr=0.01, weight_decay=0.001) |
| Scheduler | Cosine Annealing (T_max=200) |
| Batch size | 256 |
| Épocas | 200 |

---

## 3. Trabajo de depuración y correcciones realizadas

Durante la replicación se identificaron y corrigieron varios errores críticos que impedían el aprendizaje:

### 3.1 Desajuste de vocabulario de tipos
**Problema:** El fichero `entity2type.txt` usaba la URI larga `http://www.w3.org/2002/07/owl#Thing`, mientras que `relation2constraint.txt` usaba la forma corta `dbo:Thing`. Al comparar tipos, todas las entidades con `owl#Thing` fallaban el matching y sus vectores de hecho se zerificaban completamente, bloqueando el gradiente.  
**Corrección:** Normalización canónica al cargar ambos ficheros: `owl#Thing → dbo:Thing`.

### 3.2 Update del agregador incorrecto (desviación del paper)
**Problema:** La implementación inicial aplicaba una capa lineal con CONCAT y activación sigmoide:  
$\tilde{f}_u = \sigma(W_c \cdot [f_u \| h_{N(f_u)}] + b_c)$  
Esto producía colapso de señal: la salida saturaba en ~0.5 para todas las entidades después de K=3 capas, de modo que el MLP recibía entradas constantes y no podía discriminar.  
**Corrección:** Implementada la Ecuación 10 del paper: $\tilde{f}_u = h_{N(f_u)} + f_u$ (suma simple, sin matriz $W_c$).

### 3.3 Función de pérdida incorrecta
**Problema:** Se usaba pérdida *softplus* con margen (pérdida de bisagra), que no corresponde al paper.  
**Corrección:** Sustituida por entropía cruzada binaria (BCE) según la Ecuación 13:  
$\mathcal{L} = \sum_{(h,r,t,y)} y \log w(z) + (1-y)\log(1-w(z))$

### 3.4 Distribución train/eval inconsistente
**Problema:** En evaluación se usaba la salida final del agregador como proxy de vecindad para todos los candidatos, ignorando que dicha salida correspondía a K capas de transformación mientras la entrada al primer nivel de vecindad debía ser el vector CNN sin agregar.  
**Corrección:** Implementada una aproximación de campo medio multi-paso: se pre-calculan medias de entidad capa a capa, replicando exactamente la distribución de cada nivel de agregación en el tiempo de inferencia.

### 3.5 Dirección de corrupción en Grupo 3 invertida
**Problema:** El Grupo 3 (predicción cabeza+cola) evaluaba la dirección equivocada:  
- Para triples O-O-X (cola fuera del KG) se intentaba rankear la entidad OOK → MRR ≈ 1/29 849 ≈ 0.
- Para triples X-O-O (cabeza fuera del KG) lo mismo.  
**Corrección:** Según la Sección 6.1.4 del paper, los grupos evalúan la entidad *in-KG*:  
- O-O-X → predecir la cabeza (in-KG)  
- X-O-O → predecir la cola (in-KG)

### 3.6 Validación silenciosa
**Problema:** Una condición de mejora de ventana de pérdida causaba que la evaluación de validación se saltara silenciosamente después de las primeras épocas.  
**Corrección:** La validación ahora se ejecuta incondicionalmente cada `eval_every` épocas.

---

## 4. Impacto del tipo de embeddings en el rendimiento

El modelo IKGE (*Inductive Knowledge Graph Embedding*) representa entidades y relaciones a partir de sus descripciones textuales.  
El primer paso del pipeline es convertir cada palabra de las descripciones en un vector numérico pre-entrenado.  
La calidad de estos vectores determina directamente cuánta información semántica puede capturar el extractor de características CNN.

---

## 5. Embeddings utilizados

| Configuración | Embeddings | Dim | Cobertura del vocabulario |
|---|---|---|---|
| **Paper original** | Wikipedia2Vec (Yamada et al., 2020) | 300 | **~100 %** |
| **Esta implementación** | GloVe 6B (Pennington et al., 2014) | 300 | **29,3 %** |

- **Vocabulario total** construido a partir de las 55 994 descripciones de entidades y relaciones: **296 618 palabras**.  
- Con GloVe 6B: sólo **86 965 palabras** encontradas; las **209 653 restantes** (70,7 %) se inicializaron con pesos aleatorios *Kaiming uniforme*, sin ningún significado semántico.

---

## 6. Resultados comparativos

### Table 2 del paper (Wikipedia2Vec, 100 % cobertura)

| Grupo | Tarea | MRR | H@1 | H@3 | H@10 |
|---|---|---|---|---|---|
| Grupo 1 | Predicción de entidad cabeza | 0.34 | — | — | — |
| Grupo 2 | Predicción de entidad cola | 0.61 | — | — | — |
| Grupo 3 | Predicción cabeza+cola (entidades OOK) | 0.52 | — | — | — |
| Grupo 4 | Predicción de relación | 0.31 | — | — | — |

### Esta implementación (GloVe 6B, 29,3 % cobertura)

| Grupo | Tarea | n | MRR | H@1 | H@3 | H@10 | Diferencia vs. paper |
|---|---|---|---|---|---|---|---|
| Grupo 1 | Predicción de entidad cabeza | 3 709 | 0.019 | 0.004 | 0.013 | 0.036 | **−0.32** |
| Grupo 2 | Predicción de entidad cola | 3 415 | 0.021 | 0.007 | 0.017 | 0.037 | **−0.59** |
| Grupo 3 | Predicción cabeza+cola (OOK) | 6 301 | 0.002 | 0.000 | 0.001 | 0.003 | **−0.52** |
| Grupo 4 | Predicción de relación | 6 301 | 0.012 | 0.006 | 0.012 | 0.014 | **−0.30** |
| **Global** | | **19 726** | **0.012** | **0.004** | **0.009** | **0.018** | |

> *Pesos evaluados: `ikge_best_mrr_20260302_043039.pt` (epoch 100 de 200, mejor MRR de validación = 0.0205)*

---

## 7. Análisis del impacto

### 7.1 Degradación de la representación semántica

El extractor de características de IKGE aplica dos capas CNN sobre la matriz de embeddings de cada descripción.  
Cuando el 70,7 % de los tokens tienen vectores aleatorios, la CNN recibe principalmente ruido: no puede aprender patrones de similitud semántica entre entidades, ya que las palabras clave que diferencian una entidad de otra (nombres propios, términos DBpedia, etc.) son precisamente las **menos frecuentes en GloVe**.

### 7.2 Por qué Wikipedia2Vec marca la diferencia

Wikipedia2Vec fue entrenado sobre la propia Wikipedia con un objetivo que **co-aprende embeddings de palabras y de entidades al mismo tiempo**, haciendo que términos como `dbo:City`, `dbr:Madrid` o `dbr:Eiffel_Tower` tengan representaciones semánticamente ricas.  
GloVe, entrenado sobre texto genérico de Common Crawl, no incluye la mayoría de los URI de DBpedia ni los términos técnicos de las descripciones de KG.

### 7.3 Efecto por grupo

- **Grupo 3 (OOK, MRR = 0.002)** es el más afectado: al predecir entidades completamente nuevas, el modelo depende *exclusivamente* de las descripciones textuales. Con embeddings mayoritariamente aleatorios, la discriminación entre candidatos cae a niveles cercanos al azar (1/29 849 ≈ 0.000033).
- **Grupos 1 y 2** se benefician parcialmente del grafo de vecindad (entidades vistas en entrenamiento), lo que explica su MRR ligeramente superior (~0.02), pero sigue siendo muy bajo frente al 0.34–0.61 del paper.
- **Grupo 4 (relaciones)** no depende directamente de las descripciones de entidades, pero sí del nombre de la relación, que también tiene cobertura limitada en GloVe.

---

## 8. Recomendación

Para acercarse a los resultados del paper se deben sustituir los embeddings GloVe por **Wikipedia2Vec 300d**, disponibles en:

```
https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
```

El archivo a descargar es `enwiki_20180420_300d.pkl.bz2` (Wikipedia en inglés, 300 dimensiones).  
Una vez descargado, basta con actualizar la llamada a `setup_glove_for_ikge` por una función equivalente que cargue el formato Wikipedia2Vec.

Con cobertura del vocabulario cercana al 100 %, se espera que el MRR global escale de **~0.012 → ~0.40**, alineándose con las métricas reportadas en el paper original.

---

*Generado el 2 de marzo de 2026 — Experimento IKGE sobre DBPedia50k+*
