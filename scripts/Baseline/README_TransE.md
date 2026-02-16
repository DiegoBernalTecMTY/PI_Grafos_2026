# TransE: Translating Embeddings for Modeling Multi-relational Data

## Implementación Fiel del Paper Original (Bordes et al., 2013)

Esta implementación sigue meticulosamente el paper original "Translating Embeddings for Modeling Multi-relational Data" publicado en NIPS 2013.

---

## 📚 Relación con el Paper

### 1. Fundamento Teórico (Sección 2 del Paper)

**Idea Central:**
```
Si existe la relación (head, label, tail), entonces:
    h + r ≈ t
```

Donde:
- `h`: Embedding de la entidad head
- `r`: Embedding de la relación
- `t`: Embedding de la entidad tail

**Intuición Geométrica:**
Las relaciones son traslaciones en el espacio de embeddings. Por ejemplo:
- `Paris + capital_of ≈ France`
- `Entity + hypernym ≈ Parent_Entity`

### 2. Función de Energía

**Paper (Página 3):**
```
d(h, r, t) = ||h + r - t||_p
```

Donde `p` puede ser:
- `p=1` (norma L1): Distancia Manhattan
- `p=2` (norma L2): Distancia Euclidiana

**Implementación:**
```python
def score_triples(self, heads, relations, tails):
    h_emb, r_emb, t_emb = self.get_embeddings(heads, relations, tails)
    translation = h_emb + r_emb - t_emb
    distance = torch.norm(translation, p=self.norm_order, dim=1)
    return -distance  # Negativo porque menor distancia = mejor
```

### 3. Loss Function - Margin-based Ranking (Ecuación 1)

**Paper:**
```
L = Σ_(h,r,t)∈S Σ_(h',r,t')∈S' [γ + d(h,r,t) - d(h',r,t')]_+
```

Donde:
- `[x]_+ = max(0, x)`: Parte positiva
- `γ`: Margen (hiperparámetro)
- `S`: Conjunto de tripletas verdaderas
- `S'`: Conjunto de tripletas corruptas

**Implementación:**
```python
def forward(self, pos_heads, pos_rels, pos_tails, neg_heads, neg_rels, neg_tails):
    pos_scores = self.score_triples(pos_heads, pos_rels, pos_tails)
    neg_scores = self.score_triples(neg_heads, neg_rels, neg_tails)
    loss = torch.relu(self.margin - pos_scores + neg_scores).mean()
    return loss
```

### 4. Negative Sampling (Ecuación 2)

**Paper:**
```
S'_(h,r,t) = {(h', r, t) | h' ∈ E} ∪ {(h, r, t') | t' ∈ E}
```

Estrategia:
- Para cada tripleta positiva, generar UNA tripleta corrupta
- Corromper SOLO el head O el tail (no ambos)
- Selección aleatoria entre corromper head o tail

**Implementación:**
```python
def corrupt_batch(pos_triples, num_entities, device):
    neg_triples = pos_triples.clone()
    corrupt_head_mask = torch.rand(batch_size, device=device) < 0.5
    random_entities = torch.randint(0, num_entities, (batch_size,), device=device)
    neg_triples[corrupt_head_mask, 0] = random_entities[corrupt_head_mask]
    neg_triples[~corrupt_head_mask, 2] = random_entities[~corrupt_head_mask]
    return neg_triples
```

### 5. Algoritmo de Entrenamiento (Algoritmo 1)

**Paper - Pasos Clave:**

1. **Inicialización (Líneas 1-3):**
   ```
   - Relaciones: uniform(-√(6/k), √(6/k))
   - Entidades: uniform(-√(6/k), √(6/k))
   ```
   Usa la inicialización de Glorot & Bengio (2010) - referencia [4]

2. **Normalización de Relaciones (Línea 2):**
   ```
   r ← r/||r|| para cada relación r
   ```
   ⚠️ SOLO en inicialización, NO durante entrenamiento

3. **Loop Principal (Líneas 4-13):**
   ```
   Para cada época:
       a) Normalizar entidades: e ← e/||e|| (Línea 5)
       b) Samplear minibatch (Línea 6)
       c) Generar negativos (Línea 9)
       d) Actualizar con SGD (Línea 12)
   ```

**Implementación:**
```python
def train_transe(model, train_data, ...):
    for epoch in range(num_epochs):
        # (a) Normalizar entidades ANTES del batch
        model.normalize_entity_embeddings()
        
        for pos_batch in train_loader:
            # (b) Batch ya viene del loader
            # (c) Generar negativos
            neg_batch = corrupt_batch(pos_batch, num_entities, device)
            
            # (d) Forward + Backward
            loss = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 6. Restricciones de Normalización

**¿Por qué normalizar?**

Del paper (Página 3):
> "This constraint is important for our model, as it is for previous embedding-based 
> methods, because it prevents the training process to trivially minimize L by 
> artificially increasing entity embeddings norms."

Sin normalización, el modelo podría hacer trampa:
- Aumentar infinitamente las normas de los embeddings
- La loss disminuiría artificialmente sin aprender nada útil

**Restricciones:**
- ✅ Entidades: ||e|| = 1 (normalizar CADA época)
- ❌ Relaciones: SIN restricción después de inicialización

**Implementación:**
```python
def normalize_entity_embeddings(self):
    with torch.no_grad():
        self.entity_embeddings.weight.data = nn.functional.normalize(
            self.entity_embeddings.weight.data, p=2, dim=1
        )
```

---

## 🔧 Hiperparámetros del Paper

### Tabla de Configuraciones Óptimas (Sección 4.2)

| Dataset | k (dim) | λ (lr) | γ (margin) | d (norm) |
|---------|---------|--------|------------|----------|
| WN      | 20      | 0.01   | 2          | L1       |
| FB15k   | 50      | 0.01   | 1          | L1       |
| FB1M    | 50      | 0.01   | 1          | L2       |

### Búsqueda de Hiperparámetros

Del paper:
> "We selected the learning rate λ among {0.001, 0.01, 0.1}, the margin γ 
> among {1, 2, 10} and the latent dimension k among {20, 50} on the 
> validation set of each data set."

**Implementación:**
```python
EMBEDDING_DIM = 50      # k ∈ {20, 50}
LEARNING_RATE = 0.01    # λ ∈ {0.001, 0.01, 0.1}
MARGIN = 1.0            # γ ∈ {1, 2, 10}
NORM_ORDER = 1          # L1 o L2
```

---

## 🎯 Protocolo de Evaluación

### 1. Link Prediction (Sección 4.2)

**Procedimiento del Paper:**

Para cada tripleta de test `(h, r, t)`:

1. **Corromper Head:**
   - Reemplazar `h` con cada entidad del vocabulario
   - Calcular `d(h', r, t)` para todas las entidades `h'`
   - Rankear por distancia ascendente
   - Guardar el rank de la entidad correcta

2. **Corromper Tail:**
   - Reemplazar `t` con cada entidad del vocabulario
   - Calcular `d(h, r, t')` para todas las entidades `t'`
   - Rankear por distancia ascendente
   - Guardar el rank de la entidad correcta

3. **Métricas:**
   - **Mean Rank (MR):** Promedio de los ranks
   - **Mean Reciprocal Rank (MRR):** Promedio de 1/rank
   - **Hits@K:** % de ranks ≤ K

**Filtered vs Raw (del Paper):**
> "We propose to remove from the list of corrupted triplets all the triplets 
> that appear either in the training, validation or test set (except the test 
> triplet of interest). This ensures that all corrupted triplets do not belong 
> to the data set."

- **Raw:** Todos los corruptos se consideran (puede ser injusto)
- **Filtered:** Remover tripletas que aparecen en train/valid/test

### 2. Triple Classification

**Protocolo (usado en evaluación):**

1. **Generar Negativos:**
   - Para cada tripleta positiva en test, generar 1 negativo
   - Corromper aleatoriamente head o tail

2. **Encontrar Umbral Óptimo:**
   - Usar conjunto de validación
   - Probar diferentes umbrales
   - Seleccionar el que maximiza accuracy

3. **Evaluar en Test:**
   - Aplicar umbral óptimo
   - Calcular: Accuracy, F1, Precision, Recall, AUC-ROC

---

## 🚨 Manejo de Escenarios Desafiantes

### 1. Out-Of-Knowledge-Base (OOKB)

**Problema:**
En escenarios OOKB, el test contiene entidades que NUNCA aparecieron en train.

**Solución Implementada:**
```python
# Crear un embedding especial para entidades desconocidas
self.unknown_entity_embedding = nn.Parameter(
    torch.randn(embedding_dim) * init_bound
)

# Durante inferencia, detectar y manejar entidades OOKB
def get_embeddings(self, heads, relations, tails, handle_ookb=True):
    if handle_ookb:
        ookb_mask_h = heads >= self.num_entities
        ookb_mask_t = tails >= self.num_entities
        
        # Reemplazar con embedding especial
        h_emb[ookb_mask_h] = self.unknown_entity_embedding
        t_emb[ookb_mask_t] = self.unknown_entity_embedding
```

**Justificación:**
El paper original NO cubre OOKB. Esta es una extensión necesaria para:
- Evitar crashes por índices fuera de rango
- Proporcionar baseline medible (aunque subóptimo)
- Permitir comparación con métodos inductivos modernos

### 2. Inductive Learning (Nuevas Relaciones)

**Del Paper (Sección 4.4):**

Experimento: "Learning to predict new relationships with few examples"
- 40 relaciones desconocidas
- Evaluar con 0, 10, 100, 1000 ejemplos

Resultado:
> "TransE is the fastest method to learn: with only 10 examples of a new 
> relationship, the hits@10 is already 18%"

**Implementación:**
Soportado por el DataLoader en modo `'inductive'`.

---

## 📊 Resultados Esperados (del Paper)

### Tabla 3: Link Prediction Results

**FB15k (Filtered):**

| Modelo          | Mean Rank | Hits@10 (%) |
|-----------------|-----------|-------------|
| Unstructured    | 979       | 6.3         |
| SE              | 162       | 39.8        |
| SME(Linear)     | 154       | 40.8        |
| **TransE**      | **125**   | **47.1**    |

**WN (Filtered):**

| Modelo          | Mean Rank | Hits@10 (%) |
|-----------------|-----------|-------------|
| Unstructured    | 304       | 38.2        |
| LFM             | 456       | 81.6        |
| **TransE**      | **251**   | **89.2**    |

### Análisis por Categoría de Relación (Tabla 4)

TransE destaca en:
- ✅ **1-to-Many (tail):** 65.7% Hits@10
- ✅ **Many-to-1 (tail):** 66.7% Hits@10
- ✅ **1-to-1:** 43.7% Hits@10
- ⚠️ **Many-to-Many:** 47.2% / 50.0%

---

## 🎓 Diferencias Clave vs Otros Modelos

### vs Structured Embeddings (SE)

Del paper (Página 4):
> "SE is more expressive than our proposal. However, its complexity may make 
> it quite hard to learn, resulting in worse performance."

**SE:** Aprende 2 matrices por relación → Más parámetros → Más difícil optimizar
**TransE:** Aprende 1 vector por relación → Menos parámetros → Más fácil optimizar

### vs Neural Tensor Network

Del paper (Ecuación 3, Página 4):
> "TransE corresponds to the model where L is the identity matrix"

**TransE es un caso especial simplificado:**
- Menos parámetros (más eficiente)
- Entrenamiento más estable
- Rendimiento competitivo en KBs grandes

---

## 💾 Uso del Script

### Ejecución Básica

```bash
python transe_model.py
```

### Configuración de Parámetros

Editar en `main()`:

```python
# Dataset
DATASET_NAME = 'CoDEx-M'  # 'FB15k-237', 'WN18RR', etc.
MODE = 'ookb'             # 'standard', 'ookb', 'inductive'

# Hiperparámetros (según el paper)
EMBEDDING_DIM = 50
LEARNING_RATE = 0.01
MARGIN = 1.0
NORM_ORDER = 1  # 1=L1, 2=L2
```

### Salidas

1. **Durante entrenamiento:**
   - Loss por época
   - MRR en validación (para early stopping)

2. **Evaluación final:**
   - Ranking: MRR, MR, Hits@1/3/10
   - Clasificación: AUC, Accuracy, F1

3. **Reporte PDF:**
   - Gráficas ROC y Precision-Recall
   - Distribuciones de scores
   - Análisis de ranking
   - Tabla de métricas

---

## 🔬 Consideraciones Técnicas

### 1. Complejidad Computacional

**Parámetros Totales:**
```
O(n_e * k + n_r * k)
```
Donde:
- `n_e`: Número de entidades
- `n_r`: Número de relaciones
- `k`: Dimensión de embeddings

**Comparación (FB15k, Tabla 1):**
- RESCAL: 87.80M parámetros
- SE: 7.47M parámetros
- **TransE: 0.81M parámetros** ✅

### 2. Limitaciones del Modelo

Del paper (Sección 3):
> "The simple formulation of TransE... involves drawbacks. For modeling data 
> where 3-way dependencies between h, l and t are crucial, our model can fail."

**Ejemplo problemático:** Kinships dataset
- Requiere interacciones ternarias complejas
- TransE no alcanza state-of-the-art

**Fortalezas:**
- KBs grandes y heterogéneos (Freebase, WordNet)
- Relaciones jerárquicas (hypernym, part-of)
- Relaciones 1-to-1 (capital-of)

### 3. Optimización

**SGD con Learning Rate Constante:**
Del paper:
> "The parameters are then updated by taking a gradient step with constant 
> learning rate."

No usa:
- ❌ Learning rate decay
- ❌ Momentum
- ❌ Adam/AdaGrad

Solo usa:
- ✅ SGD vanilla
- ✅ Early stopping en validación

---

## 📖 Referencias

```bibtex
@inproceedings{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto 
          and Weston, Jason and Yakhnenko, Oksana},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2787--2795},
  year={2013}
}
```

---

## ✅ Checklist de Fidelidad al Paper

- [x] Inicialización Glorot uniforme (líneas 1-3, Algoritmo 1)
- [x] Normalización de relaciones solo en init (línea 2)
- [x] Normalización de entidades cada época (línea 5)
- [x] Negative sampling (Ecuación 2)
- [x] Margin ranking loss (Ecuación 1)
- [x] SGD con learning rate constante
- [x] Early stopping en validación
- [x] Evaluación filtered y raw
- [x] Link prediction protocol
- [x] Hiperparámetros del paper
- [x] Manejo de OOKB (extensión)

---

## 🚀 Mejoras Futuras (Más Allá del Paper)

1. **RotatE (Sun et al., 2019):**
   - Relaciones como rotaciones en espacio complejo
   - Mejor para relaciones simétricas

2. **ComplEx (Trouillon et al., 2016):**
   - Embeddings complejos
   - Maneja simetría/antisimetría

3. **ConvE (Dettmers et al., 2018):**
   - Convoluciones 2D
   - Más expresivo

4. **Encoder-based (Hwang et al.):**
   - GNN encoders para OOKB
   - Features de entidades

---

**Autor de la Implementación:** Claude (Anthropic)  
**Basado en:** Bordes et al., "Translating Embeddings for Modeling Multi-relational Data", NIPS 2013  
**Fecha:** Febrero 2026
