# INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs

**Implementación basada en:** Lee et al., 2023 (ICML)  
**Paper:** [INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs](https://arxiv.org/abs/2305.19987)

## 📋 Descripción

INGRAM es el primer modelo de Knowledge Graph Embedding que puede generar embeddings de **relaciones nuevas** en tiempo de inferencia, además de entidades nuevas. Esto lo hace ideal para escenarios de **Zero-Shot Relation Learning**.

### Problema que resuelve

Los modelos tradicionales de KG completion fallan cuando aparecen relaciones no vistas durante el entrenamiento:

- ❌ **GraIL, CoMPILE, SNRI**: Solo manejan entidades nuevas
- ❌ **TransE, RotatE, DistMult**: Requieren todas las relaciones en training
- ✅ **INGRAM**: Maneja relaciones Y entidades completamente nuevas

### Innovación clave: Grafo de Relaciones

INGRAM construye un grafo donde:
- **Nodos** = Relaciones del KG
- **Aristas** = Afinidad entre relaciones (basada en co-ocurrencia de entidades)

Esto permite que relaciones nuevas se representen como **combinación ponderada** de relaciones conocidas similares.

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    INGRAM Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. RELATION GRAPH BUILDER (Sección 4)                      │
│     ┌──────────────────────────────────────────┐            │
│     │ Input: Tripletas (h, r, t)              │            │
│     │ Output: Matriz A (afinidad relaciones)   │            │
│     │                                          │            │
│     │ Proceso:                                 │            │
│     │  • Eh[i,j] = freq(entidad_i como head de rel_j)      │
│     │  • Et[i,j] = freq(entidad_i como tail de rel_j)      │
│     │  • Ah = Eh^T @ Dh^(-2) @ Eh              │            │
│     │  • At = Et^T @ Dt^(-2) @ Et              │            │
│     │  • A = Ah + At (con self-loops)          │            │
│     └──────────────────────────────────────────┘            │
│                         ↓                                    │
│  2. RELATION-LEVEL AGGREGATION (Sección 5.1)                │
│     ┌──────────────────────────────────────────┐            │
│     │ L capas de atención multi-head           │            │
│     │                                          │            │
│     │ Para cada relación r_i:                  │            │
│     │   z_i^(l+1) = σ(Σ α_ij W^(l) z_j^(l))   │            │
│     │                                          │            │
│     │ donde α_ij incluye:                      │            │
│     │   • Atención local (GAT-style)           │            │
│     │   • Peso de afinidad global c_s(i,j)     │            │
│     │                                          │            │
│     │ Novedad: Binning de afinidad             │            │
│     │   s(i,j) = bin basado en rank(A[i,j])   │            │
│     └──────────────────────────────────────────┘            │
│                         ↓                                    │
│  3. ENTITY-LEVEL AGGREGATION (Sección 5.2)                  │
│     ┌──────────────────────────────────────────┐            │
│     │ L̂ capas de atención multi-head           │            │
│     │                                          │            │
│     │ Para cada entidad v_i:                   │            │
│     │   h_i^(l+1) = σ(β_ii Wc[h_i || z̄_i] +   │            │
│     │                  Σ β_ijk Wc[h_j || z_k]) │            │
│     │                                          │            │
│     │ Extensión de GATv2:                      │            │
│     │   • Incorpora vectores de relación       │            │
│     │   • z̄_i = promedio de relaciones adj.    │            │
│     └──────────────────────────────────────────┘            │
│                         ↓                                    │
│  4. SCORING FUNCTION (Sección 5.3)                          │
│     ┌──────────────────────────────────────────┐            │
│     │ Variante de DistMult:                    │            │
│     │                                          │            │
│     │   f(h, r, t) = h^T diag(W z_r) t        │            │
│     │                                          │            │
│     │ donde:                                   │            │
│     │   • h, t: entity embeddings finales      │            │
│     │   • z_r: relation embedding final        │            │
│     │   • W: matriz de transformación          │            │
│     └──────────────────────────────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Componentes Clave

### 1. Relation Graph Builder

**Paper Sección 4**

Construye la matriz de adyacencia A del grafo de relaciones:

```python
# Matrices de frecuencia
Eh[i, j] = frecuencia de entidad_i como head de relación_j
Et[i, j] = frecuencia de entidad_i como tail de relación_j

# Normalización por grado de entidad
Dh[i, i] = Σ_j Eh[i, j]  # Grado de entidad i como head
Dt[i, i] = Σ_j Et[i, j]  # Grado de entidad i como tail

# Afinidad entre relaciones
Ah = Eh^T @ Dh^(-2) @ Eh  # Afinidad vía heads
At = Et^T @ Dt^(-2) @ Et  # Afinidad vía tails

# Matriz final
A = Ah + At + I  # I = self-loops
```

**Intuición:** Dos relaciones tienen alta afinidad si comparten muchas entidades frecuentemente.

**Ejemplo:**
```
Relación "BornIn" y "LivesIn" → Alta afinidad (comparten personas/lugares)
Relación "ActedIn" y "TimeZone" → Baja afinidad (dominios diferentes)
```

### 2. Relation-Level Aggregation

**Paper Ecuaciones 1-3**

Actualiza representaciones de relaciones mediante atención:

```python
# Para cada relación r_i
z_i^(l+1) = σ(Σ_{r_j ∈ N_i} α_ij W^(l) z_j^(l))

# Coeficiente de atención
α_ij = softmax(y^(l) σ(P^(l) [z_i || z_j]) + c_s(i,j))
       \_________________________________________/   \______/
                 Atención local (GATv2)          Peso de afinidad global

# Binning de afinidad
s(i,j) = ⌊rank(A[i,j]) × B / nnz(A)⌋
```

**Diferencia clave vs GAT/GATv2:**
- GAT: Solo atención local
- INGRAM: Atención local + **pesos de afinidad global** (c_s(i,j))

El binning permite aprender B parámetros distintos para diferentes niveles de afinidad:
- c_1: Para relaciones muy afines (rank bajo)
- c_B: Para relaciones poco afines (rank alto)

### 3. Entity-Level Aggregation

**Paper Ecuación 4**

Extiende GATv2 incorporando vectores de relación:

```python
# Para cada entidad v_i
h_i^(l+1) = σ(β_ii Wc^(l)[h_i || z̄_i] + 
              Σ_{v_j ∈ N_i} Σ_{r_k ∈ R_ji} β_ijk Wc^(l)[h_j || z_k])

# z̄_i = promedio de relaciones adyacentes a v_i
z̄_i = (1/|N_i|) Σ_{v_j ∈ N_i} Σ_{r_k ∈ R_ji} z_k^(L)

# Atención
β_ijk = softmax(ŷ^(l) σ(P̂^(l) [h_i || h_j || z_k]))
```

**Extensión de GATv2:**
- GATv2: Agrega solo entidades vecinas
- INGRAM: Agrega entidades + **relaciones que las conectan**

Esto es crucial porque relaciones distintas tienen semánticas diferentes:
- `(Obama, BornIn, Hawaii)` vs `(Obama, PresidentOf, USA)`

### 4. Training Regime: División Dinámica

**Paper Sección 5.4**

Estrategia clave para generalización:

```python
Para cada época:
    1. Re-split Etr en Ftr y Ttr (ratio 3:1)
       Restricciones:
       • Ftr contiene árbol de expansión mínimo
       • Ftr cubre todas las relaciones
    
    2. Re-inicializar features (Glorot init)
       • entity_features ← Xavier_uniform(num_entities, d̂)
       • relation_features ← Xavier_uniform(num_relations, d)
    
    3. Entrenar en Ttr con loss:
       L = Σ max(0, γ - f(pos) + f(neg))
```

**¿Por qué funciona?**
- **División dinámica**: Evita memorizar configuraciones específicas
- **Re-inicialización**: Aprende a generalizar desde features aleatorios

→ En inferencia, puede manejar features de relaciones completamente nuevas

## 📊 Resultados del Paper

Comparación en datasets con **100% relaciones nuevas** (más desafiante):

| Método | NL-100 MRR | WK-100 MRR | FB-100 MRR |
|--------|------------|------------|------------|
| GraIL | 0.135 | - | - |
| RMPI | 0.220 | - | - |
| RED-GNN | 0.212 | 0.096 | 0.121 |
| NBFNet | 0.096 | 0.014 | 0.072 |
| **INGRAM** | **0.309** ↑ | **0.107** ↑ | **0.223** ↑ |

**Tiempo de entrenamiento** (NL-100):
- RMPI: 52 horas
- **INGRAM: 15 minutos** (200× más rápido)

## 🚀 Uso

### Instalación

```bash
pip install torch numpy pandas tqdm scikit-learn matplotlib seaborn
```

### Entrenamiento Básico

```python
from ingram_model import INGRAM, INGRAMTrainer

# Crear modelo
model = INGRAM(
    num_entities=1000,
    num_relations=50,
    entity_dim=32,
    relation_dim=32,
    entity_hidden_dim=128,
    relation_hidden_dim=64,
    num_relation_layers=2,
    num_entity_layers=3
)

# Entrenar
trainer = INGRAMTrainer(model, lr=0.001, margin=1.5)
loss = trainer.train_epoch(triplets, num_entities, num_relations)
```

### Inferencia con Relaciones Nuevas

```python
# Generar embeddings (relaciones pueden ser nuevas)
entity_emb, relation_emb = model(inference_triplets)

# Scoring
scores = model.score(heads, rels, tails, entity_emb, relation_emb)
```

### Script Completo

```bash
python train_ingram.py \
    --dataset CoDEx-M \
    --mode inductive \
    --split NL-25 \
    --epochs 10000 \
    --val_every 200 \
    --lr 0.001 \
    --margin 1.5
```

## 🔬 Integración con Scripts Provistos

### KGDataLoader

```python
from kg_dataloader import KGDataLoader

# Cargar datos
loader = KGDataLoader('CoDEx-M', mode='inductive', inductive_split='NL-25')
loader.load()

# Entrenar INGRAM
model = INGRAM(loader.num_entities, loader.num_relations, ...)
trainer = INGRAMTrainer(model)

for epoch in range(10000):
    loss = trainer.train_epoch(loader.train_data, ...)
```

### UnifiedKGScorer

```python
from unified_kg_scorer import UnifiedKGScorer

# Generar embeddings
entity_emb, relation_emb = model(triplets)
predict_fn = create_predict_fn(model, entity_emb, relation_emb)

# Evaluar
scorer = UnifiedKGScorer(device='cuda')

# Ranking metrics
ranking_metrics = scorer.evaluate_ranking(
    predict_fn, 
    test_triples=loader.test_data.numpy(),
    num_entities=loader.num_entities,
    k_values=[1, 3, 10]
)

# Classification metrics
class_metrics = scorer.evaluate_classification(
    predict_fn,
    valid_pos=loader.valid_data.numpy(),
    test_pos=loader.test_data.numpy(),
    num_entities=loader.num_entities
)

# Generar reporte PDF
scorer.export_report("INGRAM", "reporte_ingram.pdf")
```

## 📖 Detalles de Implementación

### Diferencias con el Paper

1. **Agregación de Relaciones (Sección 5.1)**
   - Paper: Implementación con sparse tensors
   - Implementación: Iteración explícita sobre relaciones (más clara para demo)
   - Optimización futura: Usar torch_sparse para escalabilidad

2. **División Dinámica (Sección 5.4)**
   - Paper: Minimum spanning tree exacto
   - Implementación: BFS para conectividad (más simple)
   - Ambos garantizan: Grafo conexo + todas relaciones cubiertas

3. **Multi-Head Attention**
   - Implementado según Brody et al., 2022 (GATv2)
   - Resuelve "static attention" mencionado en el paper

### Hiperparámetros Recomendados (del Paper)

**Dimensiones:**
- `entity_dim`, `relation_dim`: 32
- `entity_hidden_dim`: 128, 256
- `relation_hidden_dim`: 64, 128, 256

**Capas:**
- `num_relation_layers` (L): 1, 2, 3
- `num_entity_layers` (L̂): 2, 3, 4

**Atención:**
- `num_relation_heads` (K): 8, 16
- `num_entity_heads` (K̂): 8, 16
- `num_bins` (B): 1, 5, 10

**Entrenamiento:**
- `lr`: 0.0005, 0.001
- `margin` (γ): 1.0, 1.5, 2.0, 2.5
- `epochs`: 10,000
- `val_every`: 200

**Mejor configuración (NL datasets):**
```python
model = INGRAM(
    entity_dim=32,
    relation_dim=32,
    entity_hidden_dim=256,
    relation_hidden_dim=64,
    num_relation_layers=2,
    num_entity_layers=3,
    num_relation_heads=8,
    num_entity_heads=8,
    num_bins=10
)
```

## 🧪 Testing

Ejecutar test básico:

```bash
python ingram_model.py
```

Salida esperada:
```
================================================================================
INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs
Implementación basada en Lee et al., 2023 (ICML)
================================================================================

Dispositivo: cuda

Modelo creado con:
  - 100 entidades
  - 20 relaciones
  - XXX,XXX parámetros totales

  - 500 tripletas sintéticas generadas

Ejecutando forward pass...
  ✓ Entity embeddings: torch.Size([100, 32])
  ✓ Relation embeddings: torch.Size([20, 32])
  ✓ Scores de prueba: tensor([...])

✓ Test básico completado exitosamente!
================================================================================
```

## 📚 Referencias

**Paper principal:**
```bibtex
@inproceedings{lee2023ingram,
  title={INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs},
  author={Lee, Jaejun and Chung, Chanyoung and Whang, Joyce Jiyoung},
  booktitle={Proceedings of the 40th International Conference on Machine Learning},
  year={2023}
}
```

**Métodos relacionados:**
- GATv2: Brody et al., 2022 (resuelve static attention)
- GraIL: Teru et al., 2020 (subgraph reasoning, solo entidades nuevas)
- DistMult: Yang et al., 2015 (scoring function base)

## 🤝 Comparación con Baselines

| Característica | GraIL | RMPI | RED-GNN | INGRAM |
|---------------|-------|------|---------|--------|
| Entidades nuevas | ✅ | ✅ | ✅ | ✅ |
| Relaciones nuevas | ❌ | ❌ | ❌ | ✅ |
| Usa LLMs | ❌ | ❌ | ❌ | ❌ |
| Escalabilidad | Baja | Muy Baja | Media | Alta |
| Grafo de relaciones | ❌ | ❌ | ❌ | ✅ |
| División dinámica | ❌ | ❌ | ❌ | ✅ |

## 💡 Casos de Uso

1. **Knowledge Graphs Evolutivos**
   - Añadir nuevas relaciones sin re-entrenar
   - Ejemplo: Añadir "VacunadoCon" en un KG médico

2. **Transfer Learning entre Dominios**
   - Entrenar en un dominio, inferir en otro
   - Ejemplo: Entrenar en Freebase, aplicar a Wikidata

3. **Few-Shot Learning**
   - Pocas muestras de relaciones nuevas
   - INGRAM puede interpolar desde relaciones similares

## 🐛 Troubleshooting

**Out of Memory:**
```python
# Reducir dimensiones ocultas
entity_hidden_dim=128  # en lugar de 256
relation_hidden_dim=32  # en lugar de 64

# Reducir batch size
batch_size=64  # en lugar de 128
```

**Convergencia lenta:**
```python
# Aumentar learning rate
lr=0.002  # en lugar de 0.001

# Reducir margin
margin=1.0  # en lugar de 1.5
```

**Overfitting:**
```python
# Aumentar dropout
dropout=0.2  # en lugar de 0.1

# Asegurar división dinámica está activa
# (debería estar por defecto)
```

## 📝 TODO / Mejoras Futuras

- [ ] Soporte para grafos temporales (MTKGE extension)
- [ ] Implementación con torch_sparse para grafos grandes
- [ ] Pre-entrenamiento con contrastive learning
- [ ] Integración con HuggingFace Transformers
- [ ] Benchmarks en datasets oficiales (FB15k-237, WN18RR)
- [ ] Visualización de embeddings de relaciones
- [ ] Análisis de interpretabilidad de pesos c_s(i,j)

## 📧 Contacto

Para preguntas sobre la implementación o el paper, consultar:
- Paper: https://arxiv.org/abs/2305.19987
- Repo oficial: https://github.com/bdi-lab/InGram

---

**Implementación realizada para fines de investigación basada en:**  
Lee, J., Chung, C., & Whang, J. J. (2023). INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs. *ICML 2023*.
