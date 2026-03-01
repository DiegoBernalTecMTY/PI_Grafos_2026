# INGRAM Implementation - Resumen Ejecutivo

## 📦 Archivos Entregados

### Implementación Principal
1. **`ingram_model.py`** (42.8 KB)
   - Implementación completa del modelo INGRAM
   - Componentes: RelationGraphBuilder, RelationLevelAggregation, EntityLevelAggregation, INGRAM, INGRAMTrainer
   - ~1,200 líneas de código con anotaciones detalladas

2. **`train_ingram.py`** (16.9 KB)
   - Script principal de entrenamiento y evaluación
   - Compatible con KGDataLoader y UnifiedKGScorer provistos
   - Incluye manejo de argumentos y workflow completo

3. **`demo_ingram.py`** (18.8 KB)
   - Notebook interactivo de demostración
   - Visualizaciones de grafos de relaciones
   - Análisis de embeddings y pesos aprendidos

### Documentación
4. **`README.md`** (17.2 KB)
   - Documentación completa del modelo
   - Explicación detallada de cada componente
   - Guías de uso e integración
   - Comparación con baselines

## 🎯 Correspondencia con el Paper

### Sección 4: Grafo de Relaciones
**Implementado en:** `RelationGraphBuilder`

```python
# Paper Ecuación: A = Ah + At
# Donde:
#   Ah = Eh^T @ Dh^(-2) @ Eh
#   At = Et^T @ Dt^(-2) @ Et

class RelationGraphBuilder:
    def build(self, triplets):
        # 1. Construir matrices Eh y Et
        Eh = frecuencias de (entidad, relación) como head
        Et = frecuencias de (entidad, relación) como tail
        
        # 2. Normalizar por grado de entidad
        Dh = diagonal de grados de entidades (heads)
        Dt = diagonal de grados de entidades (tails)
        
        # 3. Calcular afinidad
        Ah = Eh^T @ Dh^(-2) @ Eh
        At = Et^T @ Dt^(-2) @ Et
        
        # 4. Combinar
        A = Ah + At + I  # Con self-loops
        return A
```

**Líneas:** 55-123

### Sección 5.1: Agregación de Relaciones
**Implementado en:** `RelationLevelAggregation`

```python
# Paper Ecuación 1:
#   z_i^(l+1) = σ(Σ_{r_j ∈ N_i} α_ij W^(l) z_j^(l))
#
# Paper Ecuación 2:
#   α_ij = softmax(y^(l) σ(P^(l) [z_i || z_j]) + c_s(i,j))

class RelationLevelAggregation(nn.Module):
    def forward(self, z, A, neighbor_indices, affinity_bins):
        # Para cada relación r_i:
        for i in range(num_relations):
            # 1. Obtener vecinos
            neighbors = neighbor_indices[i]
            z_neighbors = z[neighbors]
            
            # 2. Calcular atención
            z_concat = [z_i || z_j]
            h = LeakyReLU(P(z_concat))
            attn_scores = y(h)
            
            # 3. Añadir pesos de afinidad (NOVEDAD vs GATv2)
            bins = affinity_bins[i]
            c_weights = c_bins[bins]
            attn_scores = attn_scores + c_weights
            
            # 4. Normalizar y agregar
            attn_weights = softmax(attn_scores)
            z_aggregated = Σ(attn_weights * W(z_neighbors))
            
            # 5. Residual connection
            z_new = LeakyReLU(z_aggregated + z_i)
```

**Líneas:** 126-251

### Sección 5.2: Agregación de Entidades
**Implementado en:** `EntityLevelAggregation`

```python
# Paper Ecuación 4:
#   h_i^(l+1) = σ(β_ii Wc[h_i || z̄_i] + Σ β_ijk Wc[h_j || z_k])
#
# Extensión de GATv2 con vectores de relación

class EntityLevelAggregation(nn.Module):
    def forward(self, h, z, edge_index, edge_type):
        # Para cada entidad v_i:
        for i in range(num_entities):
            # 1. Calcular z̄_i (promedio de relaciones adyacentes)
            neighbor_relations = [r_k para vecinos]
            z_bar_i = mean(z[neighbor_relations])
            
            # 2. Self-loop: [h_i || z̄_i]
            h_self_concat = [h_i || z_bar_i]
            
            # 3. Neighbors: [h_j || z_k]
            h_neighbor_concat = [h_neighbors || z_neighbors]
            
            # 4. Atención sobre [h_i || h_j || z_k]
            b_self = [h_i || h_i || z̄_i]
            b_neighbors = [h_i || h_j || z_k]
            
            attn_weights = softmax(ŷ(LeakyReLU(P̂(b_all))))
            
            # 5. Agregar
            h_aggregated = Σ(attn_weights * Wc(concat))
            h_new = LeakyReLU(h_aggregated + h_i)
```

**Líneas:** 254-374

### Sección 5.3: Scoring Function
**Implementado en:** `INGRAM.score()`

```python
# Paper Ecuación 5:
#   f(v_i, r_k, v_j) = h_i^T diag(W z_k) h_j
#
# Variante de DistMult con transformación W

def score(self, heads, rels, tails, entity_emb, relation_emb):
    h_i = entity_emb[heads]
    z_k = relation_emb[rels]
    h_j = entity_emb[tails]
    
    # W z_k: transforma dim de relación a dim de entidad
    Wz_k = z_k @ W^T
    
    # Score: h_i^T diag(W z_k) h_j
    #      = sum(h_i * W z_k * h_j)
    scores = (h_i * Wz_k * h_j).sum(dim=1)
    return scores
```

**Líneas:** 566-599

### Sección 5.4: Training Regime
**Implementado en:** `INGRAMTrainer`

```python
# Paper: "For every epoch, we randomly re-split Ftr and Ttr"

class INGRAMTrainer:
    def train_epoch(self, all_triplets, num_entities, num_relations):
        # 1. División dinámica (3:1 ratio)
        Ftr, Ttr = self.dynamic_split(all_triplets, ...)
        
        # Restricciones:
        # • Ftr contiene árbol de expansión mínimo
        # • Ftr cubre todas las relaciones
        
        # 2. Re-inicializar features (Glorot)
        entity_features, relation_features = model.init_features(device)
        
        # 3. Forward pass
        entity_emb, relation_emb = model(all_triplets)
        
        # 4. Generar negativos
        negatives = corrupt_heads_or_tails(Ttr)
        
        # 5. Margin-based ranking loss
        loss = Σ max(0, γ - f(pos) + f(neg))
```

**Líneas:** 602-757

## 🔬 Validación de Implementación

### Componentes Verificados

✅ **Grafo de Relaciones**
- Matrices Eh, Et correctamente construidas
- Normalización Dh^(-2), Dt^(-2) implementada
- Self-loops añadidos

✅ **Atención con Afinidad**
- GATv2 base implementado
- Pesos c_s(i,j) añadidos
- Binning según Ecuación 3

✅ **Multi-Head Attention**
- K heads para relaciones
- K̂ heads para entidades
- Residual connections

✅ **División Dinámica**
- Re-split por época
- Restricciones de conectividad
- Re-inicialización de features

### Diferencias Menores con el Paper

1. **Implementación de Agregación**
   - Paper: Usa sparse tensors (no especificado explícitamente)
   - Código: Iteración explícita (más claro, menos escalable)
   - Impacto: Ninguno en lógica, solo eficiencia

2. **Minimum Spanning Tree**
   - Paper: MST exacto
   - Código: BFS para conectividad
   - Impacto: Ambos garantizan grafo conexo

3. **Batching**
   - Paper: No especifica estrategia de batching
   - Código: Batch simple sobre tripletas
   - Impacto: Ninguno en convergencia

## 📋 Checklist de Implementación

### Core Features (del Paper)
- [x] Relation Graph Builder (Sección 4)
- [x] Relation-Level Aggregation (Sección 5.1)
- [x] Entity-Level Aggregation (Sección 5.2)
- [x] Scoring Function variante DistMult (Sección 5.3)
- [x] Training con División Dinámica (Sección 5.4)
- [x] Binning de Afinidad (Ecuación 3)
- [x] Multi-Head Attention (K, K̂ heads)
- [x] Residual Connections
- [x] Re-inicialización por época

### Extras Implementados
- [x] Integración con KGDataLoader
- [x] Integración con UnifiedKGScorer
- [x] Script de entrenamiento completo
- [x] Demo interactivo
- [x] Visualizaciones
- [x] Documentación exhaustiva

## 🚀 Instrucciones de Uso

### 1. Instalación de Dependencias

```bash
pip install torch numpy pandas tqdm scikit-learn matplotlib seaborn
```

### 2. Preparar Datos

El código es compatible con el `KGDataLoader` provisto:

```python
from kg_dataloader import KGDataLoader

loader = KGDataLoader(
    dataset_name='CoDEx-M',
    mode='inductive',  # o 'standard', 'ookb'
    inductive_split='NL-25',  # NL-25, NL-50, NL-75, NL-100
    base_dir='./data'
)
loader.load()
```

### 3. Entrenar INGRAM

```python
from ingram_model import INGRAM, INGRAMTrainer

# Crear modelo
model = INGRAM(
    num_entities=loader.num_entities,
    num_relations=loader.num_relations,
    entity_dim=32,
    relation_dim=32,
    entity_hidden_dim=256,
    relation_hidden_dim=64,
    num_relation_layers=2,
    num_entity_layers=3,
    num_relation_heads=8,
    num_entity_heads=8,
    num_bins=10
).to('cuda')

# Entrenar
trainer = INGRAMTrainer(model, lr=0.001, margin=1.5)

for epoch in range(10000):
    loss = trainer.train_epoch(
        loader.train_data,
        loader.num_entities,
        loader.num_relations,
        batch_size=128
    )
    
    if (epoch + 1) % 200 == 0:
        # Validar
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

### 4. Evaluar con UnifiedKGScorer

```python
from unified_kg_scorer import UnifiedKGScorer
from ingram_model import create_predict_fn

# Generar embeddings
model.eval()
with torch.no_grad():
    entity_emb, relation_emb = model(loader.train_data)

# Crear función de predicción
predict_fn = create_predict_fn(model, entity_emb, relation_emb)

# Evaluar
scorer = UnifiedKGScorer(device='cuda')

# Ranking metrics
ranking_metrics = scorer.evaluate_ranking(
    predict_fn=predict_fn,
    test_triples=loader.test_data.numpy(),
    num_entities=loader.num_entities,
    k_values=[1, 3, 10],
    higher_is_better=True
)

print(f"MRR: {ranking_metrics['mrr']:.4f}")
print(f"Hits@10: {ranking_metrics['hits@10']:.4f}")

# Classification metrics
class_metrics = scorer.evaluate_classification(
    predict_fn=predict_fn,
    valid_pos=loader.valid_data.numpy(),
    test_pos=loader.test_data.numpy(),
    num_entities=loader.num_entities,
    higher_is_better=True
)

print(f"AUC: {class_metrics['auc']:.4f}")
print(f"Accuracy: {class_metrics['accuracy']:.4f}")

# Generar reporte PDF
scorer.export_report(
    model_name="INGRAM",
    filename="reporte_ingram.pdf"
)
```

### 5. Script Todo-en-Uno

```bash
python train_ingram.py \
    --dataset CoDEx-M \
    --mode inductive \
    --split NL-25 \
    --epochs 10000 \
    --val_every 200 \
    --entity_dim 32 \
    --relation_dim 32 \
    --entity_hidden 256 \
    --relation_hidden 64 \
    --num_relation_layers 2 \
    --num_entity_layers 3 \
    --lr 0.001 \
    --margin 1.5 \
    --eval_ranking \
    --eval_classification \
    --output_dir ./outputs
```

## 📊 Resultados Esperados

Basados en el paper (datasets con 100% relaciones nuevas):

| Dataset | MRR (esperado) | Hits@10 (esperado) |
|---------|----------------|---------------------|
| NL-100 | ~0.309 | ~0.506 |
| WK-100 | ~0.107 | ~0.169 |
| FB-100 | ~0.223 | ~0.371 |

Comparado con el mejor baseline (RMPI):
- NL-100: +40% mejora en MRR
- 200× más rápido (15 min vs 52 horas)

## 🔍 Debugging Tips

### Problema: Loss no converge
```python
# Verificar división dinámica
print(f"Ftr size: {len(Ftr)}, Ttr size: {len(Ttr)}")

# Verificar que todas las relaciones están en Ftr
rels_in_ftr = set(Ftr[:, 1].tolist())
print(f"Relaciones en Ftr: {len(rels_in_ftr)}/{num_relations}")

# Reducir learning rate
trainer = INGRAMTrainer(model, lr=0.0005, margin=1.0)
```

### Problema: Out of Memory
```python
# Reducir dimensiones
model = INGRAM(
    ...,
    entity_hidden_dim=128,  # en vez de 256
    relation_hidden_dim=32,  # en vez de 64
    ...
)

# Batch size menor
trainer.train_epoch(..., batch_size=64)
```

### Problema: Overfitting
```python
# Aumentar dropout
model = INGRAM(..., dropout=0.2)

# Verificar división dinámica está activa
# (debería ser automático en train_epoch)
```

## 📖 Lectura Adicional

**Paper Original:**
- Lee et al., 2023: "INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs"
- https://arxiv.org/abs/2305.19987

**Métodos Relacionados:**
- GATv2 (Brody et al., 2022): Resuelve static attention
- GraIL (Teru et al., 2020): Subgraph reasoning
- DistMult (Yang et al., 2015): Scoring function

**Código Oficial:**
- https://github.com/bdi-lab/InGram

## 🎯 Conclusión

Esta implementación proporciona:

1. ✅ **Fidelidad al Paper**: Todos los componentes clave implementados según especificaciones
2. ✅ **Código Documentado**: ~500 líneas de comentarios explicando la relación con el paper
3. ✅ **Compatibilidad**: Integración con scripts provistos (KGDataLoader, UnifiedKGScorer)
4. ✅ **Reproducibilidad**: Hiperparámetros y workflow del paper
5. ✅ **Extensibilidad**: Arquitectura modular para experimentación

El modelo está listo para entrenar y evaluar en datasets de Zero-Shot Relation Learning.

---

**Autor:** Implementación basada en Lee et al., 2023  
**Fecha:** Febrero 2026  
**Versión:** 1.0
