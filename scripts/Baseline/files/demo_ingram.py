"""
INGRAM Demo: Zero-Shot Relation Learning
========================================

Este notebook demuestra las capacidades de INGRAM para aprender relaciones nuevas.

Contenido:
1. Setup y carga del modelo
2. Construcción del Grafo de Relaciones
3. Visualización de afinidad entre relaciones
4. Entrenamiento con división dinámica
5. Inferencia con relaciones completamente nuevas
6. Comparación con baselines
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importar INGRAM
from ingram_model import INGRAM, INGRAMTrainer, RelationGraphBuilder

# Configuración
sns.set_style("whitegrid")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# ============================================================================
# PARTE 1: Crear un Knowledge Graph Sintético
# ============================================================================

print("\n" + "="*80)
print("PARTE 1: Creación de Knowledge Graph Sintético")
print("="*80)

# Definir relaciones con semántica clara
relation_names = [
    # Relaciones geográficas (similares entre sí)
    "BornIn", "LivesIn", "CapitalOf", "LocatedIn",
    
    # Relaciones profesionales (similares entre sí)
    "WorksFor", "CEO_Of", "FoundedBy", "EmployedBy",
    
    # Relaciones familiares (similares entre sí)
    "ParentOf", "SiblingOf", "MarriedTo", "ChildOf",
    
    # Relaciones académicas
    "GraduatedFrom", "ProfessorAt", "StudiedAt",
    
    # Relaciones culturales
    "ActedIn", "DirectedBy", "WrittenBy"
]

entity_names = [
    # Personas
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack",
    
    # Organizaciones
    "TechCorp", "University_A", "Hospital_B", "Studio_C",
    
    # Lugares
    "CityX", "CityY", "CountryZ",
    
    # Obras
    "Movie1", "Movie2", "Book1"
]

num_entities = len(entity_names)
num_relations = len(relation_names)

print(f"\nGrafo sintético creado:")
print(f"  - {num_entities} entidades")
print(f"  - {num_relations} relaciones")

# Crear tripletas con semántica coherente
def create_semantic_triplets():
    """Crear tripletas que reflejen relaciones semánticamente coherentes."""
    triplets = []
    
    # Relaciones geográficas
    triplets.extend([
        (entity_names.index("Alice"), relation_names.index("BornIn"), entity_names.index("CityX")),
        (entity_names.index("Bob"), relation_names.index("BornIn"), entity_names.index("CityY")),
        (entity_names.index("Alice"), relation_names.index("LivesIn"), entity_names.index("CityX")),
        (entity_names.index("Charlie"), relation_names.index("LivesIn"), entity_names.index("CityY")),
        (entity_names.index("CityX"), relation_names.index("LocatedIn"), entity_names.index("CountryZ")),
        (entity_names.index("CityY"), relation_names.index("LocatedIn"), entity_names.index("CountryZ")),
    ])
    
    # Relaciones profesionales
    triplets.extend([
        (entity_names.index("Alice"), relation_names.index("WorksFor"), entity_names.index("TechCorp")),
        (entity_names.index("Bob"), relation_names.index("WorksFor"), entity_names.index("TechCorp")),
        (entity_names.index("Eve"), relation_names.index("CEO_Of"), entity_names.index("TechCorp")),
        (entity_names.index("TechCorp"), relation_names.index("FoundedBy"), entity_names.index("Frank")),
    ])
    
    # Relaciones familiares
    triplets.extend([
        (entity_names.index("Alice"), relation_names.index("ParentOf"), entity_names.index("Grace")),
        (entity_names.index("Bob"), relation_names.index("ParentOf"), entity_names.index("Grace")),
        (entity_names.index("Alice"), relation_names.index("MarriedTo"), entity_names.index("Bob")),
        (entity_names.index("Charlie"), relation_names.index("SiblingOf"), entity_names.index("Diana")),
    ])
    
    # Relaciones académicas
    triplets.extend([
        (entity_names.index("Grace"), relation_names.index("GraduatedFrom"), entity_names.index("University_A")),
        (entity_names.index("Henry"), relation_names.index("ProfessorAt"), entity_names.index("University_A")),
        (entity_names.index("Iris"), relation_names.index("StudiedAt"), entity_names.index("University_A")),
    ])
    
    # Relaciones culturales
    triplets.extend([
        (entity_names.index("Jack"), relation_names.index("ActedIn"), entity_names.index("Movie1")),
        (entity_names.index("Diana"), relation_names.index("DirectedBy"), entity_names.index("Movie1")),
        (entity_names.index("Frank"), relation_names.index("WrittenBy"), entity_names.index("Book1")),
    ])
    
    return torch.tensor(triplets, dtype=torch.long)

train_triplets = create_semantic_triplets()
print(f"  - {len(train_triplets)} tripletas de entrenamiento")

# Mostrar algunas tripletas
print("\nEjemplos de tripletas:")
for i in range(min(5, len(train_triplets))):
    h, r, t = train_triplets[i]
    print(f"  ({entity_names[h]}, {relation_names[r]}, {entity_names[t]})")

# ============================================================================
# PARTE 2: Construcción y Visualización del Grafo de Relaciones
# ============================================================================

print("\n" + "="*80)
print("PARTE 2: Grafo de Relaciones")
print("="*80)

# Construir grafo de relaciones
builder = RelationGraphBuilder(num_entities, num_relations)
A = builder.build(train_triplets.to(device))

print(f"\nMatriz de adyacencia construida: {A.shape}")
print(f"Valores de afinidad únicos: {torch.unique(A).numel()}")
print(f"Afinidad promedio: {A.mean().item():.4f}")

# Visualizar matriz de afinidad
plt.figure(figsize=(12, 10))
affinity_matrix = A.cpu().numpy()

sns.heatmap(
    affinity_matrix,
    xticklabels=relation_names,
    yticklabels=relation_names,
    cmap="YlOrRd",
    annot=False,
    fmt=".2f",
    cbar_kws={'label': 'Afinidad'}
)
plt.title("Grafo de Relaciones: Matriz de Afinidad\n(Colores cálidos = mayor afinidad)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Relación", fontsize=12)
plt.ylabel("Relación", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Guardar figura
output_dir = Path('/mnt/user-data/outputs')
output_dir.mkdir(exist_ok=True, parents=True)
plt.savefig(output_dir / 'relation_graph_affinity.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualización guardada en: {output_dir / 'relation_graph_affinity.png'}")
plt.close()

# Análisis de vecindarios
print("\n📊 Análisis de Vecindarios de Relaciones:")
print("-" * 80)

for i in range(min(5, num_relations)):
    neighbors = torch.where(A[i] > 0)[0]
    if len(neighbors) > 1:  # Excluir solo self-loop
        affinities = A[i][neighbors]
        sorted_indices = torch.argsort(affinities, descending=True)[:5]  # Top-5
        
        print(f"\n{relation_names[i]}:")
        for idx in sorted_indices:
            neighbor_idx = neighbors[idx].item()
            if neighbor_idx != i:  # No mostrar self-loop
                affinity = affinities[idx].item()
                print(f"  → {relation_names[neighbor_idx]:20s} (afinidad: {affinity:.4f})")

# ============================================================================
# PARTE 3: Inicialización y Arquitectura del Modelo
# ============================================================================

print("\n" + "="*80)
print("PARTE 3: Modelo INGRAM")
print("="*80)

# Crear modelo
model = INGRAM(
    num_entities=num_entities,
    num_relations=num_relations,
    entity_dim=32,
    relation_dim=32,
    entity_hidden_dim=128,
    relation_hidden_dim=64,
    num_relation_layers=2,
    num_entity_layers=3,
    num_relation_heads=8,
    num_entity_heads=8,
    num_bins=10,
    dropout=0.1
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"\nArquitectura del modelo:")
print(f"  - Parámetros totales: {num_params:,}")
print(f"  - Capas de agregación de relaciones: {model.num_relation_layers}")
print(f"  - Capas de agregación de entidades: {model.num_entity_layers}")
print(f"  - Heads de atención (relaciones): {8}")
print(f"  - Heads de atención (entidades): {8}")

# Visualizar arquitectura por componente
print("\n📐 Distribución de parámetros:")
total_params = 0
for name, module in model.named_children():
    module_params = sum(p.numel() for p in module.parameters())
    total_params += module_params
    print(f"  {name:30s}: {module_params:8,} parámetros ({100*module_params/num_params:.1f}%)")

# ============================================================================
# PARTE 4: Entrenamiento con División Dinámica
# ============================================================================

print("\n" + "="*80)
print("PARTE 4: Entrenamiento")
print("="*80)

trainer = INGRAMTrainer(model, lr=0.001, margin=1.5)

print("\nConfiguración de entrenamiento:")
print(f"  - Learning rate: 0.001")
print(f"  - Margin (γ): 1.5")
print(f"  - División dinámica: ✅")
print(f"  - Re-inicialización por época: ✅")

# Entrenar por algunas épocas
num_epochs = 100
losses = []

print(f"\nEntrenando {num_epochs} épocas...")
for epoch in range(num_epochs):
    loss = trainer.train_epoch(
        train_triplets.to(device),
        num_entities,
        num_relations,
        batch_size=32
    )
    losses.append(loss)
    
    if (epoch + 1) % 20 == 0:
        print(f"  Época {epoch+1:3d}/{num_epochs} - Loss: {loss:.4f}")

print(f"✓ Entrenamiento completado")

# Visualizar curva de loss
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2, color='#2E86AB')
plt.xlabel('Época', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Curva de Entrenamiento de INGRAM', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'training_curve.png', dpi=150, bbox_inches='tight')
print(f"✓ Curva de entrenamiento guardada en: {output_dir / 'training_curve.png'}")
plt.close()

# ============================================================================
# PARTE 5: Generación de Embeddings e Inferencia
# ============================================================================

print("\n" + "="*80)
print("PARTE 5: Generación de Embeddings")
print("="*80)

model.eval()
with torch.no_grad():
    entity_embeddings, relation_embeddings = model(train_triplets.to(device))

print(f"\nEmbeddings generados:")
print(f"  - Entidades: {entity_embeddings.shape}")
print(f"  - Relaciones: {relation_embeddings.shape}")

# Visualizar embeddings de relaciones (PCA a 2D)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
relation_emb_2d = pca.fit_transform(relation_embeddings.cpu().numpy())

plt.figure(figsize=(14, 10))

# Definir colores por categoría semántica
relation_categories = {
    'Geográficas': ['BornIn', 'LivesIn', 'CapitalOf', 'LocatedIn'],
    'Profesionales': ['WorksFor', 'CEO_Of', 'FoundedBy', 'EmployedBy'],
    'Familiares': ['ParentOf', 'SiblingOf', 'MarriedTo', 'ChildOf'],
    'Académicas': ['GraduatedFrom', 'ProfessorAt', 'StudiedAt'],
    'Culturales': ['ActedIn', 'DirectedBy', 'WrittenBy']
}

colors = ['#E63946', '#F1A208', '#2A9D8F', '#264653', '#8338EC']
category_to_color = {}
for i, (cat, rels) in enumerate(relation_categories.items()):
    for rel in rels:
        category_to_color[rel] = colors[i]

# Plotear cada relación
for i, rel_name in enumerate(relation_names):
    color = category_to_color.get(rel_name, '#95999C')
    plt.scatter(relation_emb_2d[i, 0], relation_emb_2d[i, 1], 
               c=color, s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
    plt.annotate(rel_name, (relation_emb_2d[i, 0], relation_emb_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))

# Leyenda
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[i], label=cat) 
                   for i, cat in enumerate(relation_categories.keys())]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=12)
plt.title('Embeddings de Relaciones (Proyección PCA)\n' + 
          'Relaciones semánticamente similares deberían agruparse',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'relation_embeddings_pca.png', dpi=150, bbox_inches='tight')
print(f"✓ Visualización de embeddings guardada en: {output_dir / 'relation_embeddings_pca.png'}")
plt.close()

# ============================================================================
# PARTE 6: Demostración de Zero-Shot Relation Learning
# ============================================================================

print("\n" + "="*80)
print("PARTE 6: Zero-Shot Relation Learning")
print("="*80)

print("\n🎯 Escenario: Aparece una relación NUEVA en tiempo de inferencia")
print("   Relación nueva: 'ColleagueOf' (similar a 'WorksFor', 'EmployedBy')")
print("\n   INGRAM puede generar un embedding para esta relación nueva")
print("   basándose en relaciones conocidas similares del grafo de relaciones.")

# Simular nueva relación añadiendo tripletas con un ID temporal
new_relation_id = num_relations  # ID fuera del rango de entrenamiento
new_triplets = torch.tensor([
    [entity_names.index("Alice"), new_relation_id, entity_names.index("Charlie")],
    [entity_names.index("Bob"), new_relation_id, entity_names.index("Diana")],
], dtype=torch.long, device=device)

# Combinar con tripletas de entrenamiento
inference_triplets = torch.cat([train_triplets.to(device), new_triplets], dim=0)

print(f"\nGrafo de inferencia:")
print(f"  - Tripletas originales: {len(train_triplets)}")
print(f"  - Tripletas con relación nueva: {len(new_triplets)}")
print(f"  - Total: {len(inference_triplets)}")

# NOTA: En la práctica, necesitaríamos modificar el modelo para manejar
# relaciones con IDs fuera del rango de entrenamiento. Esto requeriría:
# 1. Expandir dinámicamente las dimensiones del modelo
# 2. O mapear la nueva relación a un espacio interpolado

print("\n⚠️  LIMITACIÓN DE ESTA DEMO:")
print("   La demostración completa de zero-shot requiere modificaciones")
print("   al modelo para manejar IDs dinámicos de relaciones.")
print("\n   Sin embargo, el MECANISMO CLAVE ya está implementado:")
print("   ✓ Grafo de relaciones captura afinidad")
print("   ✓ Agregación a nivel de relación permite interpolación")
print("   ✓ División dinámica entrena el modelo para generalizar")

# ============================================================================
# PARTE 7: Análisis de Pesos de Afinidad Aprendidos
# ============================================================================

print("\n" + "="*80)
print("PARTE 7: Análisis de Pesos de Afinidad Aprendidos")
print("="*80)

# Extraer pesos c_s(i,j) de la primera capa de agregación de relaciones
relation_layer_0 = model.relation_layers[0]
c_bins = relation_layer_0.c_bins.detach().cpu().numpy()  # (num_bins, num_heads)

print(f"\nPesos de afinidad c_s(i,j):")
print(f"  Shape: {c_bins.shape}")
print(f"  (num_bins={relation_layer_0.num_bins}, num_heads={relation_layer_0.num_heads})")

# Promediar sobre heads
c_bins_mean = c_bins.mean(axis=1)

plt.figure(figsize=(10, 6))
bins = np.arange(1, len(c_bins_mean) + 1)
plt.bar(bins, c_bins_mean, color='#2A9D8F', alpha=0.7, edgecolor='black')
plt.xlabel('Bin de Afinidad s(i,j)', fontsize=12)
plt.ylabel('Peso Aprendido c_s(i,j)', fontsize=12)
plt.title('Pesos de Afinidad Aprendidos por INGRAM\n' + 
          '(Esperado: bins pequeños → pesos altos)',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Añadir línea de tendencia
z = np.polyfit(bins, c_bins_mean, 1)
p = np.poly1d(z)
plt.plot(bins, p(bins), "r--", alpha=0.8, linewidth=2, label=f'Tendencia: {z[0]:.3f}x + {z[1]:.3f}')
plt.legend()

plt.tight_layout()
plt.savefig(output_dir / 'affinity_weights.png', dpi=150, bbox_inches='tight')
print(f"✓ Análisis de pesos guardado en: {output_dir / 'affinity_weights.png'}")
plt.close()

# Interpretación
if c_bins_mean[0] > c_bins_mean[-1]:
    print("\n✅ CORRECTO: Pesos decrecientes con bins")
    print("   → Relaciones con alta afinidad (bin 1) reciben mayor peso")
    print("   → Modelo aprendió a usar la estructura del grafo de relaciones")
else:
    print("\n⚠️  Pesos no siguen patrón esperado")
    print("   → Puede requerir más épocas de entrenamiento")
    print("   → O ajuste de hiperparámetros")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUMEN DE LA DEMOSTRACIÓN")
print("="*80)

print("""
✅ COMPONENTES IMPLEMENTADOS:

1. RELATION GRAPH BUILDER (Sección 4)
   • Matriz de adyacencia basada en co-ocurrencia de entidades
   • Normalización por grado para equidad
   • Self-loops para auto-representación

2. RELATION-LEVEL AGGREGATION (Sección 5.1)
   • Atención multi-head (GATv2-style)
   • Pesos de afinidad global c_s(i,j)
   • Binning de relaciones por ranking de afinidad

3. ENTITY-LEVEL AGGREGATION (Sección 5.2)
   • Extensión de GATv2 con vectores de relación
   • Agregación de vecinos + relaciones
   • Self-loops con relaciones promediadas

4. TRAINING REGIME (Sección 5.4)
   • División dinámica (Ftr/Ttr por época)
   • Re-inicialización de features
   • Margin-based ranking loss

5. SCORING FUNCTION (Sección 5.3)
   • Variante de DistMult
   • Transformación W para compatibilidad dimensional
""")

print("📊 ARCHIVOS GENERADOS:")
print(f"  1. {output_dir / 'relation_graph_affinity.png'}")
print(f"  2. {output_dir / 'training_curve.png'}")
print(f"  3. {output_dir / 'relation_embeddings_pca.png'}")
print(f"  4. {output_dir / 'affinity_weights.png'}")

print("""
🎓 APRENDIZAJES CLAVE:

• Grafo de relaciones captura semántica implícita
• División dinámica crucial para generalización
• Pesos de afinidad permiten interpolación de relaciones nuevas
• Arquitectura dual (relación + entidad) es esencial

🚀 PRÓXIMOS PASOS:

1. Integrar con KGDataLoader y UnifiedKGScorer reales
2. Evaluar en datasets benchmark (FB15k-237, WN18RR)
3. Comparar con baselines (GraIL, RMPI, RED-GNN)
4. Extensiones: grafos temporales, pre-entrenamiento
""")

print("\n" + "="*80)
print("Demo completada. Archivos disponibles en:", output_dir)
print("="*80)
