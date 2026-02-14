#!/usr/bin/env python3
"""
GUÍA DE INICIO RÁPIDO - TransE Implementation

Este script muestra cómo usar la implementación de TransE paso a paso.
"""

print("""
╔════════════════════════════════════════════════════════════════════════╗
║           TransE: Translating Embeddings for Multi-relational Data    ║
║                    Implementación Fiel al Paper Original              ║
║                        (Bordes et al., 2013)                           ║
╚════════════════════════════════════════════════════════════════════════╝
""")

print("\n📚 ARCHIVOS GENERADOS:\n")
print("1. transe_model.py          - Implementación completa del modelo")
print("2. README_TransE.md         - Documentación detallada")
print("3. run_experiments.py       - Configuraciones predefinidas")
print("4. NOTAS_TECNICAS.md        - Análisis técnico y comparación")
print("5. inicio_rapido.py         - Este archivo")

print("\n" + "="*70)
print("OPCIÓN 1: EJECUCIÓN ESTÁNDAR")
print("="*70)

print("""
# Editar configuración en transe_model.py (función main):
DATASET_NAME = 'CoDEx-M'
MODE = 'standard'  # 'standard', 'ookb', o 'inductive'

# Ejecutar:
python transe_model.py
""")

print("="*70)
print("OPCIÓN 2: USAR CONFIGURACIONES PREDEFINIDAS")
print("="*70)

print("""
# Listar configuraciones disponibles:
python run_experiments.py list

# Ejecutar configuración específica:
python run_experiments.py codex_standard   # Transductivo
python run_experiments.py codex_ookb       # Entidades nuevas
python run_experiments.py wordnet          # WordNet (paper)
python run_experiments.py freebase         # Freebase (paper)

# Estudio comparativo (3 escenarios):
python run_experiments.py comparative

# Estudio de ablation (impacto de hiperparámetros):
python run_experiments.py ablation
""")

print("="*70)
print("OPCIÓN 3: USO PROGRAMÁTICO")
print("="*70)

print("""
from transe_model import TransE, train_transe
from data_loader import KGDataLoader
from evaluator import UnifiedKGScorer
import torch

# 1. Cargar datos
loader = KGDataLoader('CoDEx-M', mode='standard')
loader.load()

# 2. Crear modelo
model = TransE(
    num_entities=loader.num_entities,
    num_relations=loader.num_relations,
    embedding_dim=50,
    norm_order=1,  # L1
    margin=1.0,
    device='cuda'
)

# 3. Entrenar
model, history = train_transe(
    model=model,
    train_data=loader.train_data,
    valid_data=loader.valid_data,
    num_entities=loader.num_entities,
    num_epochs=1000,
    batch_size=128,
    learning_rate=0.01
)

# 4. Evaluar
scorer = UnifiedKGScorer()

def predict_fn(h, r, t):
    model.eval()
    with torch.no_grad():
        return model.score_triples(h, r, t)

metrics = scorer.evaluate_ranking(
    predict_fn=predict_fn,
    test_triples=loader.test_data.cpu().numpy(),
    num_entities=loader.num_entities
)

# 5. Generar reporte
scorer.export_report("TransE", "reporte.pdf")
""")

print("="*70)
print("CONFIGURACIONES DEL PAPER")
print("="*70)

print("""
WordNet (WN):
  - embedding_dim: 20
  - learning_rate: 0.01
  - margin: 2.0
  - norm_order: 1 (L1)
  - Hits@10 esperado: ~89%

Freebase (FB15k):
  - embedding_dim: 50
  - learning_rate: 0.01
  - margin: 1.0
  - norm_order: 1 (L1)
  - Hits@10 esperado: ~47%

Large Scale (FB1M):
  - embedding_dim: 50
  - learning_rate: 0.01
  - margin: 1.0
  - norm_order: 2 (L2)
  - Hits@10 esperado: ~34%
""")

print("="*70)
print("ESTRUCTURA DE ARCHIVOS REQUERIDA")
print("="*70)

print("""
Su estructura de datos debe ser:

data/
├── newlinks/              # Datasets transductivos
│   ├── CoDEx-M/
│   │   ├── train.txt
│   │   ├── valid.txt
│   │   └── test.txt
│   ├── FB15k-237/
│   └── WN18RR/
│
├── newentities/           # Datasets OOKB
│   └── CoDEx-M/
│       ├── train.txt      # Entidades conocidas
│       ├── valid.txt
│       └── test.txt       # Incluye entidades nuevas
│
└── newlinks/              # Datasets inductivos
    └── CoDEx-M/
        └── NL-25/         # 25% relaciones nuevas
            ├── train.txt
            ├── valid.txt
            └── test.txt

Formato de los archivos .txt (TSV):
head_entity<TAB>relation<TAB>tail_entity
Paris<TAB>capital_of<TAB>France
Einstein<TAB>born_in<TAB>Germany
""")

print("="*70)
print("SALIDAS GENERADAS")
print("="*70)

print("""
Al ejecutar, se generan:

1. Terminal output:
   - Progreso de entrenamiento (loss, MRR validation)
   - Métricas finales (MRR, Hits@K, AUC, F1)

2. Archivo PDF:
   TransE_<dataset>_<mode>_reporte.pdf
   
   Contenido:
   - Página 1: Resumen ejecutivo con métricas
   - Página 2: Curvas ROC y Precision-Recall
   - Página 3: Distribución de scores (separabilidad)
   - Página 4: Histograma de ranks

3. Modelo entrenado:
   (Puede guardarse con torch.save si se desea)
""")

print("="*70)
print("MÉTRICAS REPORTADAS")
print("="*70)

print("""
Ranking (Link Prediction):
  - MRR (Mean Reciprocal Rank): Promedio de 1/rank
  - MR (Mean Rank): Rank promedio de la entidad correcta
  - Hits@1: % de predicciones correctas en top-1
  - Hits@3: % de predicciones correctas en top-3
  - Hits@10: % de predicciones correctas en top-10

Clasificación (Triple Classification):
  - AUC-ROC: Área bajo curva ROC
  - Accuracy: % de tripletas correctamente clasificadas
  - F1-Score: Media armónica de precision y recall
  - Precision: TP / (TP + FP)
  - Recall: TP / (TP + FN)
""")

print("="*70)
print("TROUBLESHOOTING")
print("="*70)

print("""
Problema: "FileNotFoundError: No se encontró train.txt"
Solución: Verificar que la estructura de carpetas data/ sea correcta

Problema: "RuntimeError: CUDA out of memory"
Solución: Reducir batch_size o usar device='cpu'

Problema: "KeyError en entity2id"
Solución: En modo OOKB, esto es normal. El código lo maneja automáticamente.

Problema: Rendimiento muy bajo en OOKB
Solución: Esto es esperado. TransE no fue diseñado para OOKB.
          El embedding especial es solo una baseline.

Problema: Loss no baja
Solución: 
  - Verificar que normalize_entity_embeddings() se llama
  - Probar diferentes learning rates (0.001, 0.01, 0.1)
  - Verificar que el margin no sea muy grande
""")

print("="*70)
print("PREGUNTAS FRECUENTES")
print("="*70)

print("""
Q: ¿Por qué usar -||h+r-t|| en lugar de ||h+r-t||?
A: El evaluador espera scores donde MAYOR es MEJOR.
   Distancia pequeña = buena predicción → score negativo alto.

Q: ¿Cuándo normalizar las entidades?
A: ANTES de cada época (línea 5 del Algoritmo 1).
   NO después del gradiente.

Q: ¿Por qué no normalizar las relaciones durante entrenamiento?
A: Solo se normalizan en inicialización (línea 2).
   El paper NO las renormaliza después.

Q: ¿Qué hacer con entidades OOKB?
A: El código usa un embedding especial (unknown_entity_embedding).
   El rendimiento será bajo, como se espera.

Q: ¿Cómo se compara con métodos modernos en OOKB?
A: TransE será mucho peor que GNN-based encoders.
   Eso es el objetivo: establecer baseline para comparación.

Q: ¿Puedo usar TransE para mi dataset?
A: Sí, solo necesitas archivos .txt con formato:
   head<TAB>relation<TAB>tail
""")

print("="*70)
print("LECTURAS RECOMENDADAS")
print("="*70)

print("""
1. Paper original:
   Bordes et al., "Translating Embeddings for Modeling Multi-relational Data"
   NIPS 2013

2. README_TransE.md
   Explicación detallada de cada componente vs el paper

3. NOTAS_TECNICAS.md
   Análisis de diferencias, validación, y extensiones

4. Paper de comparación (OOKB):
   Hamaguchi et al., "Knowledge Transfer for Out-of-Knowledge-Base Entities"
   IJCAI 2017
""")

print("\n" + "="*70)
print("¡LISTO PARA EMPEZAR!")
print("="*70)
print("\nEjecutar uno de estos comandos:\n")
print("  python transe_model.py")
print("  python run_experiments.py codex_standard")
print("  python run_experiments.py list")
print("\n" + "="*70 + "\n")
