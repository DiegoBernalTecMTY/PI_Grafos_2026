"""
TransE: Translating Embeddings for Modeling Multi-relational Data
Implementación basada en Bordes et al., 2013 (NIPS)

Referencia del Paper:
- Modelo: h + r ≈ t (relaciones como traslaciones en espacio embedding)
- Loss: Margin-based ranking loss con negative sampling
- Score: d(h, r, t) = -||h + r - t|| (menor es mejor)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import random


# ============================================================================
# 1. DATASET PERSONALIZADO PARA TRIPLETAS
# ============================================================================

class TripleDataset(Dataset):
    """
    Dataset para manejar tripletas de Knowledge Graph.
    
    Paper (Sección 2, Algoritmo 1):
    - Entrada: Conjunto de tripletas S = {(h, l, t)}
    - Durante entrenamiento, generamos negativos corruptos para cada positivo
    """
    def __init__(self, triples, num_entities):
        """
        Args:
            triples: Tensor [N, 3] con (head_id, relation_id, tail_id)
            num_entities: Número total de entidades (para negative sampling)
        """
        self.triples = triples
        self.num_entities = num_entities
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        """
        Retorna una tripleta positiva.
        El negative sampling se hace en el collate_fn del DataLoader.
        """
        return self.triples[idx]


# ============================================================================
# 2. MODELO TransE
# ============================================================================

class TransE(nn.Module):
    """
    TransE: Modelo de embeddings translacionales.
    
    Paper (Sección 2):
    - Entidades y relaciones se representan como vectores en R^k
    - Función de energía: d(h, r, t) = ||h + r - t||_p
    - p puede ser L1 o L2 (seleccionado por validación)
    
    Restricciones (Algoritmo 1, líneas 2 y 5):
    - Embeddings de relaciones se normalizan SOLO en inicialización
    - Embeddings de entidades se normalizan CADA iteración antes del batch
    """
    
    def __init__(self, num_entities, num_relations, embedding_dim=50, 
                 norm_order=1, margin=1.0, device='cuda'):
        """
        Args:
            num_entities: Número de entidades en el grafo
            num_relations: Número de relaciones
            embedding_dim: Dimensión de los embeddings (k en el paper)
            norm_order: 1 para L1, 2 para L2 (seleccionado en validación)
            margin: γ en la loss function (típicamente 1 o 2)
            device: 'cuda' o 'cpu'
        """
        super(TransE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.norm_order = norm_order
        self.margin = margin
        self.device = device
        
        # Paper (Algoritmo 1, líneas 1 y 3):
        # Inicialización uniforme en [-√(6/k), √(6/k)]
        # Esta es la inicialización de Glorot & Bengio (2010) - referencia [4] del paper
        init_bound = np.sqrt(6.0 / self.embedding_dim)
        
        # Embeddings de entidades (línea 3 del Algoritmo 1)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -init_bound, init_bound)
        
        # Embeddings de relaciones (línea 1 del Algoritmo 1)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.uniform_(self.relation_embeddings.weight, -init_bound, init_bound)
        
        # Normalizar relaciones SOLO en inicialización (línea 2 del Algoritmo 1)
        with torch.no_grad():
            self.relation_embeddings.weight.data = nn.functional.normalize(
                self.relation_embeddings.weight.data, p=2, dim=1
            )
        
        # Para manejar entidades OOKB (Out-Of-Knowledge-Base)
        # Usamos un embedding especial para entidades desconocidas
        self.unknown_entity_embedding = nn.Parameter(
            torch.randn(embedding_dim) * init_bound
        )
        
    def normalize_entity_embeddings(self):
        """
        Normaliza los embeddings de entidades a norma L2 = 1.
        
        Paper (Algoritmo 1, línea 5):
        "e ← e/||e|| for each entity e ∈ E"
        
        IMPORTANTE: Esto se hace ANTES de cada batch, no después.
        Previene que el modelo trivialmente minimice la loss aumentando las normas.
        """
        with torch.no_grad():
            self.entity_embeddings.weight.data = nn.functional.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
    
    def get_embeddings(self, heads, relations, tails, handle_ookb=True):
        """
        Obtiene embeddings para tripletas, manejando entidades desconocidas.
        
        Args:
            heads: Tensor [batch_size] con IDs de entidades head
            relations: Tensor [batch_size] con IDs de relaciones
            tails: Tensor [batch_size] con IDs de entidades tail
            handle_ookb: Si True, reemplaza IDs >= num_entities con embedding especial
            
        Returns:
            h_emb, r_emb, t_emb: Tensores [batch_size, embedding_dim]
        """
        if handle_ookb:
            # Identificar entidades fuera del vocabulario
            # Esto ocurre en escenarios OOKB donde el test tiene entidades nuevas
            ookb_mask_h = heads >= self.num_entities
            ookb_mask_t = tails >= self.num_entities
            
            # Clonar para evitar modificar los originales
            safe_heads = heads.clone()
            safe_tails = tails.clone()
            
            # Reemplazar IDs inválidos con 0 temporalmente (para no romper el embedding)
            safe_heads[ookb_mask_h] = 0
            safe_tails[ookb_mask_t] = 0
            
            # Obtener embeddings
            h_emb = self.entity_embeddings(safe_heads)
            t_emb = self.entity_embeddings(safe_tails)
            
            # Reemplazar con embedding desconocido donde corresponda
            h_emb[ookb_mask_h] = self.unknown_entity_embedding.unsqueeze(0).expand(
                ookb_mask_h.sum(), -1
            )
            t_emb[ookb_mask_t] = self.unknown_entity_embedding.unsqueeze(0).expand(
                ookb_mask_t.sum(), -1
            )
        else:
            # Modo estándar sin manejo de OOKB
            h_emb = self.entity_embeddings(heads)
            t_emb = self.entity_embeddings(tails)
        
        # Las relaciones nunca son OOKB en nuestros datasets
        r_emb = self.relation_embeddings(relations)
        
        return h_emb, r_emb, t_emb
    
    def score_triples(self, heads, relations, tails):
        """
        Calcula el score de energía para tripletas.
        
        Paper (Sección 2):
        Score: d(h, r, t) = ||h + r - t||_p
        
        IMPORTANTE: Menor score = mejor (más plausible la tripleta)
        Por eso retornamos el negativo para compatibilidad con evaluación.
        
        Args:
            heads, relations, tails: Tensors de IDs [batch_size]
            
        Returns:
            scores: Tensor [batch_size] con -d(h,r,t) (mayor es mejor)
        """
        h_emb, r_emb, t_emb = self.get_embeddings(heads, relations, tails)
        
        # Paper: h + r ≈ t  →  queremos ||h + r - t|| pequeño
        translation = h_emb + r_emb - t_emb
        
        # Distancia según norma configurada (L1 o L2)
        distance = torch.norm(translation, p=self.norm_order, dim=1)
        
        # Retornamos el negativo porque menor distancia = mejor score
        return -distance
    
    def forward(self, pos_heads, pos_rels, pos_tails, 
                neg_heads, neg_rels, neg_tails):
        """
        Forward pass para calcular la loss.
        
        Paper (Ecuación 1):
        L = Σ Σ [γ + d(h,r,t) - d(h',r,t')]_+
        
        Donde:
        - (h,r,t) son tripletas positivas (reales)
        - (h',r,t') son tripletas negativas (corruptas)
        - [x]_+ = max(0, x) (parte positiva)
        - γ es el margen
        """
        # Scores para tripletas positivas
        pos_scores = self.score_triples(pos_heads, pos_rels, pos_tails)
        
        # Scores para tripletas negativas
        neg_scores = self.score_triples(neg_heads, neg_rels, neg_tails)
        
        # Margin Ranking Loss
        # Paper (Ecuación 1): [γ + d(h,r,t) - d(h',r,t')]_+
        # Como usamos scores = -distancia, esto se convierte en:
        # [γ - pos_score + neg_score]_+ = [γ + (-pos_score) - (-neg_score)]_+
        loss = torch.relu(self.margin - pos_scores + neg_scores).mean()
        
        return loss


# ============================================================================
# 3. FUNCIONES DE ENTRENAMIENTO
# ============================================================================

def corrupt_batch(pos_triples, num_entities, device):
    """
    Genera tripletas negativas corrompiendo heads o tails.
    
    Paper (Ecuación 2):
    S'_(h,r,t) = {(h', r, t) | h' ∈ E} ∪ {(h, r, t') | t' ∈ E}
    
    Estrategia (Algoritmo 1, línea 9):
    - Para cada tripleta positiva, generamos UNA tripleta corrupta
    - Corrompemos aleatoriamente el head O el tail (no ambos)
    - Esto balancea la corrupción entre entidades
    
    Args:
        pos_triples: Tensor [batch_size, 3] con tripletas positivas
        num_entities: Número total de entidades
        device: torch device
        
    Returns:
        neg_triples: Tensor [batch_size, 3] con tripletas corruptas
    """
    batch_size = pos_triples.size(0)
    neg_triples = pos_triples.clone()
    
    # Máscara aleatoria: True = corromper head, False = corromper tail
    corrupt_head_mask = torch.rand(batch_size, device=device) < 0.5
    
    # Entidades aleatorias para reemplazo
    random_entities = torch.randint(0, num_entities, (batch_size,), device=device)
    
    # Corromper heads donde la máscara es True
    neg_triples[corrupt_head_mask, 0] = random_entities[corrupt_head_mask]
    
    # Corromper tails donde la máscara es False
    neg_triples[~corrupt_head_mask, 2] = random_entities[~corrupt_head_mask]
    
    return neg_triples


def train_transe(model, train_data, valid_data, num_entities,
                 num_epochs=1000, batch_size=128, learning_rate=0.01,
                 eval_every=50, patience=5, device='cuda'):
    """
    Entrena el modelo TransE con early stopping.
    
    Paper (Algoritmo 1):
    - Normalizar entidades antes de cada batch (línea 5)
    - Samplear minibatch (línea 6)
    - Generar negativos (línea 9)
    - Actualizar con SGD (línea 12)
    
    Args:
        model: Instancia de TransE
        train_data: Tensor de tripletas de entrenamiento
        valid_data: Tensor de tripletas de validación
        num_entities: Número de entidades
        num_epochs: Máximo de épocas
        batch_size: Tamaño del batch
        learning_rate: Learning rate para SGD
        eval_every: Evaluar en validación cada N épocas
        patience: Épocas sin mejora antes de early stopping
        device: 'cuda' o 'cpu'
        
    Returns:
        model: Modelo entrenado
        history: Dict con métricas de entrenamiento
    """
    model = model.to(device)
    
    # Optimizer: SGD según el paper (Algoritmo 1)
    # El paper usa SGD estándar con learning rate constante
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Dataset y DataLoader
    train_dataset = TripleDataset(train_data, num_entities)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Importante: shuffle para SGD estocástico
        num_workers=0
    )
    
    # Para early stopping
    best_valid_mrr = 0.0
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'valid_mrr': []
    }
    
    print(f"Iniciando entrenamiento de TransE...")
    print(f"  Entidades: {num_entities}, Relaciones: {model.num_relations}")
    print(f"  Dimensión: {model.embedding_dim}, Norma: L{model.norm_order}, Margen: {model.margin}")
    print(f"  Epochs: {num_epochs}, Batch size: {batch_size}, LR: {learning_rate}\n")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Paper (Algoritmo 1, línea 5):
        # Normalizar embeddings de entidades ANTES de la época
        model.normalize_entity_embeddings()
        
        # Iterar sobre batches
        for pos_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                              leave=False, disable=(epoch % eval_every != 0)):
            pos_batch = pos_batch.to(device)
            
            # Paper (Algoritmo 1, línea 9):
            # Generar tripletas corruptas
            neg_batch = corrupt_batch(pos_batch, num_entities, device)
            
            # Extraer componentes
            pos_h, pos_r, pos_t = pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2]
            neg_h, neg_r, neg_t = neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2]
            
            # Forward pass
            loss = model(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Evaluación periódica
        if (epoch + 1) % eval_every == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            # Evaluación rápida en validación (solo MRR para early stopping)
            model.eval()
            valid_mrr = quick_evaluate_mrr(model, valid_data, num_entities, 
                                          batch_size=256, device=device)
            history['valid_mrr'].append(valid_mrr)
            
            print(f"  Valid MRR: {valid_mrr:.4f}")
            
            # Early stopping
            if valid_mrr > best_valid_mrr:
                best_valid_mrr = valid_mrr
                epochs_without_improvement = 0
                # Guardar mejor modelo
                best_model_state = model.state_dict().copy()
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping: {patience} épocas sin mejora")
                # Restaurar mejor modelo
                model.load_state_dict(best_model_state)
                break
    
    print(f"\nEntrenamiento completado. Mejor Valid MRR: {best_valid_mrr:.4f}")
    return model, history


def quick_evaluate_mrr(model, test_data, num_entities, 
                       batch_size=256, max_samples=1000, device='cuda'):
    """
    Evaluación rápida de MRR para early stopping.
    
    Evalúa solo en un subconjunto de test_data para ahorrar tiempo.
    La evaluación completa se hace al final con UnifiedKGScorer.
    """
    model.eval()
    
    # Subsamplear para evaluación rápida
    if len(test_data) > max_samples:
        indices = torch.randperm(len(test_data))[:max_samples]
        test_subset = test_data[indices]
    else:
        test_subset = test_data
    
    test_subset = test_subset.to(device)
    ranks = []
    
    with torch.no_grad():
        for i in range(0, len(test_subset), batch_size):
            batch = test_subset[i:i+batch_size]
            heads = batch[:, 0]
            rels = batch[:, 1]
            tails = batch[:, 2]
            
            # Score de la tripleta correcta
            pos_scores = model.score_triples(heads, rels, tails)
            
            # Scores contra todas las entidades (tail corruption)
            batch_size_actual = len(batch)
            expanded_heads = heads.unsqueeze(1).repeat(1, num_entities).view(-1)
            expanded_rels = rels.unsqueeze(1).repeat(1, num_entities).view(-1)
            all_tails = torch.arange(num_entities, device=device).repeat(batch_size_actual)
            
            all_scores = model.score_triples(expanded_heads, expanded_rels, all_tails)
            all_scores = all_scores.view(batch_size_actual, num_entities)
            
            # Calcular ranks (mayor score = mejor)
            for j in range(batch_size_actual):
                target_score = pos_scores[j]
                better_count = (all_scores[j] > target_score).sum().item()
                ranks.append(better_count + 1)
    
    mrr = np.mean([1.0 / r for r in ranks])
    return mrr


# ============================================================================
# 4. SCRIPT PRINCIPAL DE ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

def main():
    """
    Script principal que ejecuta el pipeline completo:
    1. Carga de datos
    2. Entrenamiento de TransE
    3. Evaluación exhaustiva (Ranking + Classification)
    4. Generación de reporte PDF
    """
    
    # Importar los módulos proporcionados
    import sys
    sys.path.append('.')
    from data_loader import KGDataLoader
    from evaluator import UnifiedKGScorer
    
    # ========================================================================
    # CONFIGURACIÓN
    # ========================================================================
    
    # Dataset: 'CoDEx-M', 'FB15k-237', 'WN18RR'
    # Modo: 'standard' (transductivo), 'ookb' (entidades nuevas), 'inductive' (relaciones nuevas)
    DATASET_NAME = 'CoDEx-M'
    MODE = 'ookb'  # Cambiar a 'standard' o 'inductive' según necesidad
    INDUCTIVE_SPLIT = 'NL-25'  # Solo para mode='inductive'
    
    # Hiperparámetros del modelo (basados en el paper)
    # Paper (Sección 4.2):
    # - WN: k=20, λ=0.01, γ=2, d=L1
    # - FB15k: k=50, λ=0.01, γ=1, d=L1
    EMBEDDING_DIM = 50
    LEARNING_RATE = 0.01
    MARGIN = 1.0
    NORM_ORDER = 1  # 1 para L1, 2 para L2
    
    # Hiperparámetros de entrenamiento
    NUM_EPOCHS = 1000
    BATCH_SIZE = 128
    EVAL_EVERY = 50
    PATIENCE = 5
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {DEVICE}\n")
    
    # ========================================================================
    # 1. CARGA DE DATOS
    # ========================================================================
    
    print("="*70)
    print("PASO 1: CARGA DE DATOS")
    print("="*70)
    
    loader = KGDataLoader(
        dataset_name=DATASET_NAME,
        mode=MODE,
        inductive_split=INDUCTIVE_SPLIT if MODE == 'inductive' else None
    )
    loader.load()
    
    # Extraer datos
    train_data = loader.train_data
    valid_data = loader.valid_data
    test_data = loader.test_data
    num_entities = loader.num_entities
    num_relations = loader.num_relations
    
    print(f"\nDatos cargados exitosamente:")
    print(f"  Train: {len(train_data)} tripletas")
    print(f"  Valid: {len(valid_data)} tripletas")
    print(f"  Test: {len(test_data)} tripletas")
    print(f"  Entidades: {num_entities}")
    print(f"  Relaciones: {num_relations}")
    
    # ========================================================================
    # 2. ENTRENAMIENTO
    # ========================================================================
    
    print("\n" + "="*70)
    print("PASO 2: ENTRENAMIENTO DEL MODELO TransE")
    print("="*70)
    
    # Inicializar modelo
    model = TransE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=EMBEDDING_DIM,
        norm_order=NORM_ORDER,
        margin=MARGIN,
        device=DEVICE
    )
    
    # Entrenar
    model, history = train_transe(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        num_entities=num_entities,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_every=EVAL_EVERY,
        patience=PATIENCE,
        device=DEVICE
    )
    
    # ========================================================================
    # 3. EVALUACIÓN EXHAUSTIVA
    # ========================================================================
    
    print("\n" + "="*70)
    print("PASO 3: EVALUACIÓN EXHAUSTIVA")
    print("="*70)
    
    # Función de predicción para el evaluador
    def predict_fn(heads, rels, tails):
        """
        Wrapper para compatibilidad con UnifiedKGScorer.
        
        IMPORTANTE: El evaluador espera scores donde MAYOR es MEJOR.
        TransE produce -distancia, así que ya cumple con esto.
        """
        model.eval()
        with torch.no_grad():
            scores = model.score_triples(heads, rels, tails)
        return scores
    
    # Inicializar evaluador
    scorer = UnifiedKGScorer(device=DEVICE)
    
    # -----------------------------------------------------------------------
    # 3A. RANKING EVALUATION (Link Prediction)
    # -----------------------------------------------------------------------
    
    print("\n[A] Evaluación de Ranking (Link Prediction)")
    print("-" * 70)
    
    ranking_metrics = scorer.evaluate_ranking(
        predict_fn=predict_fn,
        test_triples=test_data.cpu().numpy(),
        num_entities=num_entities,
        batch_size=128,
        k_values=[1, 3, 10],
        higher_is_better=True,  # Scores de TransE: mayor = mejor
        verbose=True
    )
    
    print("\nResultados de Ranking:")
    print(f"  MRR:     {ranking_metrics['mrr']:.4f}")
    print(f"  MR:      {ranking_metrics['mr']:.2f}")
    print(f"  Hits@1:  {ranking_metrics['hits@1']:.4f}")
    print(f"  Hits@3:  {ranking_metrics['hits@3']:.4f}")
    print(f"  Hits@10: {ranking_metrics['hits@10']:.4f}")
    
    # -----------------------------------------------------------------------
    # 3B. TRIPLE CLASSIFICATION
    # -----------------------------------------------------------------------
    
    print("\n[B] Evaluación de Clasificación (Triple Classification)")
    print("-" * 70)
    
    classification_metrics = scorer.evaluate_classification(
        predict_fn=predict_fn,
        valid_pos=valid_data.cpu().numpy(),
        test_pos=test_data.cpu().numpy(),
        num_entities=num_entities,
        higher_is_better=True
    )
    
    print("\nResultados de Clasificación:")
    print(f"  AUC:       {classification_metrics['auc']:.4f}")
    print(f"  Accuracy:  {classification_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {classification_metrics['f1']:.4f}")
    
    # ========================================================================
    # 4. GENERACIÓN DE REPORTE
    # ========================================================================
    
    print("\n" + "="*70)
    print("PASO 4: GENERACIÓN DE REPORTE PDF")
    print("="*70)
    
    model_name = f"TransE (dim={EMBEDDING_DIM}, L{NORM_ORDER}, γ={MARGIN}) - {DATASET_NAME} ({MODE})"
    report_filename = f"TransE_{DATASET_NAME}_{MODE}_reporte.pdf"
    
    scorer.export_report(
        model_name=model_name,
        filename=report_filename
    )
    
    # ========================================================================
    # 5. ANÁLISIS ADICIONAL (OOKB)
    # ========================================================================
    
    if MODE == 'ookb':
        print("\n" + "="*70)
        print("ANÁLISIS ADICIONAL: Out-Of-Knowledge-Base (OOKB)")
        print("="*70)
        
        unknown_entities = loader.get_unknown_entities_mask()
        print(f"\nEntidades desconocidas en test: {len(unknown_entities)}")
        print(f"Porcentaje: {100 * len(unknown_entities) / num_entities:.2f}%")
        
        # Nota: El modelo ya maneja esto automáticamente usando unknown_entity_embedding
        print("\nNota: TransE usa un embedding especial para entidades OOKB.")
        print("Esto permite evaluar sin errores, aunque el rendimiento será bajo.")
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    
    print("\n" + "="*70)
    print("RESUMEN FINAL")
    print("="*70)
    
    print(f"\nDataset: {DATASET_NAME} ({MODE})")
    print(f"Modelo: TransE")
    print(f"  - Dimensión embeddings: {EMBEDDING_DIM}")
    print(f"  - Norma: L{NORM_ORDER}")
    print(f"  - Margen: {MARGIN}")
    
    print(f"\nMétricas de Ranking:")
    print(f"  - MRR:     {ranking_metrics['mrr']:.4f}")
    print(f"  - Hits@10: {ranking_metrics['hits@10']:.4f}")
    
    print(f"\nMétricas de Clasificación:")
    print(f"  - AUC:      {classification_metrics['auc']:.4f}")
    print(f"  - Accuracy: {classification_metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {classification_metrics['f1']:.4f}")
    
    print(f"\nReporte guardado en: {report_filename}")
    print("\n" + "="*70)
    print("EJECUCIÓN COMPLETADA")
    print("="*70)


if __name__ == "__main__":
    # Configurar semilla para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Ejecutar pipeline completo
    main()
