"""
INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs
Implementación basada en Lee et al., 2023 (ICML)

Este modelo permite el aprendizaje Zero-Shot de relaciones nuevas mediante:
1. Construcción de un Grafo de Relaciones basado en co-ocurrencia de entidades
2. Agregación atencional a nivel de relación (Relation-Level Aggregation)
3. Agregación atencional a nivel de entidad (Entity-Level Aggregation)
4. División dinámica durante entrenamiento para mayor generalización
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class RelationGraphBuilder:
    """
    Construye el Grafo de Relaciones según la Sección 4 del paper.
    
    El grafo de relaciones es un grafo ponderado donde:
    - Cada nodo representa una relación
    - Los pesos de las aristas representan la afinidad entre relaciones
    - La afinidad se calcula basándose en cuántas entidades comparten
    
    Proceso (Paper Sección 4):
    1. Crear matrices Eh y Et que registran frecuencias de (entidad, relación)
    2. Normalizar por grado de entidad: Ah = Eh^T @ Dh^(-2) @ Eh
    3. Combinar: A = Ah + At (matriz de adyacencia del grafo de relaciones)
    """
    
    def __init__(self, num_entities: int, num_relations: int):
        self.num_entities = num_entities
        self.num_relations = num_relations
        
    def build(self, triplets: torch.Tensor) -> torch.Tensor:
        """
        Construye la matriz de adyacencia del grafo de relaciones.
        
        Args:
            triplets: Tensor de forma (num_triplets, 3) con formato (head, rel, tail)
            
        Returns:
            A: Matriz de adyacencia (num_relations, num_relations)
        
        Paper Ecuación: A = Ah + At donde:
        - Ah = Eh^T @ Dh^(-2) @ Eh
        - At = Et^T @ Dt^(-2) @ Et
        """
        device = triplets.device
        
        # Paso 1: Crear matrices Eh y Et (Paper Sección 4)
        # Eh[i, j] = frecuencia de entidad i apareciendo como head de relación j
        # Et[i, j] = frecuencia de entidad i apareciendo como tail de relación j
        Eh = torch.zeros(self.num_entities, self.num_relations, device=device)
        Et = torch.zeros(self.num_entities, self.num_relations, device=device)
        
        heads, rels, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        
        # Contar frecuencias
        for h, r, t in zip(heads, rels, tails):
            Eh[h, r] += 1.0
            Et[t, r] += 1.0
        
        # Paso 2: Calcular matrices de grado Dh y Dt (Paper Sección 4)
        # Dh[i, i] = suma de frecuencias de entidad i como head
        # La normalización Dh^(-2) permite que la suma de pesos por entidad = 1
        Dh_diag = Eh.sum(dim=1)  # Grado de cada entidad como head
        Dt_diag = Et.sum(dim=1)  # Grado de cada entidad como tail
        
        # Evitar división por cero
        Dh_diag = torch.clamp(Dh_diag, min=1e-8)
        Dt_diag = torch.clamp(Dt_diag, min=1e-8)
        
        # Dh^(-2): normalización cuadrática inversa
        Dh_inv2 = 1.0 / (Dh_diag ** 2)
        Dt_inv2 = 1.0 / (Dt_diag ** 2)
        
        # Paso 3: Calcular Ah y At (Paper Ecuación en Sección 4)
        # Aplicar normalización: cada entidad contribuye equitativamente
        Eh_normalized = Eh * Dh_inv2.unsqueeze(1)
        Et_normalized = Et * Dt_inv2.unsqueeze(1)
        
        # Ah = Eh^T @ Dh^(-2) @ Eh (simplificado porque ya normalizamos)
        Ah = Eh.t() @ Eh_normalized
        At = Et.t() @ Et_normalized
        
        # Paso 4: Combinar para obtener matriz de adyacencia final (Paper Sección 4)
        # A[i,j] = afinidad entre relación i y relación j
        A = Ah + At
        
        # Añadir self-loops (cada relación es vecina de sí misma)
        A = A + torch.eye(self.num_relations, device=device)
        
        return A


class RelationLevelAggregation(nn.Module):
    """
    Agregación a Nivel de Relación mediante Atención (Paper Sección 5.1).
    
    Actualiza las representaciones de relaciones agregando información
    de relaciones vecinas usando mecanismo de atención con:
    1. Atención basada en representaciones locales (α_ij en Ecuación 2)
    2. Pesos de afinidad global (c_s(i,j) en Ecuación 2 y 3)
    
    Diferencia clave vs GATv2: incorpora pesos de afinidad global del grafo
    de relaciones para reflejar la importancia estructural de cada vecino.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_bins: int = 10, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_bins = num_bins
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible por num_heads"
        
        # Parámetros para atención (Paper Ecuación 2)
        # P^(l): matriz de transformación para concatenación [z_i || z_j]
        self.P = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        
        # y^(l): vector de pesos para calcular score de atención
        # Aplicado DESPUÉS de σ(·) para resolver static attention (Brody et al., 2022)
        self.y = nn.Linear(hidden_dim, num_heads, bias=False)
        
        # W^(l): matriz de transformación para actualización
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # c_s(i,j): parámetros aprendibles para binning de afinidad (Paper Ecuación 2-3)
        # Un parámetro por cada bin de afinidad
        self.c_bins = nn.Parameter(torch.randn(num_bins, num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Residual connection (Paper Sección 5.1)
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, z: torch.Tensor, A: torch.Tensor, 
                neighbor_indices: torch.Tensor,
                affinity_bins: torch.Tensor) -> torch.Tensor:
        """
        Actualiza representaciones de relaciones mediante agregación atencional.
        
        Args:
            z: Representaciones de relaciones (num_relations, hidden_dim)
            A: Matriz de adyacencia del grafo de relaciones (num_relations, num_relations)
            neighbor_indices: Índices de vecinos para cada relación (num_relations, max_neighbors)
            affinity_bins: Bins de afinidad para pesos c_s(i,j) (num_relations, max_neighbors)
            
        Returns:
            z_new: Representaciones actualizadas (num_relations, hidden_dim)
            
        Implementa Ecuación 1 del paper:
        z_i^(l+1) = σ(Σ_{r_j ∈ N_i} α_ij^(l) W^(l) z_j^(l))
        """
        num_relations = z.size(0)
        batch_size = num_relations
        
        # Para cada relación, calcular atención con sus vecinos
        z_updated = []
        
        for i in range(num_relations):
            # Obtener vecinos de la relación i (incluyendo self-loop)
            neighbors = neighbor_indices[i]
            valid_mask = neighbors >= 0  # Máscara para vecinos válidos (padding = -1)
            
            if valid_mask.sum() == 0:
                # Si no hay vecinos, mantener representación actual
                z_updated.append(z[i].unsqueeze(0))
                continue
            
            neighbors = neighbors[valid_mask]
            z_neighbors = z[neighbors]  # (num_neighbors, hidden_dim)
            z_i = z[i].unsqueeze(0).expand(len(neighbors), -1)  # (num_neighbors, hidden_dim)
            
            # Calcular coeficientes de atención α_ij (Paper Ecuación 2)
            # Paso 1: Concatenar z_i y z_j
            z_concat = torch.cat([z_i, z_neighbors], dim=1)  # (num_neighbors, 2*hidden_dim)
            
            # Paso 2: Aplicar transformación lineal P^(l)
            h = self.P(z_concat)  # (num_neighbors, hidden_dim)
            
            # Paso 3: Aplicar activación LeakyReLU
            h = self.leaky_relu(h)
            
            # Paso 4: Calcular scores de atención con y^(l) (multi-head)
            attn_scores = self.y(h)  # (num_neighbors, num_heads)
            
            # Paso 5: Añadir pesos de afinidad c_s(i,j) (Paper Ecuación 2-3)
            # s(i,j) determina el bin basado en rank de afinidad
            bins = affinity_bins[i][valid_mask]  # (num_neighbors,)
            c_weights = self.c_bins[bins]  # (num_neighbors, num_heads)
            
            attn_scores = attn_scores + c_weights
            
            # Paso 6: Softmax para normalizar (por cada head)
            attn_weights = F.softmax(attn_scores, dim=0)  # (num_neighbors, num_heads)
            attn_weights = self.dropout(attn_weights)
            
            # Paso 7: Aplicar transformación W^(l) a vecinos
            z_transformed = self.W(z_neighbors)  # (num_neighbors, hidden_dim)
            
            # Paso 8: Agregación multi-head
            # Reshape para multi-head: (num_neighbors, num_heads, head_dim)
            z_transformed = z_transformed.view(len(neighbors), self.num_heads, self.head_dim)
            attn_weights = attn_weights.unsqueeze(2)  # (num_neighbors, num_heads, 1)
            
            # Weighted sum para cada head
            z_aggregated = (attn_weights * z_transformed).sum(dim=0)  # (num_heads, head_dim)
            z_aggregated = z_aggregated.view(-1)  # (hidden_dim,)
            
            # Paso 9: Residual connection (Paper Sección 5.1)
            z_new = self.leaky_relu(z_aggregated + self.residual_weight * z[i])
            
            z_updated.append(z_new.unsqueeze(0))
        
        return torch.cat(z_updated, dim=0)


class EntityLevelAggregation(nn.Module):
    """
    Agregación a Nivel de Entidad (Paper Sección 5.2).
    
    Actualiza representaciones de entidades agregando:
    1. Representaciones de entidades vecinas
    2. Representaciones de relaciones que conectan con los vecinos
    3. Su propia representación con relaciones adyacentes promediadas
    
    Extensión de GATv2 que incorpora vectores de relación en cada paso
    de agregación (Paper Ecuación 4).
    """
    
    def __init__(self, entity_dim: int, relation_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.num_heads = num_heads
        self.head_dim = entity_dim // num_heads
        
        assert entity_dim % num_heads == 0, "entity_dim debe ser divisible por num_heads"
        
        # Transformación para [h_i || z_k] (entidad + relación)
        # Paper Ecuación 4: Wc^(l) transforma la concatenación
        self.Wc = nn.Linear(entity_dim + relation_dim, entity_dim, bias=False)
        
        # Atención: P̂^(l) para [h_i || h_j || z_k]
        self.P_hat = nn.Linear(2 * entity_dim + relation_dim, entity_dim, bias=False)
        
        # ŷ^(l): vector de pesos para score de atención
        self.y_hat = nn.Linear(entity_dim, num_heads, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, h: torch.Tensor, z: torch.Tensor, 
                edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Actualiza representaciones de entidades mediante agregación atencional.
        
        Args:
            h: Representaciones de entidades (num_entities, entity_dim)
            z: Representaciones de relaciones (num_relations, relation_dim)
            edge_index: Aristas del KG (2, num_edges) formato [source, target]
            edge_type: Tipo de relación para cada arista (num_edges,)
            
        Returns:
            h_new: Representaciones actualizadas (num_entities, entity_dim)
            
        Implementa Ecuación 4 del paper:
        h_i^(l+1) = σ(β_ii Wc^(l)[h_i^(l) || z̄_i^(L)] + 
                      Σ β_ijk Wc^(l)[h_j^(l) || z_k^(L)])
        """
        num_entities = h.size(0)
        device = h.device
        
        # Construir diccionario de vecinos para cada entidad
        # neighbor_dict[i] = lista de (vecino_j, relacion_k)
        neighbor_dict = {i: [] for i in range(num_entities)}
        
        for idx in range(edge_index.size(1)):
            src, dst = edge_index[0, idx].item(), edge_index[1, idx].item()
            rel = edge_type[idx].item()
            # En el paper, vecinos son entrantes: (vj, rk, vi) ∈ F
            neighbor_dict[dst].append((src, rel))
        
        h_updated = []
        
        for i in range(num_entities):
            neighbors = neighbor_dict[i]
            
            if len(neighbors) == 0:
                # Sin vecinos: solo self-loop con promedio de relaciones vacío
                # En práctica, esto no debería ocurrir en un grafo conectado
                h_updated.append(h[i].unsqueeze(0))
                continue
            
            # Calcular z̄_i: promedio de representaciones de relaciones adyacentes (Paper Sección 5.2)
            neighbor_entities = [n[0] for n in neighbors]
            neighbor_relations = [n[1] for n in neighbors]
            
            z_neighbors = z[neighbor_relations]  # (num_neighbors, relation_dim)
            z_bar_i = z_neighbors.mean(dim=0, keepdim=True)  # (1, relation_dim)
            
            # Self-loop: β_ii con [h_i || z̄_i]
            h_i = h[i].unsqueeze(0)  # (1, entity_dim)
            h_self_concat = torch.cat([h_i, z_bar_i], dim=1)  # (1, entity_dim + relation_dim)
            
            # Neighbor aggregation: β_ijk con [h_j || z_k]
            h_neighbors = h[neighbor_entities]  # (num_neighbors, entity_dim)
            h_neighbor_concat = torch.cat([h_neighbors, z_neighbors], dim=1)  # (num_neighbors, entity_dim + relation_dim)
            
            # Combinar self-loop y neighbors para calcular atención
            # b_ii = [h_i || h_i || z̄_i]
            # b_ijk = [h_i || h_j || z_k]
            h_i_expanded = h_i.expand(len(neighbors), -1)  # (num_neighbors, entity_dim)
            
            b_self = torch.cat([h_i, h_i, z_bar_i], dim=1)  # (1, 2*entity_dim + relation_dim)
            b_neighbors = torch.cat([h_i_expanded, h_neighbors, z_neighbors], dim=1)  # (num_neighbors, 2*entity_dim + relation_dim)
            
            # Calcular scores de atención (Paper: β_ii y β_ijk)
            attn_self = self.y_hat(self.leaky_relu(self.P_hat(b_self)))  # (1, num_heads)
            attn_neighbors = self.y_hat(self.leaky_relu(self.P_hat(b_neighbors)))  # (num_neighbors, num_heads)
            
            # Concatenar y aplicar softmax
            attn_all = torch.cat([attn_self, attn_neighbors], dim=0)  # (1 + num_neighbors, num_heads)
            attn_weights = F.softmax(attn_all, dim=0)  # (1 + num_neighbors, num_heads)
            attn_weights = self.dropout(attn_weights)
            
            # Separar pesos
            attn_self_weight = attn_weights[0:1]  # (1, num_heads)
            attn_neighbor_weights = attn_weights[1:]  # (num_neighbors, num_heads)
            
            # Aplicar transformación Wc a las concatenaciones
            h_self_transformed = self.Wc(h_self_concat)  # (1, entity_dim)
            h_neighbor_transformed = self.Wc(h_neighbor_concat)  # (num_neighbors, entity_dim)
            
            # Combinar para multi-head aggregation
            h_all_transformed = torch.cat([h_self_transformed, h_neighbor_transformed], dim=0)  # (1 + num_neighbors, entity_dim)
            
            # Reshape para multi-head
            h_all_transformed = h_all_transformed.view(-1, self.num_heads, self.head_dim)  # (1 + num_neighbors, num_heads, head_dim)
            attn_weights_expanded = attn_weights.unsqueeze(2)  # (1 + num_neighbors, num_heads, 1)
            
            # Weighted sum
            h_aggregated = (attn_weights_expanded * h_all_transformed).sum(dim=0)  # (num_heads, head_dim)
            h_aggregated = h_aggregated.view(-1)  # (entity_dim,)
            
            # Residual connection y activación
            h_new = self.leaky_relu(h_aggregated + self.residual_weight * h[i])
            
            h_updated.append(h_new.unsqueeze(0))
        
        return torch.cat(h_updated, dim=0)


class INGRAM(nn.Module):
    """
    INGRAM: INductive knowledge GRAph eMbedding
    
    Modelo completo que combina:
    1. Relation Graph Builder (Sección 4)
    2. Relation-Level Aggregation (Sección 5.1)
    3. Entity-Level Aggregation (Sección 5.2)
    4. Relation-Entity Interaction Modeling (Sección 5.3)
    
    Capacidad clave: Generar embeddings de relaciones y entidades NUEVAS
    en tiempo de inferencia mediante agregación de vecinos.
    """
    
    def __init__(self,
                 num_entities: int,
                 num_relations: int,
                 entity_dim: int = 32,
                 relation_dim: int = 32,
                 entity_hidden_dim: int = 128,
                 relation_hidden_dim: int = 64,
                 num_relation_layers: int = 2,
                 num_entity_layers: int = 3,
                 num_relation_heads: int = 8,
                 num_entity_heads: int = 8,
                 num_bins: int = 10,
                 dropout: float = 0.1):
        """
        Args:
            num_entities: Número de entidades en el grafo
            num_relations: Número de relaciones en el grafo
            entity_dim: Dimensión de embeddings finales de entidades (d̂ en paper)
            relation_dim: Dimensión de embeddings finales de relaciones (d en paper)
            entity_hidden_dim: Dimensión oculta para entidades (d̂' en paper)
            relation_hidden_dim: Dimensión oculta para relaciones (d' en paper)
            num_relation_layers: L en paper (capas de agregación de relaciones)
            num_entity_layers: L̂ en paper (capas de agregación de entidades)
            num_relation_heads: K en paper (heads de atención para relaciones)
            num_entity_heads: K̂ en paper (heads de atención para entidades)
            num_bins: B en paper (número de bins para afinidad)
            dropout: Tasa de dropout
        """
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity_dim = entity_dim
        self.relation_dim = relation_dim
        self.entity_hidden_dim = entity_hidden_dim
        self.relation_hidden_dim = relation_hidden_dim
        self.num_relation_layers = num_relation_layers
        self.num_entity_layers = num_entity_layers
        
        # Paper Sección 5.1: Proyección inicial de features aleatorios a espacio oculto
        # H: R^{d × d'} proyecta features de relaciones
        self.relation_feature_proj = nn.Linear(relation_dim, relation_hidden_dim)
        
        # Paper Sección 5.2: Proyección inicial de features de entidades
        # Ĥ: R^{d̂ × d̂'} proyecta features de entidades
        self.entity_feature_proj = nn.Linear(entity_dim, entity_hidden_dim)
        
        # Capas de agregación a nivel de relación (Paper Sección 5.1)
        self.relation_layers = nn.ModuleList([
            RelationLevelAggregation(
                hidden_dim=relation_hidden_dim,
                num_heads=num_relation_heads,
                num_bins=num_bins,
                dropout=dropout
            ) for _ in range(num_relation_layers)
        ])
        
        # Capas de agregación a nivel de entidad (Paper Sección 5.2)
        self.entity_layers = nn.ModuleList([
            EntityLevelAggregation(
                entity_dim=entity_hidden_dim,
                relation_dim=relation_hidden_dim,
                num_heads=num_entity_heads,
                dropout=dropout
            ) for _ in range(num_entity_layers)
        ])
        
        # Paper Sección 5.3: Proyecciones finales para embeddings
        # M: R^{d × d'} proyecta representaciones de relaciones a embeddings finales
        self.relation_output_proj = nn.Linear(relation_hidden_dim, relation_dim)
        
        # M̂: R^{d̂ × d̂'} proyecta representaciones de entidades a embeddings finales
        self.entity_output_proj = nn.Linear(entity_hidden_dim, entity_dim)
        
        # Paper Sección 5.3: Matriz W para scoring function
        # W: R^{d̂ × d} convierte dimensión de relación a dimensión de entidad
        self.scoring_weight = nn.Parameter(torch.randn(entity_dim, relation_dim))
        nn.init.xavier_uniform_(self.scoring_weight)
        
        # Relation Graph Builder (Paper Sección 4)
        self.relation_graph_builder = RelationGraphBuilder(num_entities, num_relations)
        
    def init_features(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inicializa features aleatorios usando Glorot initialization.
        
        Paper Sección 5.4: "We randomly re-initialize all feature vectors per epoch
        during training, INGRAM learns how to compute embedding vectors using random
        features, and this is beneficial for computing embeddings with random features
        at inference time."
        
        Esta estrategia permite que el modelo aprenda a generalizar independientemente
        de los valores iniciales específicos.
        """
        # Glorot initialization para entidades
        entity_features = torch.empty(self.num_entities, self.entity_dim, device=device)
        nn.init.xavier_uniform_(entity_features)
        
        # Glorot initialization para relaciones
        relation_features = torch.empty(self.num_relations, self.relation_dim, device=device)
        nn.init.xavier_uniform_(relation_features)
        
        return entity_features, relation_features
    
    def build_relation_graph(self, triplets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construye el grafo de relaciones y estructuras auxiliares para agregación.
        
        Returns:
            A: Matriz de adyacencia (num_relations, num_relations)
            neighbor_indices: Índices de vecinos para cada relación (num_relations, max_neighbors)
            affinity_bins: Bins de afinidad para cada vecino (num_relations, max_neighbors)
        """
        # Construir matriz de adyacencia del grafo de relaciones (Paper Sección 4)
        A = self.relation_graph_builder.build(triplets)
        
        # Preparar estructuras para agregación eficiente
        num_relations = A.size(0)
        device = A.device
        
        # Encontrar vecinos (relaciones con afinidad > 0) para cada relación
        neighbor_lists = []
        affinity_lists = []
        max_neighbors = 0
        
        for i in range(num_relations):
            # Obtener afinidades no-cero
            affinities = A[i]
            nonzero_mask = affinities > 0
            neighbors = torch.where(nonzero_mask)[0]
            neighbor_affinities = affinities[neighbors]
            
            neighbor_lists.append(neighbors)
            affinity_lists.append(neighbor_affinities)
            max_neighbors = max(max_neighbors, len(neighbors))
        
        # Crear tensores paddeados
        neighbor_indices = torch.full((num_relations, max_neighbors), -1, 
                                     dtype=torch.long, device=device)
        affinity_values = torch.zeros((num_relations, max_neighbors), device=device)
        
        for i, (neighbors, affinities) in enumerate(zip(neighbor_lists, affinity_lists)):
            neighbor_indices[i, :len(neighbors)] = neighbors
            affinity_values[i, :len(neighbors)] = affinities
        
        # Calcular bins de afinidad según Paper Ecuación 3
        # s(i,j) = ⌊rank(a_ij) × B / nnz(A)⌋
        # donde rank(a_ij) es el ranking de a_ij en orden descendente
        affinity_bins = self._compute_affinity_bins(A, neighbor_indices)
        
        return A, neighbor_indices, affinity_bins
    
    def _compute_affinity_bins(self, A: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        Computa bins de afinidad según Paper Ecuación 3.
        
        Paper: "We divide the relation pairs into B different bins according to
        their affinity scores. Each relation pair has an index value of 1 ≤ s(i,j) ≤ B"
        
        Relaciones con alta afinidad → bin pequeño (s(i,j) cercano a 1)
        Relaciones con baja afinidad → bin grande (s(i,j) cercano a B)
        """
        num_relations, max_neighbors = neighbor_indices.shape
        device = A.device
        
        # Obtener todos los valores de afinidad no-cero y ordenarlos
        nonzero_affinities = A[A > 0]
        sorted_affinities, _ = torch.sort(nonzero_affinities, descending=True)
        
        num_bins = self.relation_layers[0].num_bins
        nnz = len(nonzero_affinities)
        
        # Crear bins
        affinity_bins = torch.zeros_like(neighbor_indices)
        
        for i in range(num_relations):
            for j in range(max_neighbors):
                neighbor_idx = neighbor_indices[i, j]
                
                if neighbor_idx < 0:  # Padding
                    continue
                
                affinity = A[i, neighbor_idx]
                
                if affinity == 0:
                    continue
                
                # Encontrar rank de esta afinidad
                rank = (sorted_affinities > affinity).sum().item() + 1
                
                # Calcular bin según Ecuación 3
                bin_idx = int((rank * num_bins) / nnz)
                bin_idx = min(bin_idx, num_bins - 1)  # Asegurar que esté en rango [0, B-1]
                
                affinity_bins[i, j] = bin_idx
        
        return affinity_bins
    
    def forward(self, triplets: torch.Tensor, 
                entity_features: Optional[torch.Tensor] = None,
                relation_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass completo de INGRAM.
        
        Args:
            triplets: Tripletas del grafo (num_triplets, 3)
            entity_features: Features iniciales de entidades (opcional, se inicializan aleatoriamente si no se proveen)
            relation_features: Features iniciales de relaciones (opcional)
            
        Returns:
            entity_embeddings: Embeddings finales de entidades (num_entities, entity_dim)
            relation_embeddings: Embeddings finales de relaciones (num_relations, relation_dim)
        """
        device = triplets.device
        
        # Inicializar features si no se proveen (Paper Sección 5.4)
        if entity_features is None or relation_features is None:
            entity_features, relation_features = self.init_features(device)
        
        # PASO 1: Construir grafo de relaciones (Paper Sección 4)
        A, neighbor_indices, affinity_bins = self.build_relation_graph(triplets)
        
        # PASO 2: Proyectar features a espacio oculto
        # Paper Sección 5.1: z^(0)_i = H x_i
        z = self.relation_feature_proj(relation_features)  # (num_relations, relation_hidden_dim)
        
        # Paper Sección 5.2: h^(0)_i = Ĥ x̂_i
        h = self.entity_feature_proj(entity_features)  # (num_entities, entity_hidden_dim)
        
        # PASO 3: Agregación a nivel de relación (Paper Sección 5.1)
        # Actualizar z^(l) para l = 0, ..., L-1
        for layer in self.relation_layers:
            z = layer(z, A, neighbor_indices, affinity_bins)
        
        # z ahora contiene z^(L) - representaciones finales de nivel de relación
        
        # PASO 4: Preparar edge_index y edge_type para agregación de entidades
        # Formato: edge_index[0] = source, edge_index[1] = target
        edge_index = torch.stack([triplets[:, 0], triplets[:, 2]], dim=0)
        edge_type = triplets[:, 1]
        
        # PASO 5: Agregación a nivel de entidad (Paper Sección 5.2)
        # Actualizar h^(l) para l = 0, ..., L̂-1
        # Nota: Siempre usamos z^(L) (representaciones finales de relaciones)
        for layer in self.entity_layers:
            h = layer(h, z, edge_index, edge_type)
        
        # h ahora contiene h^(L̂) - representaciones finales de nivel de entidad
        
        # PASO 6: Proyección a embeddings finales (Paper Sección 5.3)
        # z_k := M z^(L)_k para relaciones
        relation_embeddings = self.relation_output_proj(z)
        
        # h_i := M̂ h^(L̂)_i para entidades
        entity_embeddings = self.entity_output_proj(h)
        
        return entity_embeddings, relation_embeddings
    
    def score(self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor,
              entity_embeddings: torch.Tensor, relation_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calcula score de plausibilidad para tripletas.
        
        Paper Ecuación 5: f(v_i, r_k, v_j) = h_i^T diag(W z_k) h_j
        
        Esta es una variante de DistMult que incorpora la transformación W
        para convertir dimensión de relación a dimensión de entidad.
        
        Args:
            head: Índices de entidades head (batch_size,)
            relation: Índices de relaciones (batch_size,)
            tail: Índices de entidades tail (batch_size,)
            entity_embeddings: Embeddings de entidades (num_entities, entity_dim)
            relation_embeddings: Embeddings de relaciones (num_relations, relation_dim)
            
        Returns:
            scores: Scores de plausibilidad (batch_size,)
        """
        # Obtener embeddings
        h_i = entity_embeddings[head]  # (batch_size, entity_dim)
        z_k = relation_embeddings[relation]  # (batch_size, relation_dim)
        h_j = entity_embeddings[tail]  # (batch_size, entity_dim)
        
        # Aplicar transformación W: d × d̂ × d̂
        # W z_k: (batch_size, relation_dim) @ (entity_dim, relation_dim)^T → (batch_size, entity_dim)
        Wz_k = torch.matmul(z_k, self.scoring_weight.t())  # (batch_size, entity_dim)
        
        # Calcular score: h_i^T diag(W z_k) h_j
        # Equivalente a: sum(h_i * W z_k * h_j) elemento a elemento
        scores = (h_i * Wz_k * h_j).sum(dim=1)  # (batch_size,)
        
        return scores


class INGRAMTrainer:
    """
    Entrenador para INGRAM con división dinámica y re-inicialización.
    
    Paper Sección 5.4: Training Regime
    - División dinámica de Ftr y Ttr en cada época (ratio 3:1)
    - Re-inicialización de features en cada época
    - Restricciones: Ftr contiene árbol de expansión mínimo y todas las relaciones
    """
    
    def __init__(self, model: INGRAM, lr: float = 0.001, margin: float = 1.0):
        """
        Args:
            model: Modelo INGRAM
            lr: Learning rate
            margin: Margen γ para margin-based ranking loss (Paper Sección 5.3)
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.margin = margin
        
    def dynamic_split(self, all_triplets: torch.Tensor, 
                      num_entities: int, num_relations: int,
                      train_ratio: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        División dinámica de tripletas en Ftr (facts) y Ttr (training targets).
        
        Paper Sección 5.4: "For every epoch, we randomly re-split Ftr and Ttr with
        the minimal constraint that Ftr includes the minimum spanning tree of Gtr
        and Ftr covers all relations in Rtr so that all entity and relation embedding
        vectors are appropriately learned."
        
        Restricciones:
        1. Ftr debe contener árbol de expansión mínimo (conectividad)
        2. Ftr debe cubrir todas las relaciones (para que todas se aprendan)
        3. Ratio aproximado 3:1 (Ftr:Ttr)
        """
        device = all_triplets.device
        num_triplets = len(all_triplets)
        
        # Paso 1: Asegurar que todas las relaciones están representadas en Ftr
        relation_coverage = {}
        for r in range(num_relations):
            mask = all_triplets[:, 1] == r
            if mask.sum() > 0:
                # Tomar al menos una tripleta de cada relación para Ftr
                rel_triplets = all_triplets[mask]
                relation_coverage[r] = rel_triplets[0].unsqueeze(0)
        
        ftr_triplets = list(relation_coverage.values())
        used_indices = set()
        
        for r in range(num_relations):
            if r in relation_coverage:
                # Encontrar índice de esta tripleta en all_triplets
                rel_triplet = relation_coverage[r][0]
                for idx, triplet in enumerate(all_triplets):
                    if torch.all(triplet == rel_triplet):
                        used_indices.add(idx)
                        break
        
        # Paso 2: Construir árbol de expansión mínimo (simplificado con BFS)
        # Esto asegura que el grafo Ftr sea conexo
        entity_visited = set()
        queue = []
        
        # Iniciar desde entidades en relation_coverage
        for triplets in relation_coverage.values():
            h, r, t = triplets[0]
            entity_visited.add(h.item())
            entity_visited.add(t.item())
            queue.append((h.item(), r.item(), t.item()))
        
        # BFS para añadir tripletas que conecten nuevas entidades
        remaining_indices = [i for i in range(num_triplets) if i not in used_indices]
        
        while len(entity_visited) < num_entities and remaining_indices:
            added = False
            for idx in remaining_indices[:]:
                h, r, t = all_triplets[idx]
                h_in = h.item() in entity_visited
                t_in = t.item() in entity_visited
                
                # Añadir si conecta una entidad nueva con una existente
                if (h_in and not t_in) or (t_in and not h_in):
                    ftr_triplets.append(all_triplets[idx].unsqueeze(0))
                    entity_visited.add(h.item())
                    entity_visited.add(t.item())
                    used_indices.add(idx)
                    remaining_indices.remove(idx)
                    added = True
                    break
            
            if not added:
                break  # No se pueden añadir más sin crear ciclos
        
        # Paso 3: Completar Ftr hasta el ratio deseado
        target_ftr_size = int(num_triplets * train_ratio)
        remaining_indices = [i for i in range(num_triplets) if i not in used_indices]
        
        if len(ftr_triplets) < target_ftr_size and remaining_indices:
            # Seleccionar aleatoriamente más tripletas
            additional_count = min(target_ftr_size - len(ftr_triplets), len(remaining_indices))
            perm = torch.randperm(len(remaining_indices))[:additional_count]
            
            for i in perm:
                idx = remaining_indices[i]
                ftr_triplets.append(all_triplets[idx].unsqueeze(0))
                used_indices.add(idx)
        
        # Paso 4: Ttr = tripletas restantes
        ttr_indices = [i for i in range(num_triplets) if i not in used_indices]
        
        Ftr = torch.cat(ftr_triplets, dim=0) if ftr_triplets else torch.empty(0, 3, device=device)
        Ttr = all_triplets[ttr_indices] if ttr_indices else torch.empty(0, 3, device=device)
        
        return Ftr, Ttr
    
    def generate_negatives(self, positive_triplets: torch.Tensor, 
                          num_entities: int, num_negatives: int = 10) -> torch.Tensor:
        """
        Genera tripletas negativas corrompiendo heads o tails.
        
        Paper Sección 5.3: "We create negative triplets by corrupting a head or
        a tail entity of a positive triplet."
        """
        device = positive_triplets.device
        num_pos = len(positive_triplets)
        
        negatives = []
        
        for _ in range(num_negatives):
            neg_triplets = positive_triplets.clone()
            
            # Decidir aleatoriamente si corromper head o tail (50/50)
            corrupt_head = torch.rand(num_pos, device=device) < 0.5
            
            # Generar entidades aleatorias
            random_entities = torch.randint(0, num_entities, (num_pos,), device=device)
            
            # Corromper heads
            neg_triplets[corrupt_head, 0] = random_entities[corrupt_head]
            
            # Corromper tails
            neg_triplets[~corrupt_head, 2] = random_entities[~corrupt_head]
            
            negatives.append(neg_triplets)
        
        return torch.cat(negatives, dim=0)
    
    def train_epoch(self, all_triplets: torch.Tensor, 
                    num_entities: int, num_relations: int,
                    batch_size: int = 128) -> float:
        """
        Entrena una época con división dinámica y re-inicialización.
        
        Returns:
            avg_loss: Loss promedio de la época
        """
        self.model.train()
        device = next(self.model.parameters()).device
        
        # PASO 1: División dinámica (Paper Sección 5.4)
        Ftr, Ttr = self.dynamic_split(all_triplets, num_entities, num_relations)
        
        if len(Ttr) == 0:
            return 0.0
        
        # Combinar Ftr y Ttr para construir el grafo completo
        # (necesario para construir el grafo de relaciones)
        full_graph = torch.cat([Ftr, Ttr], dim=0)
        
        # PASO 2: Forward pass con features aleatorios re-inicializados
        # Paper Sección 5.4: "At the beginning of each epoch, we initialize all
        # feature vectors using Glorot initialization."
        entity_embeddings, relation_embeddings = self.model(full_graph)
        
        # PASO 3: Generar negativos
        num_negatives = 10
        negative_triplets = self.generate_negatives(Ttr, num_entities, num_negatives)
        
        # PASO 4: Calcular loss en batches
        num_batches = (len(Ttr) + batch_size - 1) // batch_size
        total_loss = 0.0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(Ttr))
            
            # Batch de positivos
            pos_batch = Ttr[start_idx:end_idx]
            pos_heads, pos_rels, pos_tails = pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2]
            
            # Batch de negativos correspondiente
            neg_start = start_idx * num_negatives
            neg_end = end_idx * num_negatives
            neg_batch = negative_triplets[neg_start:neg_end]
            neg_heads, neg_rels, neg_tails = neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2]
            
            # Calcular scores
            pos_scores = self.model.score(pos_heads, pos_rels, pos_tails,
                                         entity_embeddings, relation_embeddings)
            neg_scores = self.model.score(neg_heads, neg_rels, neg_tails,
                                         entity_embeddings, relation_embeddings)
            
            # Margin-based ranking loss (Paper Sección 5.3)
            # L = Σ max(0, γ - f(v_i, r_k, v_j) + f(v̊_i, r_k, v̊_j))
            # Expandir pos_scores para comparar con todos los negativos
            pos_scores_expanded = pos_scores.repeat_interleave(num_negatives)
            
            loss = F.relu(self.margin - pos_scores_expanded + neg_scores).mean()
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / num_batches


def create_predict_fn(model: INGRAM, entity_embeddings: torch.Tensor, 
                      relation_embeddings: torch.Tensor):
    """
    Crea función de predicción para el evaluador.
    
    Args:
        model: Modelo INGRAM entrenado
        entity_embeddings: Embeddings de entidades
        relation_embeddings: Embeddings de relaciones
        
    Returns:
        predict_fn: Función que toma (heads, rels, tails) y retorna scores
    """
    def predict_fn(heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scores = model.score(heads, rels, tails, entity_embeddings, relation_embeddings)
        return scores
    
    return predict_fn


if __name__ == "__main__":
    print("="*80)
    print("INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs")
    print("Implementación basada en Lee et al., 2023 (ICML)")
    print("="*80)
    
    # Test básico
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDispositivo: {device}")
    
    # Crear modelo de prueba
    num_entities = 100
    num_relations = 20
    
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
        num_bins=10
    ).to(device)
    
    print(f"\nModelo creado con:")
    print(f"  - {num_entities} entidades")
    print(f"  - {num_relations} relaciones")
    print(f"  - {sum(p.numel() for p in model.parameters())} parámetros totales")
    
    # Generar grafo sintético
    num_triplets = 500
    triplets = torch.randint(0, num_entities, (num_triplets, 3), device=device)
    triplets[:, 1] = torch.randint(0, num_relations, (num_triplets,), device=device)
    
    print(f"\n  - {num_triplets} tripletas sintéticas generadas")
    
    # Forward pass
    print("\nEjecutando forward pass...")
    entity_embeddings, relation_embeddings = model(triplets)
    
    print(f"  ✓ Entity embeddings: {entity_embeddings.shape}")
    print(f"  ✓ Relation embeddings: {relation_embeddings.shape}")
    
    # Test scoring
    test_heads = torch.tensor([0, 1, 2], device=device)
    test_rels = torch.tensor([0, 1, 2], device=device)
    test_tails = torch.tensor([3, 4, 5], device=device)
    
    scores = model.score(test_heads, test_rels, test_tails, entity_embeddings, relation_embeddings)
    print(f"  ✓ Scores de prueba: {scores}")
    
    print("\n✓ Test básico completado exitosamente!")
    print("="*80)
