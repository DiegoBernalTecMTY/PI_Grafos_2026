"""
End-to-End DBPedia50k+ Training Pipeline for IKGE
===========================================

This script loads the DBPedia50k+ dataset, initializes the IKGE model, runs the
training loop with negative sampling, evaluates with UnifiedKGScorer, and saves weights.
"""

import os
import sys
import time
import argparse
import urllib.request
import tarfile
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# Fix relative imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from line_graph import create_line_graph
from fact_feature_extractor import FactFeatureExtractor
from attentive_aggregator import AttentiveAggregator
from download_w2v import setup_w2v_for_ikge, tokenize_for_w2v

try:
    from evaluation_utils import UnifiedKGScorer
except ImportError:
    print("Warning: evaluation_utils.py not found. Evaluation metrics may fail.")


# ==============================================================================
# Logging helper  – duplicates every print() to a timestamped log file
# ==============================================================================

class TeeLogger:
    """Redirect stdout so every print() goes to both the terminal and a log file."""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w', encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        sys.stdout = self.terminal  # restore before closing
        self.log.close()


# ==============================================================================
# 1. Dataset Downloading and Parsing
# ==============================================================================

def get_dataset_dir(dataset_dir=None):
    """Validates the existence of the expected DBPedia50k+ dataset."""
    if dataset_dir is None:
        # Resolve relative to this script: ikge/ikge/train_ikge.py → ikge/ikge/data/DBPedia50k+
        dataset_dir = str(Path(__file__).resolve().parent / 'data' / 'DBPedia50k+')
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}. Please run generate_dbpedia50k.py first.")
    return dataset_dir


class FB20kDataset(Dataset):
    """Dataset for training with Negative Sampling."""
    def __init__(self, triples, num_entities):
        self.triples = triples
        self.num_entities = num_entities
        
    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        
        # Negative sampling (corrupt either head or tail)
        neg_h, neg_t = h, t
        if random.random() < 0.5:
            neg_h = random.randint(0, self.num_entities - 1)
        else:
            neg_t = random.randint(0, self.num_entities - 1)
            
        return {
            'pos': torch.tensor([h, r, t], dtype=torch.long),
            'neg': torch.tensor([neg_h, r, neg_t], dtype=torch.long)
        }


# ==============================================================================
# 2. Main IKGE Model Architecture
# ==============================================================================

class IKGENetwork(nn.Module):
    """
    End-to-end IKGE Network combining:
    1. Line Graph & Neighborhoods
    2. FactFeatureExtractor
    3. AttentiveAggregator
    4. 2-Layer MLP Scoring Function
    """
    def __init__(self, embedding_matrix, word_emb_dim, fact_emb_dim, conv_channels, num_types, num_layers, device, dropout=0.25):
        super(IKGENetwork, self).__init__()
        self.device = device

        # Extract features from textual description
        self.fact_extractor = FactFeatureExtractor(
            word_embedding_matrix=embedding_matrix,
            word_embedding_dim=word_emb_dim,
            fact_embedding_dim=fact_emb_dim,
            conv_channels=conv_channels,
            num_types=num_types,
            dropout=dropout,  # paper: 0.25
            device=device
        )
        # Word embeddings remain frozen per paper Section 5.1.1.
        # (FactFeatureExtractor already sets freeze=True; no override needed.)

        # Aggregate graph neighbourhood (K=3 for DBPedia50k+ per paper)
        self.aggregator = AttentiveAggregator(
            fact_embedding_dim=fact_emb_dim,
            num_layers=num_layers,
            device=device
        )

        # Scoring function: LayerNorm → Dropout → 2-FC layers → logit (no sigmoid).
        # LayerNorm is critical: 3 residual aggregation layers accumulate feature
        # magnitudes to ~3-4× the CNN output range.  Kaiming-init MLP weights are
        # calibrated for unit-variance inputs, so unnormalized aggregator outputs
        # produce logits of ±15, saturating sigmoid and making BCE unbounded.
        # LayerNorm resets magnitude to unit-variance before every MLP call.
        # sigmoid is applied ONLY at inference/scoring time; training uses
        # F.binary_cross_entropy_with_logits for numerically stable gradients.
        self.score_norm   = nn.LayerNorm(fact_emb_dim)
        self.score_drop   = nn.Dropout(dropout)
        self.score_layer1 = nn.Linear(fact_emb_dim, 512)
        self.relu         = nn.ReLU()
        self.score_layer2 = nn.Linear(512, 256)
        self.score_out    = nn.Linear(256, 1)
        
        self.to(device)
        
    def extract_fact_features(self, batch_data):
        return self.fact_extractor(
            head_descriptions=batch_data['head_desc'],
            tail_descriptions=batch_data['tail_desc'],
            head_names=batch_data['head_name'],
            tail_names=batch_data['tail_name'],
            relation_names=batch_data['rel_name'],
            head_types=batch_data['head_type'],
            tail_types=batch_data['tail_type'],
            relation_domain_types=batch_data['rel_domain'],
            relation_range_types=batch_data['rel_range'],
            relation_domain_words=batch_data['rel_domain_words'],
            relation_range_words=batch_data['rel_range_words'],
            head_desc_lengths=batch_data['head_len'],
            tail_desc_lengths=batch_data['tail_len']
        )

    def forward(self, features, return_logits: bool = False):
        """
        Score facts via LayerNorm + Dropout + 2-layer MLP.

        Returns sigmoid probabilities (0-1) by default (for ranking/evaluation).
        Pass return_logits=True to get raw logits (use with F.binary_cross_entropy_with_logits
        during training for numerically stable gradients).

        Paper Eq 12: w(z) = sigmoid(W_f2 * ReLU(W_f1 * z + b_f1) + b_f2)
        Section 6.1.3: "2 layers with 512, 256 dimensions"
        Architecture: d -[LN]-> -[drop]-> -[W_f1]-> 512 -[ReLU]-> 256 -[W_f2(256→1)]
        One ReLU only, between score_layer1 and score_layer2, as per Eq 12.
        """
        x = self.score_norm(features)    # normalise accumulated residual magnitudes
        x = self.score_drop(x)
        x = self.score_layer1(x)
        x = self.relu(x)
        x = self.score_layer2(x)
        logit = self.score_out(x).squeeze(-1)
        if return_logits:
            return logit
        return torch.sigmoid(logit)


# ==============================================================================
# 3. Training & Evaluation Pipeline
# ==============================================================================

def prepare_batch_tensors(triples, entity2desc, relation2name, entity2types, rel2domain, rel2range, word2idx, type2idx, max_desc_len, device):
    """
    Converts a batch of string triples into tokenized tensors.

    entity2types: dict[entity_str -> list[type_str]]  (multi-hot support)
    """
    batch_size = len(triples)

    # Initialize tensors
    head_desc = torch.zeros(batch_size, max_desc_len, dtype=torch.long)
    tail_desc = torch.zeros(batch_size, max_desc_len, dtype=torch.long)
    head_len  = torch.ones(batch_size, dtype=torch.long)
    tail_len  = torch.ones(batch_size, dtype=torch.long)
    rel_name  = torch.zeros(batch_size, 10, dtype=torch.long)  # up to 10 tokens for relation names

    num_types  = len(type2idx)
    head_type  = torch.zeros(batch_size, num_types, dtype=torch.float)
    tail_type  = torch.zeros(batch_size, num_types, dtype=torch.float)
    rel_domain = torch.zeros(batch_size, num_types, dtype=torch.float)
    rel_range  = torch.zeros(batch_size, num_types, dtype=torch.float)

    for i, (h, r, t) in enumerate(triples):
        # Tokenize text fields
        h_words = tokenize_for_w2v(entity2desc.get(h, 'unknown'))[:max_desc_len]
        t_words = tokenize_for_w2v(entity2desc.get(t, 'unknown'))[:max_desc_len]
        r_words = tokenize_for_w2v(relation2name.get(r, r.split('/')[-1].replace('_', ' ')))[:10]

        for j, w in enumerate(h_words): head_desc[i, j] = word2idx.get(w, 1)  # 1 = <UNK>
        for j, w in enumerate(t_words): tail_desc[i, j] = word2idx.get(w, 1)
        for j, w in enumerate(r_words): rel_name[i, j]  = word2idx.get(w, 1)

        head_len[i] = max(1, len(h_words))
        tail_len[i] = max(1, len(t_words))

        # Multi-hot entity type encoding (FIX: entity can have multiple types)
        for typ in entity2types.get(h, []):
            if typ in type2idx:
                head_type[i, type2idx[typ]] = 1.0
        for typ in entity2types.get(t, []):
            if typ in type2idx:
                tail_type[i, type2idx[typ]] = 1.0

        # Relation type constraint encoding (multi-type support)
        for typ in rel2domain.get(r, []):
            if typ in type2idx:
                rel_domain[i, type2idx[typ]] = 1.0
        for typ in rel2range.get(r, []):
            if typ in type2idx:
                rel_range[i, type2idx[typ]] = 1.0

    return {
        'head_desc':  head_desc.to(device),
        'tail_desc':  tail_desc.to(device),
        'head_len':   head_len.to(device),
        'tail_len':   tail_len.to(device),
        'rel_name':   rel_name.to(device),
        'head_type':  head_type.to(device),
        'tail_type':  tail_type.to(device),
        'rel_domain': rel_domain.to(device),
        'rel_range':  rel_range.to(device),
    }


# ==============================================================================
# Precompute helpers – tokenize once, index every epoch
# ==============================================================================

def precompute_entity_tensors(entities, entity2desc, entity2types, type2idx, word2idx,
                               max_desc_len, num_types, max_name_len=10):
    """Tokenize every entity description, name, and type vector once into CPU tensors.

    max_name_len: max tokens for entity name (last URI segment), used as U_e
                  attention context per paper Section 5.1.2.
    """
    n = len(entities)
    desc    = torch.zeros(n, max_desc_len, dtype=torch.long)
    lengths = torch.ones(n, dtype=torch.long)
    types   = torch.zeros(n, num_types, dtype=torch.float)
    names   = torch.zeros(n, max_name_len, dtype=torch.long)  # entity name tokens U_e
    for i, e in enumerate(entities):
        words = tokenize_for_w2v(entity2desc.get(e, 'unknown'))[:max_desc_len]
        for j, w in enumerate(words):
            desc[i, j] = word2idx.get(w, 1)  # 1 = <UNK>
        lengths[i] = max(1, len(words))
        for typ in entity2types.get(e, []):
            if typ in type2idx:
                types[i, type2idx[typ]] = 1.0
        # Entity name: for DBPedia use the last URI segment ("Barack_Obama" → "barack obama");
        # for FB20k+ Freebase MIDs (/m/010016 → "010016") fall back to the first words
        # of the description, which typically begin with the entity's real name.
        seg = e.split('/')[-1].split('#')[-1].replace('_', ' ')
        non_alpha = sum(1 for c in seg if not c.isalpha() and c != ' ')
        if seg and non_alpha > len(seg) * 0.4:
            raw_desc = entity2desc.get(e, '')
            seg = ' '.join(raw_desc.split()[:4]) if raw_desc else seg
        ent_name_str = seg
        name_words = tokenize_for_w2v(ent_name_str)[:max_name_len]
        for j, w in enumerate(name_words):
            names[i, j] = word2idx.get(w, 1)
    return desc, lengths, types, names


def precompute_relation_tensors(relations, relation2name, rel2domain, rel2range,
                                 type2idx, word2idx, num_types, max_type_len=5):
    """Tokenize every relation name, constraint vector, and type-word tokens once.

    Returns (rel_name_t, rel_domain_t, rel_range_t,
             rel_domain_words_t, rel_range_words_t) on CPU.
    rel_domain_words_t / rel_range_words_t encode each type constraint as word
    indices so FactFeatureExtractor can embed them via the shared word embedding
    (paper Section 5.1.1: shared vocabulary T_{r,*}).
    """
    n = len(relations)
    rel_name_t         = torch.zeros(n, 10,           dtype=torch.long)
    rel_domain_t       = torch.zeros(n, num_types,    dtype=torch.float)
    rel_range_t        = torch.zeros(n, num_types,    dtype=torch.float)
    rel_domain_words_t = torch.zeros(n, max_type_len, dtype=torch.long)
    rel_range_words_t  = torch.zeros(n, max_type_len, dtype=torch.long)
    for i, r in enumerate(relations):
        name_words = tokenize_for_w2v(relation2name.get(r, r.split('/')[-1].replace('_', ' ')))[:10]
        for j, w in enumerate(name_words):
            rel_name_t[i, j] = word2idx.get(w, 1)
        # Multi-type support: iterate all domain/range types (paper Section 3)
        for typ in rel2domain.get(r, []):
            if typ in type2idx:
                rel_domain_t[i, type2idx[typ]] = 1.0
        for typ in rel2range.get(r, []):
            if typ in type2idx:
                rel_range_t[i, type2idx[typ]] = 1.0
        # Type word tokens (T_{r,d}): readable name of primary domain constraint
        domain_types = rel2domain.get(r, [])
        if domain_types:
            dom_name  = domain_types[0].split('/')[-1].split('#')[-1].replace('_', ' ')
            dom_words = tokenize_for_w2v(dom_name)[:max_type_len]
            for j, w in enumerate(dom_words):
                rel_domain_words_t[i, j] = word2idx.get(w, 1)
        range_types = rel2range.get(r, [])
        if range_types:
            rng_name  = range_types[0].split('/')[-1].split('#')[-1].replace('_', ' ')
            rng_words = tokenize_for_w2v(rng_name)[:max_type_len]
            for j, w in enumerate(rng_words):
                rel_range_words_t[i, j] = word2idx.get(w, 1)
    return (rel_name_t, rel_domain_t, rel_range_t,
            rel_domain_words_t, rel_range_words_t)


def build_batch_from_precomputed(h_ids, r_ids, t_ids,
                                  ent_desc, ent_len, ent_type, ent_names,
                                  rel_name_t, rel_domain_t, rel_range_t,
                                  rel_domain_words_t, rel_range_words_t,
                                  device):
    """Assemble a batch dict from GPU-resident lookup tables via index selection.

    Lookup tables (ent_desc, ent_type, etc.) live on GPU after startup.
    Index tensors (h_ids, r_ids, t_ids) may arrive on CPU (e.g. from BFS
    loop); we move them to device here so every indexed result is already
    on GPU — no post-index .to(device) copy needed.

    New keys vs. old version:
      head_name / tail_name      : entity name word-index tensors (U_e, paper Sec 5.1.2)
      rel_domain_words / rel_range_words : type-constraint word tokens (T_{r,*})
    """
    # Move indices to the same device as the lookup tables
    h_ids = h_ids.to(device)
    r_ids = r_ids.to(device)
    t_ids = t_ids.to(device)
    return {
        'head_desc':        ent_desc[h_ids],
        'tail_desc':        ent_desc[t_ids],
        'head_len':         ent_len[h_ids],
        'tail_len':         ent_len[t_ids],
        'head_name':        ent_names[h_ids],
        'tail_name':        ent_names[t_ids],
        'rel_name':         rel_name_t[r_ids],
        'head_type':        ent_type[h_ids],
        'tail_type':        ent_type[t_ids],
        'rel_domain':       rel_domain_t[r_ids],
        'rel_range':        rel_range_t[r_ids],
        'rel_domain_words': rel_domain_words_t[r_ids],
        'rel_range_words':  rel_range_words_t[r_ids],
    }


def generate_neg_indices(h_ids, r_ids, t_ids, positive_set, in_kg_ents,
                         rel_tail_type_ents=None, rel_head_type_ents=None):
    """
    Generate one filtered negative triple per positive, corrupting ONLY with
    in-KG entities (those present in the training graph).

    Sampling negatives from ALL entities (including OOK/unseen) creates a
    structural shortcut: positives always have a rich K-hop subgraph (they are
    training facts), while OOK-entity negatives have an EMPTY subgraph.  The
    scoring MLP trivially learns 'rich neighbourhood = positive' in ~10 epochs,
    collapsing loss to ln(2) before ever touching text features.

    Restricting to in-KG entities gives every negative the same structural
    richness as its paired positive, forcing the model to discriminate on
    textual and type content instead.

    If rel_tail_type_ents / rel_head_type_ents are provided (dicts mapping
    relation id → list of in-KG entity ids that match the relation's
    range / domain type constraints), negatives are sampled from those
    type-constrained buckets first.  This produces *hard negatives*:
    semantically-plausible wrong answers that the model can only reject by
    reading entity descriptions, not by exploiting type mismatch.  Fallback
    to uniform in_kg_ents when the relation has no type bucket or when
    200 type-bucket samples all collide with positive_set.
    """
    n = len(h_ids)
    neg_h = h_ids.clone()
    neg_t = t_ids.clone()
    coin  = torch.randint(0, 2, (n,), dtype=torch.bool)
    num_kg = len(in_kg_ents)
    for i in range(n):
        h, r, t = h_ids[i].item(), r_ids[i].item(), t_ids[i].item()
        if coin[i]:           # corrupt head
            bucket = rel_head_type_ents.get(r) if rel_head_type_ents else None
            nb = len(bucket) if bucket else 0
            found = False
            if bucket:
                for _ in range(200):
                    c = bucket[random.randint(0, nb - 1)]
                    if (c, r, t) not in positive_set:
                        neg_h[i] = c
                        found = True
                        break
            if not found:
                for _ in range(200):
                    c = in_kg_ents[random.randint(0, num_kg - 1)]
                    if (c, r, t) not in positive_set:
                        neg_h[i] = c
                        break
        else:                 # corrupt tail
            bucket = rel_tail_type_ents.get(r) if rel_tail_type_ents else None
            nb = len(bucket) if bucket else 0
            found = False
            if bucket:
                for _ in range(200):
                    c = bucket[random.randint(0, nb - 1)]
                    if (h, r, c) not in positive_set:
                        neg_t[i] = c
                        found = True
                        break
            if not found:
                for _ in range(200):
                    c = in_kg_ents[random.randint(0, num_kg - 1)]
                    if (h, r, c) not in positive_set:
                        neg_t[i] = c
                        break
    return neg_h, neg_t


# ==============================================================================
# Entity mean-field context  (shared by training loop and evaluate_model)
# ==============================================================================

def compute_entity_layer_means(model, th_dev, tr_dev, tt_dev,
                               ent_desc, ent_len, ent_type, ent_names,
                               rel_name_t, rel_domain_t, rel_range_t,
                               rel_domain_words_t, rel_range_words_t,
                               device, num_layers, n_ent, CHUNK=4096):
    """
    Precompute per-entity mean CNN features at each aggregation layer k=0..K.

    Returns list of (K+1) float32 tensors, each [n_ent, feat_dim].
    Entities absent from training (OOK) get a zero vector.

    Must be called while model is in eval mode.  Caller owns mode switch.
    """
    n_train = th_dev.shape[0]
    # Tables are already GPU-resident (moved to device once at startup in _main).
    with torch.no_grad():
        chunks = []
        for s in range(0, n_train, CHUNK):
            e = min(s + CHUNK, n_train)
            ct = build_batch_from_precomputed(
                th_dev[s:e], tr_dev[s:e], tt_dev[s:e],
                ent_desc, ent_len, ent_type, ent_names,
                rel_name_t, rel_domain_t, rel_range_t,
                rel_domain_words_t, rel_range_words_t, device)
            chunks.append(model.extract_fact_features(ct).float())
        z = torch.cat(chunks)             # [n_train, feat_dim]
        feat_dim = z.shape[1]

        cnt = torch.zeros(n_ent, device=device)
        cnt.scatter_add_(0, th_dev, torch.ones(n_train, device=device))
        cnt.scatter_add_(0, tt_dev, torch.ones(n_train, device=device))
        inv_cnt = (1.0 / cnt.clamp(min=1)).unsqueeze(1)

        def _emean(zz):
            m = torch.zeros(n_ent, feat_dim, device=device)
            m.scatter_add_(0, th_dev.unsqueeze(1).expand(-1, feat_dim), zz)
            m.scatter_add_(0, tt_dev.unsqueeze(1).expand(-1, feat_dim), zz)
            return m * inv_cnt

        means = [_emean(z)]
        for _ in range(num_layers):
            em = means[-1]
            z = z + torch.tanh(em[th_dev].float() + em[tt_dev].float())  # Eq 9-10
            means.append(_emean(z))

    return means   # length = num_layers + 1


# ==============================================================================
# Per-batch subgraph aggregation helpers (kept for reference / GloVe pipeline)
# ==============================================================================

def sample_subgraph_for_triple(qh, qt, entity_to_facts,
                                pos_h_list, pos_t_list, K=3, max_facts=32,
                                query_fact_id=-1,
                                drop_head=False, drop_tail=False):
    """
    BFS K-hop neighbourhood of query entities (qh, qt) in the training LINE GRAPH
    (nodes = training facts, edges = shared entity).  Returns:
      fact_ids    : list of training fact indices collected during BFS
      virtual_idx : index of the query triple's virtual node = len(fact_ids)
      edge_src / edge_dst : undirected line-graph edges including edges to virtual node

    query_fact_id: if >= 0, exclude this fact from neighbours to prevent
                  self-referential leakage (the query triple should not appear
                  as its own context; otherwise the model memorises topology
                  instead of learning from text descriptions).

    drop_head / drop_tail: if True, exclude that entity from the BFS frontier,
                  simulating an OOK entity with no training neighborhood.  The
                  virtual node still connects to it (it still appears in the
                  query triple); it just contributes no neighbour facts.
    """
    seed_ents = set()
    if not drop_head:
        seed_ents.add(qh)
    if not drop_tail:
        seed_ents.add(qt)
    visited  = {}           # fact_id -> local_index (dict preserves insertion order)
    frontier = seed_ents.copy()
    seen_ents = set(frontier)

    for _ in range(K):
        if len(visited) >= max_facts:
            break
        next_ents = set()
        for e in frontier:
            for fi in entity_to_facts.get(e, []):
                if fi == query_fact_id:      # ← skip self-referential fact
                    continue
                if fi not in visited:
                    visited[fi] = len(visited)
                    h_i, t_i = pos_h_list[fi], pos_t_list[fi]
                    for ne in (h_i, t_i):
                        if ne not in seen_ents:
                            next_ents.add(ne)
                            seen_ents.add(ne)
                    if len(visited) >= max_facts:
                        break
            if len(visited) >= max_facts:
                break
        frontier = next_ents
        if not frontier:
            break

    fact_ids    = list(visited.keys())
    virtual_idx = len(fact_ids)   # virtual query node appended last

    # entity -> local fact indices (for edge construction)
    e2lf = {}
    for li, fi in enumerate(fact_ids):
        for e in (pos_h_list[fi], pos_t_list[fi]):
            e2lf.setdefault(e, []).append(li)
    # virtual node shares both query entities
    e2lf.setdefault(qh, []).append(virtual_idx)
    e2lf.setdefault(qt, []).append(virtual_idx)

    edge_src, edge_dst = [], []
    for local_facts in e2lf.values():
        for i in range(len(local_facts)):
            for j in range(i + 1, len(local_facts)):
                a, b = local_facts[i], local_facts[j]
                edge_src.extend([a, b])
                edge_dst.extend([b, a])

    return fact_ids, virtual_idx, edge_src, edge_dst


def build_training_batch(bh, br, bt, neg_h, neg_t,
                         entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
                         ent_desc, ent_len, ent_type, ent_names,
                         rel_name_t, rel_domain_t, rel_range_t,
                         rel_domain_words_t, rel_range_words_t,
                         device, K=3, max_neighbor_facts=32,
                         triple_to_fact_id=None,
                         ook_dropout_p=0.0):
    """
    Build a batched DISJOINT-UNION of subgraphs for all pos + neg pairs.

    Each positive (h,r,t) and negative (h',r,t') becomes a 'virtual query node'
    at the end of its own K-hop training-fact subgraph, with CURRENT parameters.

    triple_to_fact_id: dict mapping (h,r,t) int-tuple -> training fact index.
                       When provided, the positive's own fact is excluded from its
                       neighbourhood (prevents self-referential leakage).

    Returns:
      feat_tensors  – input dict for extract_fact_features for ALL nodes
      edge_index    – combined line-graph edges (GPU)
      pos_q_idx     – GPU indices of each positive's virtual node
      neg_q_idx     – GPU indices of each negative's virtual node
      pos_raw_idx   – same as pos_q_idx (virtual nodes ARE the raw query features)
      neg_raw_idx   – same as neg_q_idx
    """
    B = len(bh)
    all_h, all_r, all_t  = [], [], []
    all_esrc, all_edst   = [], []
    pos_q, neg_q         = [], []
    offset = 0

    for i in range(B):
        ph, pr, pt = bh[i].item(), br[i].item(), bt[i].item()
        nh,     nt = neg_h[i].item(), neg_t[i].item()

        for (qh, qr, qt), q_list in (((ph, pr, pt), pos_q),
                                      ((nh, pr, nt), neg_q)):
            # Exclude the positive's own training fact from its neighbourhood
            # (negatives are never in training, so their query_fact_id = -1)
            if triple_to_fact_id is not None and q_list is pos_q:
                qfid = triple_to_fact_id.get((qh, qr, qt), -1)
            else:
                qfid = -1

            # OOK dropout: randomly blank one side's BFS frontier so the model
            # learns to score using text features alone (mirrors eval for OOK entities).
            drop_h = drop_t = False
            if ook_dropout_p > 0.0 and random.random() < ook_dropout_p:
                if random.random() < 0.5:
                    drop_h = True
                else:
                    drop_t = True

            fids, virt, esrc, edst = sample_subgraph_for_triple(
                qh, qt, entity_to_facts, pos_h_list, pos_t_list,
                K, max_neighbor_facts, query_fact_id=qfid,
                drop_head=drop_h, drop_tail=drop_t)

            for fi in fids:
                all_h.append(pos_h_list[fi])
                all_r.append(pos_r_list[fi])
                all_t.append(pos_t_list[fi])
            all_h.append(qh); all_r.append(qr); all_t.append(qt)

            all_esrc.extend(s + offset for s in esrc)
            all_edst.extend(d + offset for d in edst)
            q_list.append(offset + virt)
            offset += len(fids) + 1

    h_t = torch.tensor(all_h, dtype=torch.long)
    r_t = torch.tensor(all_r, dtype=torch.long)
    t_t = torch.tensor(all_t, dtype=torch.long)
    feat_tensors = build_batch_from_precomputed(
        h_t, r_t, t_t,
        ent_desc, ent_len, ent_type, ent_names,
        rel_name_t, rel_domain_t, rel_range_t,
        rel_domain_words_t, rel_range_words_t, device)

    if all_esrc:
        edge_index = torch.tensor([all_esrc, all_edst], dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    pq = torch.tensor(pos_q, dtype=torch.long, device=device)
    nq = torch.tensor(neg_q, dtype=torch.long, device=device)
    return feat_tensors, edge_index, pq, nq


def validate_loss(model, val_triples, positive_set,
                  entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
                  ent_desc, ent_len, ent_type, ent_names,
                  rel_name_t, rel_domain_t, rel_range_t,
                  rel_domain_words_t, rel_range_words_t,
                  ent2id, rel2id, device,
                  in_kg_ents,
                  num_layers, batch_size=256, max_neighbor_facts=32,
                  margin=1.0,
                  fixed_negs=None,
                  rel_tail_type_ents=None, rel_head_type_ents=None):
    """
    Loss-based validation: compute the same hinge ranking loss as training on
    held-out val triples with one in-KG negative per positive (identical mechanics
    to the training loop).

    Returns (val_loss, mean_pos_score, mean_neg_score).

    Comparing val_loss to train_loss reveals:
      val ≈ train        → healthy generalisation
      val >> train       → overfitting
      both high / flat   → underfitting / gradient failure
    mean_pos ≈ mean_neg ≈ 0.5 → logit margin not opening up
    mean_pos >> mean_neg   → model is separating pos from neg
    """
    model.eval()
    val_int = []
    for h, r, t in val_triples:
        if h in ent2id and r in rel2id and t in ent2id:
            val_int.append((ent2id[h], rel2id[r], ent2id[t]))
    if not val_int:
        return 0.0, 0.5, 0.5

    total_loss        = 0.0
    n_batches         = 0
    all_pos_scores:   list = []
    all_neg_scores:   list = []

    with torch.no_grad():
        for start in range(0, len(val_int), batch_size):
            batch = val_int[start:start + batch_size]
            bh = torch.tensor([h for h, r, t in batch], dtype=torch.long)
            br = torch.tensor([r for h, r, t in batch], dtype=torch.long)
            bt = torch.tensor([t for h, r, t in batch], dtype=torch.long)

            # Use pre-computed fixed negatives if provided for a stable,
            # noise-free val signal; otherwise sample fresh (legacy).
            if fixed_negs is not None:
                end = min(start + batch_size, len(val_int))
                neg_h = fixed_negs[0][start:end]
                neg_t = fixed_negs[1][start:end]
            else:
                neg_h, neg_t = generate_neg_indices(
                    bh, br, bt, positive_set, in_kg_ents,
                    rel_tail_type_ents=rel_tail_type_ents,
                    rel_head_type_ents=rel_head_type_ents)

            feat_tensors, edge_index, pq, nq = build_training_batch(
                bh, br, bt, neg_h, neg_t,
                entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
                ent_desc, ent_len, ent_type, ent_names,
                rel_name_t, rel_domain_t, rel_range_t,
                rel_domain_words_t, rel_range_words_t,
                device, K=num_layers, max_neighbor_facts=max_neighbor_facts,
                triple_to_fact_id=None)

            all_z = model.extract_fact_features(feat_tensors).float()
            all_z = model.aggregator(all_z, edge_index)

            ps_logit = model(all_z[pq], return_logits=True).float()
            ns_logit = model(all_z[nq], return_logits=True).float()

            loss = F.relu(margin - ps_logit + ns_logit).mean()

            total_loss += loss.item()
            n_batches  += 1
            all_pos_scores.append(torch.sigmoid(ps_logit).cpu())
            all_neg_scores.append(torch.sigmoid(ns_logit).cpu())

    val_loss = total_loss / max(n_batches, 1)
    pos_mean = float(torch.cat(all_pos_scores).mean().item()) if all_pos_scores else 0.5
    neg_mean = float(torch.cat(all_neg_scores).mean().item()) if all_neg_scores else 0.5
    model.train()
    return val_loss, pos_mean, neg_mean


def evaluate_model(model, eval_triples, metadata, word2idx, type2idx, max_desc_len,
                   device, ent2id, rel2id, id2ent, id2rel,
                   fact_edge_index, train_triples,
                   ent_desc, ent_len, ent_type, ent_names,
                   rel_name_t, rel_domain_t, rel_range_t,
                   rel_domain_words_t, rel_range_words_t,
                   all_triples_for_filter,
                   train_ent_set, train_rel_set,
                   is_test=False,
                   report_filename="ikge_evaluation_report.pdf"):
    """
    Full-ranking evaluation for final test scoring (paper-exact 4-group eval).
    Only called with is_test=True at the end of training.

    train_ent_set / train_rel_set : sets of *string* entity/relation IDs seen in
                                    train_triples, used to classify test triples.
    all_triples_for_filter        : full train+val+test list for filter masks.
    """
    entity2desc, relation2name, entity2types, rel2domain, rel2range = metadata
    scorer = UnifiedKGScorer(device=str(device))

    # Integer-ID conversion, drop unknowns
    eval_int = []
    for triple in eval_triples:
        h, r, t = triple[0], triple[1], triple[2]
        if h in ent2id and r in rel2id and t in ent2id:
            eval_int.append((ent2id[h], rel2id[r], ent2id[t]))

    if not eval_int:
        print("Warning: no valid eval triples after ID mapping.")
        return 0.0

    # Build filter dicts from ALL known triples
    from collections import defaultdict as _dd
    filter_tails = _dd(list)   # (h,r)   -> [t ...]
    filter_heads = _dd(list)   # (r,t)   -> [h ...]
    filter_rels  = _dd(list)   # (h,t)   -> [r ...]
    for triple in all_triples_for_filter:
        h_s, r_s, t_s = triple[0], triple[1], triple[2]
        if h_s in ent2id and r_s in rel2id and t_s in ent2id:
            h_i, r_i, t_i = ent2id[h_s], rel2id[r_s], ent2id[t_s]
            filter_tails[(h_i, r_i)].append(t_i)
            filter_heads[(r_i, t_i)].append(h_i)
            filter_rels [(h_i, t_i)].append(r_i)
    filter_tails = dict(filter_tails)
    filter_heads = dict(filter_heads)
    filter_rels  = dict(filter_rels)

    # ---------------------------------------------------------------
    # Lookup tables are already GPU-resident (moved to device once in _main).
    # Short aliases for readability inside this function.
    # ---------------------------------------------------------------
    model.eval()
    _ed  = ent_desc
    _el  = ent_len
    _et  = ent_type
    _en  = ent_names
    _rn  = rel_name_t
    _rd  = rel_domain_t
    _rr  = rel_range_t
    _rdw = rel_domain_words_t
    _rrw = rel_range_words_t

    # -----------------------------------------------------------------------
    # Paper Section 5.2.3 – Inference
    #
    # "the fact feature information of other facts in a training KG is
    #  already extracted at the training phase."
    #
    # Pre-compute initial CNN features φ(f_i) for every training fact.
    # At inference, each target fact (h, r, t_cand) is scored by:
    #   1. Extracting its own features: z = φ(h, r, t_cand)
    #   2. Finding its K-hop BFS neighbourhood in the training line graph
    #   3. Assembling a small subgraph with pre-cached training feat vectors
    #   4. Running K layers of attentive aggregation (Equations 6–11)
    #   5. Scoring the virtual query node with the MLP
    #
    # OOK entities (unseen at training) have no training neighbours;
    # their scoring relies entirely on CNN text features (zero neighbourhood
    # contribution), which is exactly the paper's intended behaviour.
    # -----------------------------------------------------------------------
    print("  Pre-computing training fact CNN features (paper Sec. 5.2.3)...")
    _agg_mod = model._orig_mod.aggregator if hasattr(model, '_orig_mod') else model.aggregator
    CHUNK = 4096

    _pos_h_l_ev = [ent2id[h] for h, r, t in train_triples
                   if h in ent2id and r in rel2id and t in ent2id]
    _pos_r_l_ev = [rel2id[r] for h, r, t in train_triples
                   if h in ent2id and r in rel2id and t in ent2id]
    _pos_t_l_ev = [ent2id[t] for h, r, t in train_triples
                   if h in ent2id and r in rel2id and t in ent2id]
    n_train_ev  = len(_pos_h_l_ev)

    _th_ev = torch.tensor(_pos_h_l_ev, dtype=torch.long, device=device)
    _tr_ev = torch.tensor(_pos_r_l_ev, dtype=torch.long, device=device)
    _tt_ev = torch.tensor(_pos_t_l_ev, dtype=torch.long, device=device)

    # entity_id -> list of training fact indices (BFS neighbourhood lookup)
    _e2f_ev: dict = {}
    for _i in range(n_train_ev):
        for _e in (_pos_h_l_ev[_i], _pos_t_l_ev[_i]):
            _e2f_ev.setdefault(_e, []).append(_i)

    with torch.no_grad():
        raw_chunks = []
        for s in range(0, n_train_ev, CHUNK):
            e = min(s + CHUNK, n_train_ev)
            ct = build_batch_from_precomputed(
                _th_ev[s:e], _tr_ev[s:e], _tt_ev[s:e],
                _ed, _el, _et, _en, _rn, _rd, _rr, _rdw, _rrw, device)
            raw_chunks.append(model.extract_fact_features(ct).float().cpu())
        # Keep on CPU to preserve VRAM during the large eval ranking loop
        z_train_init = torch.cat(raw_chunks)   # [n_train, d]

    print(f"  z_train_init cached: {z_train_init.shape}  "
          f"| BFS index covers {len(_e2f_ev)} entities.")

    # -----------------------------------------------------------------------
    # GPU-vectorised full-ranking scorer
    # -----------------------------------------------------------------------
    # Key idea: replace the per-candidate Python BFS loop with dense padded
    # tensors.  For each test triple we score ALL n_ents candidates at once:
    #   1.  CNN chunked over n_ents (no Python loop over candidates)
    #   2.  Neighbour features fetched from a pre-built [n_ents, HALF] table
    #   3.  K-layer attention computed as batched einsum – no graph objects
    # -----------------------------------------------------------------------
    MAX_NBRS_EVAL = 32                   # neighbour cap per entity
    HALF        = MAX_NBRS_EVAL // 2     # 16 neighbours stored per entity
    n_ents      = len(ent2id)
    n_rels      = len(rel2id)
    n_train_fac = z_train_init.shape[0]  # also used as PAD index
    CNN_CHUNK   = 1024                   # max total items per CNN call (Q × sub_cands ≤ this)
    CAND_CHUNK  = 512                    # candidate block for aggregation+score
    Q_BATCH     = 4                      # test triples processed in parallel (reduced for OOM)
    NEG_INF     = float('-inf')

    # Build padded entity→(train-fact neighbours) table: [n_ents, HALF]
    print("  Building padded entity→fact neighbour table...")
    ent_nbrs_cpu = torch.full((n_ents, HALF), n_train_fac, dtype=torch.long)
    for e, facts in _e2f_ev.items():
        k = min(HALF, len(facts))
        ent_nbrs_cpu[e, :k] = torch.tensor(facts[:k], dtype=torch.long)

    # Append a zero row so that index n_train_fac always returns a zero vector
    z_train_pad = torch.cat([
        z_train_init.to(device, dtype=torch.float32),
        torch.zeros(1, z_train_init.shape[1], device=device, dtype=torch.float32),
    ], dim=0)   # [n_train_fac + 1, d]

    # Pre-fetch ALL entity neighbour features to GPU once
    # [n_ents, HALF]: 29902 × 4 × 256 × 4B ≈ 122 MB – fits in VRAM
    ent_nbrs_gpu  = ent_nbrs_cpu.to(device)           # [n_ents, HALF]
    ent_nbr_feats = z_train_pad[ent_nbrs_gpu]          # [n_ents, HALF, d]
    ent_nbr_mask  = (ent_nbrs_cpu != n_train_fac).to(device)  # [n_ents, HALF]

    d_emb = z_train_pad.shape[1]
    agg   = model.aggregator

    # ------------------------------------------------------------------
    # Target filtering tables (paper Section 6.2.1)
    # "evaluates only the candidate entities whose relation-entity
    #  combinations exist in the training KG."
    #
    # CORRECT granularity: keyed by the specific (context_entity, relation)
    # training pair, NOT just by relation.
    #   pair_tail_cands[(h, r)] = tails seen for this exact (h, r) in training
    #   pair_head_cands[(r, t)] = heads seen for this exact (r, t) in training
    #
    # OOK-relation fallback (when (h, r_ook) has no training pair):
    #   ent_tail_cands[h] = all tails seen with h across ANY relation in training
    #   ent_head_cands[t] = all heads seen with t across ANY relation in training
    # This gives ~1–30 candidates per triple instead of 30k, keeping eval fast.
    # ------------------------------------------------------------------
    print("  Building target-filtering candidate tables...")
    # 4-tier candidate resolution (paper Section 6.2.1):
    #   Tier 1 pair_*_cands[(h,r)/(r,t)]: exact (context-entity, relation) pair in training
    #   Tier 2 rel_*_cands[r]:            all entities seen with relation r in training
    #                                      ← KEY tier: handles OOK entity + known relation
    #                                         (O-O-X head pred, X-O-O tail pred)
    #   Tier 3 ent_*_cands[e]:            all entities seen with context-entity across any rel
    #                                      ← handles OOK relation + known entity (O-X-O)
    #   Tier 4:                            full n_ents ranking (last resort)
    pair_tail_cands: dict = {}   # (h_id, r_id) -> set[int]
    pair_head_cands: dict = {}   # (r_id, t_id) -> set[int]
    rel_tail_cands:  dict = {}   # r_id -> set[int]   (OOK-entity, known-relation fallback)
    rel_head_cands:  dict = {}   # r_id -> set[int]   (OOK-entity, known-relation fallback)
    ent_tail_cands:  dict = {}   # h_id -> set[int]   (OOK-relation, known-entity fallback)
    ent_head_cands:  dict = {}   # t_id -> set[int]   (OOK-relation, known-entity fallback)
    for h_i, r_i, t_i in zip(_pos_h_l_ev, _pos_r_l_ev, _pos_t_l_ev):
        pair_tail_cands.setdefault((h_i, r_i), set()).add(t_i)
        pair_head_cands.setdefault((r_i, t_i), set()).add(h_i)
        rel_tail_cands.setdefault(r_i, set()).add(t_i)
        rel_head_cands.setdefault(r_i, set()).add(h_i)
        ent_tail_cands.setdefault(h_i, set()).add(t_i)
        ent_head_cands.setdefault(t_i, set()).add(h_i)
    # Convert to sorted lists for deterministic ordering
    pair_tail_cands = {k: sorted(s) for k, s in pair_tail_cands.items()}
    pair_head_cands = {k: sorted(s) for k, s in pair_head_cands.items()}
    rel_tail_cands  = {k: sorted(s) for k, s in rel_tail_cands.items()}
    rel_head_cands  = {k: sorted(s) for k, s in rel_head_cands.items()}
    ent_tail_cands  = {k: sorted(s) for k, s in ent_tail_cands.items()}
    ent_head_cands  = {k: sorted(s) for k, s in ent_head_cands.items()}
    ptl = [len(v) for v in pair_tail_cands.values()]
    phl = [len(v) for v in pair_head_cands.values()]
    rtl = [len(v) for v in rel_tail_cands.values()]
    rhl = [len(v) for v in rel_head_cands.values()]
    etl = [len(v) for v in ent_tail_cands.values()]
    ehl = [len(v) for v in ent_head_cands.values()]
    print(f"  T1 pair_tail_cands: {len(pair_tail_cands)} pairs,    avg {sum(ptl)/max(1,len(ptl)):.1f} cands/pair")
    print(f"  T1 pair_head_cands: {len(pair_head_cands)} pairs,    avg {sum(phl)/max(1,len(phl)):.1f} cands/pair")
    print(f"  T2 rel_tail_cands:  {len(rel_tail_cands)} relations, avg {sum(rtl)/max(1,len(rtl)):.1f} cands/rel")
    print(f"  T2 rel_head_cands:  {len(rel_head_cands)} relations, avg {sum(rhl)/max(1,len(rhl)):.1f} cands/rel")
    print(f"  T3 ent_tail_cands:  {len(ent_tail_cands)} entities,  avg {sum(etl)/max(1,len(etl)):.1f} cands/entity")
    print(f"  T3 ent_head_cands:  {len(ent_head_cands)} entities,  avg {sum(ehl)/max(1,len(ehl)):.1f} cands/entity")

    # Pre-project the static ent_nbr_feats through each attention layer W_k once.
    # Correct formula (Eq 8, after attentive_aggregator fix): ATSCORE_v = f_v^T W_a f_u
    # W_a is applied to the SOURCE / QUERY (z_q), not to neighbors.
    # We cannot pre-project z_q (it changes each aggregation step), so the
    # neighbor features are kept raw and z_q is projected on-the-fly below.
    # ent_nbr_proj_layers is NOT precomputed here — it was dead code after the fix.
    # (The old code incorrectly pre-projected neighbors; that is removed.)

    # ------------------------------------------------------------------
    # _score_gpu: score Q queries × N_C candidates (OOK full-entity ranking).
    # Outer loop over CAND_CHUNK=512 blocks for aggregation granularity.
    # Inner loop over CNN sub-chunks: CNN_CHUNK//Q candidates per call, so
    # CNN sees ≤ CNN_CHUNK total items → peak [1024, 300, 50] conv input (~60 MB).
    # Aggregation runs on the full CAND_CHUNK block: [Q=16, 512, 300] ≈ 9 MB.
    # ------------------------------------------------------------------
    def _score_gpu(fixed_ids: list, r_ids_list: list,
                   cand_ids: torch.Tensor,
                   corrupt_dim: str) -> torch.Tensor:
        Q    = len(fixed_ids)
        N_C  = cand_ids.shape[0]
        fix_t = torch.tensor(fixed_ids,  dtype=torch.long, device=device)
        r_t   = torch.tensor(r_ids_list, dtype=torch.long, device=device)

        # Pre-fetch the fixed entity's neighbours once — same for every candidate chunk
        fix_nf = ent_nbr_feats[fix_t]   # [Q, HALF, d]
        fix_nm = ent_nbr_mask[fix_t]    # [Q, HALF]

        all_scores = torch.empty(Q, N_C, dtype=torch.float32)  # accumulated on CPU

        with torch.no_grad():
            for cs in range(0, N_C, CAND_CHUNK):
                ce      = min(cs + CAND_CHUNK, N_C)
                csz     = ce - cs
                c_chunk = cand_ids[cs:ce]

                # Build z [Q, csz, d] via CNN sub-chunks so each call is ≤ CNN_CHUNK items
                # (CNN_CHUNK items total = Q × sub_cands  →  sub_cands = CNN_CHUNK // Q)
                sub_cands = max(1, CNN_CHUNK // Q)
                z = torch.empty(Q, csz, d_emb, device=device, dtype=torch.float32)
                for ci in range(0, csz, sub_cands):
                    cj     = min(ci + sub_cands, csz)
                    ssz    = cj - ci
                    sub_c  = c_chunk[ci:cj]
                    fx_exp = fix_t.unsqueeze(1).expand(Q, ssz).reshape(-1)
                    rx_exp = r_t.unsqueeze(1).expand(Q, ssz).reshape(-1)
                    cx_exp = sub_c.unsqueeze(0).expand(Q, ssz).reshape(-1)
                    if corrupt_dim == 'tail':
                        hx, tx = fx_exp, cx_exp
                    else:
                        hx, tx = cx_exp, fx_exp
                    sub_feat = build_batch_from_precomputed(
                        hx, rx_exp, tx, _ed, _el, _et, _en, _rn, _rd, _rr, _rdw, _rrw, device)
                    z[:, ci:cj, :] = model.extract_fact_features(sub_feat).float().view(Q, ssz, d_emb)

                # Candidate neighbours for this chunk
                cand_nf = ent_nbr_feats[c_chunk]   # [csz, HALF, d]
                cand_nm = ent_nbr_mask[c_chunk]     # [csz, HALF]

                # K-layer aggregation (same einsum pattern as training)
                for li, layer in enumerate(agg.attention_layers):
                    W      = layer.weight
                    z_proj = torch.matmul(z, W.T)   # [Q, csz, d]  W_a f_u
                    att_fix = torch.einsum('qkd,qnd->qnk', fix_nf, z_proj)
                    att_fix.masked_fill_(~fix_nm.unsqueeze(1).expand(Q, csz, HALF), NEG_INF)
                    att_can = torch.einsum('nkd,qnd->qnk', cand_nf, z_proj)
                    att_can.masked_fill_(~cand_nm.unsqueeze(0).expand(Q, csz, HALF), NEG_INF)
                    att_all = torch.cat([att_fix, att_can], dim=2)
                    att_w   = torch.nan_to_num(torch.softmax(att_all, dim=2), 0.0)
                    agg_fix = torch.einsum('qnk,qkd->qnd', att_w[:, :, :HALF], fix_nf)
                    agg_can = torch.einsum('qnk,nkd->qnd', att_w[:, :, HALF:], cand_nf)
                    z       = z + torch.tanh(agg_fix + agg_can)

                # Score and write directly to CPU output — no large GPU accumulation
                all_scores[:, cs:ce] = model(z.reshape(Q * csz, d_emb)).float().view(Q, csz).cpu()

        return all_scores

    # ------------------------------------------------------------------
    # _score_flat_gpu: score a flat list of N individual (head,r,tail) facts.
    # Processes all stages (CNN + K-layer aggregation + score) in FLAT_CHUNK
    # blocks so peak VRAM is O(FLAT_CHUNK) not O(N).
    # With N=4M after the T2 fix, old code tried [4M, 16, 300] ≈ 80 GB.
    # At FLAT_CHUNK=2048: [2048, 16, 300] × 2 ≈ 75 MB per chunk.
    # ------------------------------------------------------------------
    FLAT_CHUNK = 2048
    def _score_flat_gpu(h_ids: list, r_ids: list, t_ids: list) -> torch.Tensor:
        N       = len(h_ids)
        h_t_cpu = torch.tensor(h_ids, dtype=torch.long)
        r_t_cpu = torch.tensor(r_ids, dtype=torch.long)
        t_t_cpu = torch.tensor(t_ids, dtype=torch.long)
        scores  = torch.empty(N, dtype=torch.float32)   # CPU output
        _t0_flat = time.time()

        with torch.no_grad():
            for cs in range(0, N, FLAT_CHUNK):
                ce   = min(cs + FLAT_CHUNK, N)
                bh   = h_t_cpu[cs:ce].to(device)
                br   = r_t_cpu[cs:ce].to(device)
                bt   = t_t_cpu[cs:ce].to(device)
                bsz  = ce - cs

                # 1. CNN
                feat = build_batch_from_precomputed(
                    bh, br, bt, _ed, _el, _et, _en, _rn, _rd, _rr, _rdw, _rrw, device)
                z = model.extract_fact_features(feat).float()   # [bsz, d]

                # 2. Neighbour features for this chunk only
                h_nf = ent_nbr_feats[bh]   # [bsz, HALF, d]
                t_nf = ent_nbr_feats[bt]   # [bsz, HALF, d]
                h_nm = ent_nbr_mask[bh]    # [bsz, HALF]
                t_nm = ent_nbr_mask[bt]    # [bsz, HALF]

                # 3. K-layer aggregation
                for li, layer in enumerate(agg.attention_layers):
                    W      = layer.weight
                    z_proj = torch.matmul(z, W.T)                         # [bsz, d]
                    att_h  = torch.einsum('nkd,nd->nk', h_nf, z_proj)    # [bsz, HALF]
                    att_t  = torch.einsum('nkd,nd->nk', t_nf, z_proj)    # [bsz, HALF]
                    att_h.masked_fill_(~h_nm, NEG_INF)
                    att_t.masked_fill_(~t_nm, NEG_INF)
                    att_all = torch.cat([att_h, att_t], dim=1)            # [bsz, 2*HALF]
                    att_w   = torch.nan_to_num(torch.softmax(att_all, dim=1), 0.0)
                    agg_h   = torch.einsum('nk,nkd->nd', att_w[:, :HALF], h_nf)
                    agg_t   = torch.einsum('nk,nkd->nd', att_w[:, HALF:], t_nf)
                    z       = z + torch.tanh(agg_h + agg_t)

                # 4. Score → write directly to CPU
                scores[cs:ce] = model(z).float().view(-1).cpu()
                # Progress every 200 chunks
                chunk_idx = cs // FLAT_CHUNK + 1
                if chunk_idx % 200 == 0 or ce >= N:
                    elapsed = time.time() - _t0_flat
                    eta     = elapsed / ce * (N - ce) if ce < N else 0
                    print(f"      [flat] {ce}/{N} facts  {elapsed:.0f}s elapsed  ETA {eta:.0f}s",
                          flush=True)

        return scores

    # ------------------------------------------------------------------
    # _score_rels_gpu: score Q queries against every relation.
    # No target filtering for relation prediction (paper Section 6.2.2).
    # ------------------------------------------------------------------
    # Diagnostic flag: print score-distribution stats for first G4 batch (once)
    _g4_diag_done = [False]

    def _score_rels_gpu(h_ids_l: list, t_ids_l: list) -> torch.Tensor:
        """Returns [Q, n_rels] float32 CPU tensor.

        DESIGN NOTE — why we skip entity-neighbour aggregation here:
        The model was trained with entity corruption only (relation held fixed).
        The aggregation adds a per-relation additive shift derived from fixed
        entity neighbours, which biases all relation candidates identically and
        collapses the score distribution.  Scoring directly from the raw CNN
        features (which encode the triple text including relation description)
        gives the MLP the best chance to discriminate relations.
        Retraining with relation corruption will make aggregation useful here too.
        """
        Q   = len(h_ids_l)
        h_t = torch.tensor(h_ids_l, dtype=torch.long, device=device)
        t_t = torch.tensor(t_ids_l, dtype=torch.long, device=device)
        r_t = torch.arange(n_rels,   dtype=torch.long, device=device)
        h_exp = h_t.unsqueeze(1).expand(Q, n_rels).reshape(-1)
        r_exp = r_t.unsqueeze(0).expand(Q, -1).reshape(-1)
        t_exp = t_t.unsqueeze(1).expand(Q, n_rels).reshape(-1)
        with torch.no_grad():
            feat   = build_batch_from_precomputed(
                h_exp, r_exp, t_exp, _ed, _el, _et, _en, _rn, _rd, _rr, _rdw, _rrw, device)
            z_r    = model.extract_fact_features(feat).float().view(Q, n_rels, d_emb)
            # Raw CNN features only — skip entity-neighbour aggregation.
            # (Aggregation uses entity-neighbour context which was trained with
            #  entity corruption only; applying it here collapses rel scores.)
            scores = model(z_r.reshape(Q * n_rels, d_emb)).float().view(Q, n_rels)

        # One-time diagnostic: print score distribution of first batch
        if not _g4_diag_done[0]:
            _g4_diag_done[0] = True
            with torch.no_grad():
                s0 = scores[0]  # [n_rels]
                print(f"    [G4 diag] first query score dist: "
                      f"min={s0.min().item():.4f}  max={s0.max().item():.4f}  "
                      f"mean={s0.mean().item():.4f}  std={s0.std().item():.4f}  "
                      f"n_rels={n_rels}", flush=True)

        return scores.cpu()

    # ------------------------------------------------------------------
    # _rank_gpu: compute filtered ranks for one corruption direction.
    #
    # Entity prediction: uses target-filtered candidate sets (per the paper
    #   Section 6.2.1: "only candidate entities whose relation-entity
    #   combinations exist in the training KG").  The true answer is always
    #   added to the set so that OOK targets (which may not appear in
    #   training) are still ranked.  Within the filtered set, standard
    #   "filtered" MRR masking removes other known true facts to avoid
    #   penalising correct predictions.
    #
    # Relation prediction: full ranking over all n_rels (no target filtering,
    #   paper Section 6.2.2).
    # ------------------------------------------------------------------
    def _rank_gpu(triples: list, corrupt_dim: str, filter_dict: dict) -> list:
        if not triples:
            return []

        if corrupt_dim == 'relation':
            # No target filtering — rank all relations
            print(f"    [relation] {len(triples)} triples × {n_rels} candidates (no filter)")
            ranks = []
            torch.cuda.empty_cache()
            for qi in tqdm(range(0, len(triples), Q_BATCH)):
                batch = triples[qi:qi + Q_BATCH]
                b     = len(batch)
                b_h   = [t[0] for t in batch]
                b_t   = [t[2] for t in batch]
                sc    = _score_rels_gpu(b_h, b_t)
                for i in range(b):
                    row    = sc[i].clone()
                    true_e = batch[i][1]
                    known  = filter_dict.get((b_h[i], b_t[i]), [])
                    for e in known:
                        if e != true_e:
                            row[e] = NEG_INF
                    ranks.append(int((row > row[true_e]).sum().item()) + 1)
            return ranks

        # Entity prediction — per-triple candidate sets (target filtering).
        # Use _score_flat_gpu: flatten ALL (h, r, t_cand) facts across every
        # test triple into one big batch, score them in one pass, then
        # reconstruct per-triple ranks.  This avoids both the dense Q×N_C
        # matrix and the per-group GPU-call overhead.
        ranks           = [0] * len(triples)
        all_ent_ids_gpu = torch.arange(n_ents, dtype=torch.long, device=device)
        pair_cands_t = pair_tail_cands if corrupt_dim == 'tail' else pair_head_cands
        rel_cands_t  = rel_tail_cands  if corrupt_dim == 'tail' else rel_head_cands
        ent_cands_t  = ent_tail_cands  if corrupt_dim == 'tail' else ent_head_cands

        # Build flat fact arrays for target-filtered triples; collect OOK separately
        flat_h     = []   # head entity id for each individual scored fact
        flat_r     = []
        flat_t     = []
        flat_qi    = []   # which test triple does this fact belong to
        flat_ce    = []   # candidate entity id (= head or tail depending on direction)
        ook_list   = []   # (orig_idx, trip) for truly-OOK triples → full ranking
        tier_counts = [0, 0, 0, 0]  # T1/T2/T3/T4 hit counts

        per_triple_cands  = {}   # orig_idx -> {cand_e -> flat_position}
        per_triple_true   = {}   # orig_idx -> true_e

        for idx, trip in enumerate(triples):
            h_i, r_i, t_i = trip
            true_e   = t_i if corrupt_dim == 'tail' else h_i
            pair_key = (h_i, r_i) if corrupt_dim == 'tail' else (r_i, t_i)
            ent_key  = h_i if corrupt_dim == 'tail' else t_i
            # 4-tier candidate resolution
            base = pair_cands_t.get(pair_key)          # T1: exact (entity, relation) pair
            if base is None:
                base = rel_cands_t.get(r_i)            # T2: all entities for this relation
            if base is None:
                base = ent_cands_t.get(ent_key)        # T3: all entities for context entity
            tier = (1 if pair_cands_t.get(pair_key) is not None else
                    2 if rel_cands_t.get(r_i) is not None else
                    3 if ent_cands_t.get(ent_key) is not None else 4)
            tier_counts[tier - 1] += 1
            if base is not None:
                cand_set  = set(base)
                cand_set.add(true_e)           # always include true answer
                cands     = sorted(cand_set)
                start_pos = len(flat_h)
                cand_pos  = {}
                for ci, cand_e in enumerate(cands):
                    if corrupt_dim == 'tail':
                        flat_h.append(h_i);  flat_r.append(r_i);  flat_t.append(cand_e)
                    else:
                        flat_h.append(cand_e); flat_r.append(r_i); flat_t.append(t_i)
                    flat_qi.append(idx)
                    flat_ce.append(cand_e)
                    cand_pos[cand_e] = start_pos + ci
                per_triple_cands[idx] = cand_pos
                per_triple_true[idx]  = true_e
            else:
                # Both entity and pair unseen in training → full entity ranking
                ook_list.append((idx, trip))

        print(f"    [{corrupt_dim}] {len(per_triple_cands)} filtered "
              f"({len(flat_h)} facts) | T1:{tier_counts[0]} T2:{tier_counts[1]} "
              f"T3:{tier_counts[2]} T4(full):{tier_counts[3]}")
        if tier_counts[1] > 0:
            t2_sizes = []
            for idx, trip in enumerate(triples):
                h_i, r_i, t_i = trip
                pair_key = (h_i, r_i) if corrupt_dim == 'tail' else (r_i, t_i)
                ent_key  = h_i if corrupt_dim == 'tail' else t_i
                if pair_cands_t.get(pair_key) is None and rel_cands_t.get(r_i) is not None:
                    t2_sizes.append(len(rel_cands_t[r_i]))
            if t2_sizes:
                print(f"    [T2 stats] avg {sum(t2_sizes)/len(t2_sizes):.0f} cands, "
                      f"min {min(t2_sizes)}, max {max(t2_sizes)} (vs 30k full ranking)")

        # --- Score all target-filtered facts in one flat pass ---
        if flat_h:
            t_flat0 = time.time()
            flat_scores = _score_flat_gpu(flat_h, flat_r, flat_t)  # [N_total]
            print(f"    [flat pass] {len(flat_h)} facts scored in {time.time()-t_flat0:.1f}s")
            # Reconstruct per-triple ranks
            filtered_ranks = []
            for idx, cand_pos in per_triple_cands.items():
                true_e = per_triple_true[idx]
                h_i, r_i, t_i = triples[idx]
                known  = filter_dict.get((h_i, r_i) if corrupt_dim == 'tail' else (r_i, t_i), [])
                # Build score vector for this triple's candidates
                pos_list = sorted(cand_pos.values())   # contiguous slice
                row      = flat_scores[pos_list[0] : pos_list[-1] + 1].clone()
                local    = {e: p - pos_list[0] for e, p in cand_pos.items()}
                for e in known:
                    if e != true_e and e in local:
                        row[local[e]] = NEG_INF
                true_local = local.get(true_e, -1)
                r_val = (int((row > row[true_local]).sum().item()) + 1
                         if true_local >= 0 else len(cand_pos) + 1)
                ranks[idx] = r_val
                filtered_ranks.append(r_val)
            if filtered_ranks:
                import numpy as _np
                _fa = _np.array(filtered_ranks, dtype=float)
                print(f"    [filtered MRR] {len(filtered_ranks)} triples: "
                      f"MRR={float(_np.mean(1.0/_fa)):.4f}  "
                      f"H@1={float(_np.mean(_fa<=1)):.4f}  "
                      f"H@10={float(_np.mean(_fa<=10)):.4f}  "
                      f"MeanRank={float(_np.mean(_fa)):.1f}")

        # --- Score fully-OOK triples: entity unseen in training → full 30k ranking ---
        import numpy as _np
        n_ook       = len(ook_list)
        ook_t0      = time.time()
        ook_ranks_sofar = []
        for qi in range(0, n_ook, Q_BATCH):
            sub   = ook_list[qi:qi + Q_BATCH]
            b     = len(sub)
            orig_indices = [s[0] for s in sub]
            trips        = [s[1] for s in sub]
            fixed = [tr[0] for tr in trips] if corrupt_dim == 'tail' else [tr[2] for tr in trips]
            r_lst = [tr[1] for tr in trips]
            sc    = _score_gpu(fixed, r_lst, all_ent_ids_gpu, corrupt_dim)  # [b, n_ents]
            for i in range(b):
                row    = sc[i].clone()
                h_i, r_i, t_i = trips[i]
                true_e = t_i if corrupt_dim == 'tail' else h_i
                known  = filter_dict.get((h_i, r_i) if corrupt_dim == 'tail' else (r_i, t_i), [])
                for e in known:
                    if e != true_e:
                        row[e] = NEG_INF
                r_val = int((row > row[true_e]).sum().item()) + 1
                ranks[orig_indices[i]] = r_val
                ook_ranks_sofar.append(r_val)
            # Progress every batch
            done    = qi + b
            elapsed = time.time() - ook_t0
            spt     = elapsed / done
            eta     = spt * (n_ook - done)
            partial_mrr = float(_np.mean(1.0 / _np.array(ook_ranks_sofar, dtype=float)))
            # Print every 10 batches (or first/last) to avoid spam
            batch_num = qi // Q_BATCH + 1
            n_batches = (n_ook + Q_BATCH - 1) // Q_BATCH
            if batch_num == 1 or batch_num % 10 == 0 or done >= n_ook:
                print(f"    [OOK {corrupt_dim}] {done}/{n_ook} triples  "
                      f"MRR={partial_mrr:.4f}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                      f"({spt:.2f}s/triple)", flush=True)

        if ook_ranks_sofar:
            _fa = _np.array(ook_ranks_sofar, dtype=float)
            print(f"    [OOK final] MRR={float(_np.mean(1.0/_fa)):.4f}  "
                  f"H@1={float(_np.mean(_fa<=1)):.4f}  "
                  f"H@10={float(_np.mean(_fa<=10)):.4f}  "
                  f"MeanRank={float(_np.mean(_fa)):.1f}  "
                  f"total={time.time()-ook_t0:.1f}s")
        # Combined (filtered + OOK) final group MRR
        all_group_ranks = [r for r in ranks if r > 0]
        if all_group_ranks:
            _ga = _np.array(all_group_ranks, dtype=float)
            print(f"    *** [{corrupt_dim} GROUP TOTAL] n={len(_ga)}  "
                  f"MRR={float(_np.mean(1.0/_ga)):.4f}  "
                  f"H@1={float(_np.mean(_ga<=1)):.4f}  "
                  f"H@3={float(_np.mean(_ga<=3)):.4f}  "
                  f"H@10={float(_np.mean(_ga<=10)):.4f}  "
                  f"MR={float(_np.mean(_ga)):.1f} ***")
        return ranks

    print("Running Full-Ranking Test Evaluation (GPU-vectorised)...")

    # ── Test: paper-exact 4-group evaluation ────────────────────────────────
    # Classify each test triple by (h∈train, r∈train, t∈train).
    # O=in-KG, X=out-of-KG  →  notation is (head, rel, tail)
    oot   = []   # O-O-X  tail OOK   → G1 head-pred, G3 head-pred, G4 rel-pred
    xoo   = []   # X-O-O  head OOK   → G2 tail-pred, G3 tail-pred, G4 rel-pred
    oxo   = []   # O-X-O  rel OOK    → G1 head-pred, G2 tail-pred
    oxx   = []   # O-X-X  rel+tail OOK → G1 head-pred
    xxo   = []   # X-X-O  head+rel OOK → G2 tail-pred
    xox   = []   # X-O-X  head+tail OOK → G4 rel-pred only
    # O-O-O (all in-KG) → closed-world, not evaluated here

    for h_i, r_i, t_i in eval_int:
        h_s = id2ent[h_i]; r_s = id2rel[r_i]; t_s = id2ent[t_i]
        h_in = h_s in train_ent_set
        r_in = r_s in train_rel_set
        t_in = t_s in train_ent_set
        key  = (h_in, r_in, t_in)
        if   key == (True,  True,  False): oot.append((h_i, r_i, t_i))
        elif key == (False, True,  True ): xoo.append((h_i, r_i, t_i))
        elif key == (True,  False, True ): oxo.append((h_i, r_i, t_i))
        elif key == (True,  False, False): oxx.append((h_i, r_i, t_i))
        elif key == (False, False, True ): xxo.append((h_i, r_i, t_i))
        elif key == (False, True,  False): xox.append((h_i, r_i, t_i))
        # (True, True, True) → O-O-O: skip (closed-world, not evaluated)

    n_classified = len(oot)+len(xoo)+len(oxo)+len(oxx)+len(xxo)+len(xox)
    print(f"  Test triple breakdown:")
    print(f"    O-O-X  tail OOK:      {len(oot):>5}  → G1(head) G3(head) G4(rel)")
    print(f"    X-O-O  head OOK:      {len(xoo):>5}  → G2(tail) G3(tail) G4(rel)")
    print(f"    O-X-O  rel OOK:       {len(oxo):>5}  → G1(head) G2(tail)")
    print(f"    O-X-X  rel+tail OOK:  {len(oxx):>5}  → G1(head)")
    print(f"    X-X-O  head+rel OOK:  {len(xxo):>5}  → G2(tail)")
    print(f"    X-O-X  head+tail OOK: {len(xox):>5}  → G4(rel)")
    print(f"    O-O-O  all in-KG:     {len(eval_int)-n_classified:>5}  → closed-world (not evaluated here)")

    # ── Paper-exact 4-group evaluation ──────────────────────────────────────
    # Table 2 / Group 1 – Head entity prediction (O-O-X, O-X-X, O-X-O patterns)
    _g1_t0 = time.time()
    print(f"\n  Group 1 – Head entity prediction (Table 2: O-O-X + O-X-X + O-X-O):"
          f"  [{len(oot)+len(oxx)+len(oxo)} triples]")
    g1_ranks = _rank_gpu(oot + oxx + oxo, 'head', filter_heads)
    _g1_mrr  = float(np.mean(1.0 / np.array(g1_ranks, dtype=float))) if g1_ranks else 0.0
    print(f"  --> G1 done in {time.time()-_g1_t0:.0f}s  |  MRR={_g1_mrr:.4f}")

    # Table 3 / Group 2 – Tail entity prediction (X-O-O, X-X-O, O-X-O patterns)
    _g2_t0 = time.time()
    print(f"\n  Group 2 – Tail entity prediction (Table 3: X-O-O + X-X-O + O-X-O):"
          f"  [{len(xoo)+len(xxo)+len(oxo)} triples]")
    g2_ranks = _rank_gpu(xoo + xxo + oxo, 'tail', filter_tails)
    _g2_mrr  = float(np.mean(1.0 / np.array(g2_ranks, dtype=float))) if g2_ranks else 0.0
    print(f"  --> G2 done in {time.time()-_g2_t0:.0f}s  |  MRR={_g2_mrr:.4f}")

    # Table 4 / Group 3 – Head+Tail entity prediction (O-O-X head, X-O-O tail)
    print(f"\n  Group 3 – Head+Tail entity prediction (Table 4: O-O-X head + X-O-O tail):")
    g3h_ranks = g1_ranks[:len(oot)]
    g3t_ranks = g2_ranks[:len(xoo)]
    g3_ranks  = g3h_ranks + g3t_ranks
    _g3_mrr   = float(np.mean(1.0 / np.array(g3_ranks, dtype=float))) if g3_ranks else 0.0
    print(f"    (reused from G1/G2: {len(g3h_ranks)} head ranks + {len(g3t_ranks)} tail ranks)"
          f"  |  MRR={_g3_mrr:.4f}")

    # Table 5 / Group 4 – Relation prediction (O-O-X, X-O-O, X-O-X patterns)
    _g4_t0 = time.time()
    print(f"\n  Group 4 – Relation prediction (Table 5: O-O-X + X-O-O + X-O-X):"
          f"  [{len(oot)+len(xoo)+len(xox)} triples]")
    g4_ranks = _rank_gpu(oot + xoo + xox, 'relation', filter_rels)
    _g4_mrr  = float(np.mean(1.0 / np.array(g4_ranks, dtype=float))) if g4_ranks else 0.0
    print(f"  --> G4 done in {time.time()-_g4_t0:.0f}s  |  MRR={_g4_mrr:.4f}")

    def _metrics(rank_list: list, k_vals=(1, 3, 10)) -> dict:
        if not rank_list:
            return {'mrr': 0.0, 'mr': 0.0, 'n': 0, **{f'hits@{k}': 0.0 for k in k_vals}}
        a  = np.array(rank_list, dtype=float)
        m  = {'mrr': float(np.mean(1.0 / a)), 'mr': float(np.mean(a)), 'n': len(a)}
        for k in k_vals:
            m[f'hits@{k}'] = float(np.mean(a <= k))
        return m

    kv = [1, 3, 10]
    group_results = {
        'Group 1 - Head entity prediction (Table 2: O-O-X+O-X-X+O-X-O)': _metrics(g1_ranks, kv),
        'Group 2 - Tail entity prediction (Table 3: X-O-O+X-X-O+O-X-O)': _metrics(g2_ranks, kv),
        'Group 3 - Head+Tail OOK entity (Table 4: O-O-X head+X-O-O tail)': _metrics(g3_ranks, kv),
        'Group 4 - Relation prediction (Table 5: O-O-X+X-O-O+X-O-X)':    _metrics(g4_ranks, kv),
    }
    all_ranks = g1_ranks + g2_ranks + g3_ranks + g4_ranks
    group_results['overall'] = _metrics(all_ranks, kv)

    # Pretty-print
    # NOTE: paper_mrr contains FB20k+ reference values from the original paper.
    # DBPedia50k+ values (0.34/0.61/0.52/0.31) have been removed to avoid
    # misleading comparisons; update these once FB20k+ paper baselines are known.
    paper_mrr: dict = {}   # populated when FB20k+ baselines are known
    sep = '─' * 80
    print(f"\n{sep}")
    print(f"  IKGE FB20k+ Group Evaluation  (paper baseline: n/a — update paper_mrr dict)")
    print(sep)
    print(f"  {'Group':<52} {'n':>5}  {'MRR':>7}  {'H@1':>6}  {'H@3':>6}  {'H@10':>6}  {'MR':>7}")
    print(sep)
    for gname, gm in group_results.items():
        if gname == 'overall':
            continue
        paper = paper_mrr.get(gname)
        flag  = f" (paper MRR={paper:.3f})" if paper is not None else ""
        print(f"  {gname:<52} {gm['n']:>5}  {gm['mrr']:>7.4f}  "
              f"{gm.get('hits@1',0):>6.4f}  {gm.get('hits@3',0):>6.4f}  "
              f"{gm.get('hits@10',0):>6.4f}  {gm.get('mr',0):>7.1f}{flag}")
    ov = group_results['overall']
    print(sep)
    print(f"  {'Overall (all groups)':<52} {ov['n']:>5}  {ov['mrr']:>7.4f}  "
          f"{ov.get('hits@1',0):>6.4f}  {ov.get('hits@3',0):>6.4f}  "
          f"{ov.get('hits@10',0):>6.4f}  {ov.get('mr',0):>7.1f}")
    print(f"{sep}\n")

    # Populate scorer for optional PDF report
    if report_filename:
        scorer.group_data    = group_results
        scorer.ranking_data  = {'ranks': np.array(all_ranks),
                                'metrics': group_results['overall'],
                                'k_values': kv}
        scorer.export_report("IKGE FB20k+ Model", filename=report_filename)

    return group_results.get('overall', {}).get('mrr', 0.0)


def main(fraction: float = 1.0, run_name: str = "",
         epochs: int = 200, eval_every: int = 1):
    # -----------------------------------------------------------------------
    # Logging – every print also lands in a timestamped log file
    # -----------------------------------------------------------------------
    ts        = time.strftime("%Y%m%d_%H%M%S")
    tag       = f"_{run_name}" if run_name else ""
    log_dir   = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path  = str(log_dir / f"fb20k_train_{ts}{tag}.log")
    logger    = TeeLogger(log_path)
    sys.stdout = logger
    print(f"Logging to: {log_path}")

    try:
        _main(fraction=fraction, ts=ts, epochs=epochs, eval_every=eval_every)
    finally:
        logger.close()


def _main(fraction: float = 1.0, ts: str = "",
          epochs: int = 200, eval_every: int = 1):
    if not ts:
        ts = time.strftime("%Y%m%d_%H%M%S")
    print("="*80)
    print("Initializing IKGE FB20k+ Pipeline")
    print("="*80)

    # -----------------------------------------------------------------------
    # Config  (aligned with paper: fact_emb_dim=256, max_desc_len=50, epochs>=200)
    # -----------------------------------------------------------------------
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Enable TF32 Tensor Core matmuls on Ampere/Ada/Blackwell GPUs (free throughput gain)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    word_emb_dim     = 300
    fact_emb_dim     = 300   # paper: single d=300 throughout (Section 5.2.4)
    conv_channels    = 300   # paper: single d=300 throughout
    num_layers       = 2     # K=2: DBPedia50k+ has avg 2.7 facts/entity — K=3 rehashes
                           #      the same sparse neighbours (see Gap #6 in paper_code_correspondence.md)
    dropout          = 0.1   # reduced from 0.25: with score gap ~0.03 dropout was masking signal
    max_desc_len     = 50
    # epochs / eval_every come from caller (default 200 / 1; overridable via --epochs / --eval-every)
    # eval_every=1: validate_loss is a cheap forward pass, same cost as training.
    # Increase only if you want reduced console output.
    MARGIN           = 0.5   # hinge margin: reduced from 1.0; with logits starting near 0
    OOK_DROPOUT_P    = 0.3   # probability per triple of blanking one side's BFS frontier
                             # to simulate OOK entities during training (forces the model
                             # to learn text-only scoring, matching the eval condition
                             # where the true answer has an empty neighbourhood)
    train_batch_size= 256  # mini-batch size
    max_neighbor_facts  = 32   # BFS cap per subgraph: 500 OOMed (builds B×2×501 CNN inputs)
    print(f"Using device: {device}")

    # Output directory anchored to the script's own location
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(exist_ok=True, parents=True)

    # -----------------------------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------------------------
    data_dir = get_dataset_dir(dataset_dir='/workspace/data/FB20k+')

    def load_txt(path):
        with open(path, 'r') as f:
            return [line.strip().split('\t') for line in f if line.strip()]

    train_triples = load_txt(os.path.join(data_dir, 'train.txt'))   # list of [h, r, t]
    val_triples   = load_txt(os.path.join(data_dir, 'valid.txt'))
    test_triples  = load_txt(os.path.join(data_dir, 'test.txt'))

    # Subsample training triples when fraction < 1.0  (for quick experiments)
    if fraction < 1.0:
        random.seed(42)
        random.shuffle(train_triples)
        train_triples = train_triples[:max(1, int(len(train_triples) * fraction))]
        print(f"Using {fraction*100:.0f}% of training data  →  {len(train_triples):,} triples")
    # Sets of in-KG entity/relation strings (entities/relations seen during training).
    # Used to classify test triples into paper-defined groups.
    train_ent_set = set(t[0] for t in train_triples) | set(t[2] for t in train_triples)
    train_rel_set = set(t[1] for t in train_triples)
    print(f"In-KG entities: {len(train_ent_set):,}  |  In-KG relations: {len(train_rel_set):,}")
    entity2desc_raw = load_txt(os.path.join(data_dir, 'entity2text.txt'))
    entity2desc     = {x[0]: x[1] for x in entity2desc_raw if len(x) == 2}

    # Multi-hot entity types: one entity may appear on multiple lines.
    # Normalize 'http://www.w3.org/2002/07/owl#Thing' → 'dbo:Thing' so that
    # entity type strings are consistent with relation2constraint.txt (which
    # uses the short 'dbo:Thing' form). Without this, any entity typed as
    # owl#Thing fails every dbo:Thing type constraint check, zeroing its
    # fact features and breaking training gradient flow.
    entity2types = defaultdict(list)
    for x in load_txt(os.path.join(data_dir, 'entity2type.txt')):
        if len(x) == 2:
            typ = x[1]
            if 'owl#Thing' in typ:   # canonical normalisation
                typ = 'dbo:Thing'
            entity2types[x[0]].append(typ)

    rel2constraint_raw = load_txt(os.path.join(data_dir, 'relation2constraint.txt'))
    # Normalize owl#Thing → dbo:Thing in constraint types (same canonical form as
    # entity types above; relation2constraint.txt mixes both URI forms).
    def _norm_type(t):
        return 'dbo:Thing' if 'owl#Thing' in t else t
    # Collect ALL domain/range type constraints per relation (paper Section 3).
    # A plain dict comprehension would silently keep only the LAST row for each
    # relation; defaultdict(list) preserves every constraint.
    rel2domain = defaultdict(list)
    rel2range  = defaultdict(list)
    for x in rel2constraint_raw:
        if len(x) == 3:
            rel2domain[x[0]].append(_norm_type(x[1]))
            rel2range[x[0]].append(_norm_type(x[2]))

    # -----------------------------------------------------------------------
    # 2. Build entity / relation maps from ALL triples  (FIX: complete coverage)
    # -----------------------------------------------------------------------
    all_triples = train_triples + val_triples + test_triples
    all_entities_sorted  = sorted(set(t[0] for t in all_triples) | set(t[2] for t in all_triples))
    all_relations_sorted = sorted(set(t[1] for t in all_triples))

    ent2id = {e: i for i, e in enumerate(all_entities_sorted)}
    rel2id = {r: i for i, r in enumerate(all_relations_sorted)}
    id2ent = {i: e for e, i in ent2id.items()}
    id2rel = {i: r for r, i in rel2id.items()}

    # Relation names: use the last URI path component, underscores -> spaces
    relation2name = {
        r: r.split('/')[-1].split('#')[-1].replace('_', ' ')
        for r in all_relations_sorted
    }

    # -----------------------------------------------------------------------
    # 3. Build type vocabulary from entity types + relation constraints
    # -----------------------------------------------------------------------
    all_types = sorted(
        set(typ for types in entity2types.values() for typ in types)
        | set(t for ts in rel2domain.values() for t in ts)
        | set(t for ts in rel2range.values() for t in ts)
    )
    type2idx  = {t: i for i, t in enumerate(all_types)}
    num_types = len(type2idx)
    print(f"Entities: {len(ent2id)} | Relations: {len(rel2id)} | Types: {num_types}")

    # -----------------------------------------------------------------------
    # 4. Wikipedia2Vec embeddings  (~100% coverage on DBPedia descriptions)
    # -----------------------------------------------------------------------
    # Shared vocabulary W must include words from ALL side information (paper
    # Section 5.1.1): entity descriptions D_e, relation names U_r, entity names
    # U_e (last URI segment), and type constraint names T_{r,*}.
    # Without explicit inclusion, entity-name tokens (U_e) and type-name tokens
    # (T_{r,*}) would silently fall back to <UNK> if their words don't happen
    # to appear in entity descriptions.
    def _entity_name(e: str, desc: str = '') -> str:
        """Return a human-readable name for an entity.
        DBPedia:  dbr:Rich_Harrison   → 'Rich Harrison'  (URI segment is the name)
        FB20k+:   /m/010016 or /m/0c94fn → first 4 words of description ('Denton is a city')
        Heuristic: if the URI segment has >40% non-alpha chars it's a Freebase MID.
        """
        seg = e.split('/')[-1].split('#')[-1].replace('_', ' ')
        non_alpha = sum(1 for c in seg if not c.isalpha() and c != ' ')
        if seg and non_alpha > len(seg) * 0.4:
            words = desc.split()[:4]
            return ' '.join(words) if words else seg
        return seg

    entity_name_strings = [
        _entity_name(e, entity2desc.get(e, '')) for e in all_entities_sorted
    ]
    type_name_strings = [
        t.split('/')[-1].split('#')[-1].replace('_', ' ') for t in all_types
    ]
    descriptions = (list(entity2desc.values())
                    + list(relation2name.values())
                    + entity_name_strings
                    + type_name_strings)
    embedding_matrix, word2idx, _ = setup_w2v_for_ikge(
        entity_descriptions=descriptions,
        output_dir=str(output_dir / 'embeddings'),
        embedding_dim=word_emb_dim
    )

    # -----------------------------------------------------------------------
    # 5. Build line graph with integer-ID tensor  (FIX: was passing string lists)
    # -----------------------------------------------------------------------
    print("\nBuilding Line Graph...")
    id_train_triples = [
        (ent2id[h], rel2id[r], ent2id[t])
        for h, r, t in train_triples
    ]
    train_triple_tensor = torch.tensor(id_train_triples, dtype=torch.long)
    fact_edge_index, _ = create_line_graph(train_triple_tensor)
    fact_edge_index = fact_edge_index.to(device)

    # -----------------------------------------------------------------------
    # 6. Positive set for filtered negative sampling  (FIX: was unfiltered)
    # -----------------------------------------------------------------------
    positive_set = set(tuple(x) for x in id_train_triples)
    for h, r, t in val_triples + test_triples:
        if h in ent2id and r in rel2id and t in ent2id:
            positive_set.add((ent2id[h], rel2id[r], ent2id[t]))

    num_ents = len(all_entities_sorted)

    # -----------------------------------------------------------------------
    # 7. Pre-tokenize all entities and relations ONCE (fast epoch-level indexing)
    # -----------------------------------------------------------------------
    print("\nPre-tokenizing entities and relations...")
    ent_desc, ent_len, ent_type, ent_names = precompute_entity_tensors(
        all_entities_sorted, entity2desc, entity2types, type2idx, word2idx,
        max_desc_len, num_types
    )
    rel_name_t, rel_domain_t, rel_range_t, rel_domain_words_t, rel_range_words_t = \
        precompute_relation_tensors(
            all_relations_sorted, relation2name, rel2domain, rel2range,
            type2idx, word2idx, num_types
        )

    # Move all lookup tables to GPU once — every downstream index into these
    # tensors (build_batch_from_precomputed, diagnostic block, etc.) then runs
    # entirely on GPU with no per-batch host→device copies.
    #   ent_type  : ~26 MB   ent_desc  :  ~6 MB   ent_names : ~1 MB
    #   ent_len   : ~0.1 MB  rel_*     :  <1 MB total
    print("Moving lookup tables to GPU...")
    ent_desc           = ent_desc.to(device)
    ent_len            = ent_len.to(device)
    ent_type           = ent_type.to(device)
    ent_names          = ent_names.to(device)
    rel_name_t         = rel_name_t.to(device)
    rel_domain_t       = rel_domain_t.to(device)
    rel_range_t        = rel_range_t.to(device)
    rel_domain_words_t = rel_domain_words_t.to(device)
    rel_range_words_t  = rel_range_words_t.to(device)
    print("  Lookup tables on GPU.")

    # Fixed integer-ID arrays for the training triples (never change between epochs)
    pos_h_ids = torch.tensor([ent2id[h] for h, r, t in train_triples], dtype=torch.long)
    pos_r_ids = torch.tensor([rel2id[r] for h, r, t in train_triples], dtype=torch.long)
    pos_t_ids = torch.tensor([ent2id[t] for h, r, t in train_triples], dtype=torch.long)

    # Python int-lists for zero-overhead subgraph BFS (no .item() calls in the loop)
    pos_h_list = pos_h_ids.tolist()
    pos_r_list = pos_r_ids.tolist()
    pos_t_list = pos_t_ids.tolist()

    # entity_id -> list of training fact indices  (for K-hop subgraph sampling)
    entity_to_facts: dict = {}
    for _i in range(len(pos_h_list)):
        for _e in (pos_h_list[_i], pos_t_list[_i]):
            entity_to_facts.setdefault(_e, []).append(_i)
    print(f"entity_to_facts: {len(entity_to_facts)} entities indexed over "
          f"{len(pos_h_list)} training facts.")

    # Sorted list of in-KG entity IDs (entities with training neighbours).
    # Used by validate_loss and the training loop for negative sampling:
    # sampling from ALL entities gives a structural shortcut (OOK entities have
    # empty subgraphs and are trivially easy negatives, inflating scores even
    # with a random model).
    in_kg_ents = sorted(entity_to_facts.keys())

    # ------------------------------------------------------------------
    # Pre-build type-constrained negative-sampling buckets.
    # rel_tail_type_ents[r] = in-KG entities whose type vector has any
    #   overlap with rel_range_t[r]  (semantically valid tail candidates).
    # rel_head_type_ents[r] = in-KG entities matching rel_domain_t[r].
    # Used as *hard negative* pools during training: the model cannot
    # reject them on type grounds alone and must read descriptions.
    # Fallback to uniform in_kg_ents when a bucket has < 5 entries.
    # (See paper_code_correspondence.md — Gap 16.)
    # ------------------------------------------------------------------
    print("  Building type-constrained negative-sampling buckets...")
    _et_cpu  = ent_type.cpu()            # [n_ents, num_types]
    _rr_cpu  = rel_range_t.cpu()         # [n_rels, num_types]
    _rd_cpu  = rel_domain_t.cpu()        # [n_rels, num_types]
    _in_kg_tensor = torch.tensor(in_kg_ents, dtype=torch.long)
    _et_in_kg = _et_cpu[_in_kg_tensor]   # [n_in_kg, num_types]  — only in-KG rows
    rel_tail_type_ents: dict[int, list[int]] = {}
    rel_head_type_ents: dict[int, list[int]] = {}
    _n_rels_local = _rr_cpu.size(0)
    for _r in range(_n_rels_local):
        # --- tail bucket (range constraint) ---
        _range_mask = _rr_cpu[_r]        # [num_types]
        if _range_mask.sum() > 0:
            _match = ((_et_in_kg * _range_mask.unsqueeze(0)).sum(1) > 0).nonzero(as_tuple=True)[0]
            _bucket = [in_kg_ents[_idx.item()] for _idx in _match]
            if len(_bucket) >= 5:
                rel_tail_type_ents[_r] = _bucket
        # --- head bucket (domain constraint) ---
        _domain_mask = _rd_cpu[_r]       # [num_types]
        if _domain_mask.sum() > 0:
            _match = ((_et_in_kg * _domain_mask.unsqueeze(0)).sum(1) > 0).nonzero(as_tuple=True)[0]
            _bucket = [in_kg_ents[_idx.item()] for _idx in _match]
            if len(_bucket) >= 5:
                rel_head_type_ents[_r] = _bucket
    del _et_cpu, _rr_cpu, _rd_cpu, _et_in_kg, _in_kg_tensor
    print(f"  Type buckets: {len(rel_tail_type_ents)}/{_n_rels_local} rels have tail buckets, "
          f"{len(rel_head_type_ents)}/{_n_rels_local} rels have head buckets.")

    # Sorted list of ALL entity IDs (for unrestricted negative sampling, paper Sec 5.2.2)
    all_ent_ids = list(range(num_ents))

    print("Pre-tokenization complete.")

    # -----------------------------------------------------------------------
    # 8. Initialize model
    # -----------------------------------------------------------------------
    model = IKGENetwork(
        embedding_matrix=embedding_matrix,
        word_emb_dim=word_emb_dim,
        fact_emb_dim=fact_emb_dim,
        conv_channels=conv_channels,
        num_types=num_types,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )

    # Word embeddings are frozen (paper Section 5.1.1) so requires_grad=False already.
    other_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in other_params):,}")

    # torch.compile is intentionally NOT used here.
    # The BFS training loop produces a dynamically-sized edge_index and all_z
    # tensor every mini-batch.  The compiled model's forward() is traced for a
    # static graph, so calling model(all_z[pq]) on a tensor that flowed through
    # the uncompiled model.aggregator breaks the autograd graph: PyTorch's dynamo
    # cannot stitch the two computation paths and silently produces constant
    # output with zero gradients (loss locks at exactly 2*ln(2)=1.3863).
    # The paper authors used vanilla PyTorch (pre-2.0, no compile); matching that.

    # lr=1e-3 (reduced from paper's 1e-2): the 3-hop residual aggregator accumulates
    # feature magnitudes that even after LayerNorm can produce large gradient steps
    # with lr=0.01, destabilising training in the first few epochs.
    optimizer = torch.optim.AdamW(other_params, lr=1e-3, weight_decay=1e-3)
    # Paper (Section 6.1): cosine annealing LR scheduler, no T_max given.
    # We use T_max=1000 so the LR decays very gently (~0.7% per epoch at epoch 10),
    # leaving early stopping as the true convergence criterion rather than
    # an artificial LR floor imposed by a short T_max.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # BF16 AMP: uses Tensor Cores on RTX 30/40/50-series without loss scaling.
    # BF16 has identical dynamic range to FP32 so training trajectory is unchanged.
    use_amp = device.type == 'cuda'

    metadata = (entity2desc, relation2name, entity2types, rel2domain, rel2range)

    # -----------------------------------------------------------------------
    # 9. Training loop  — BFS subgraph training with real attention gradients
    #
    #   Each mini-batch:
    #     1. For each pos (h,r,t) and neg (h',r,t'), BFS K hops from h and t
    #        to collect neighbor training facts (excluding (h,r,t) itself to
    #        prevent self-referential topology leakage).
    #     2. Build a disjoint union of small subgraphs.
    #     3. extract_fact_features on all nodes → z^(0) for every node.
    #     4. Run model.aggregator (K layers of attention) over the line-graph
    #        edges → z^(K) with REAL gradients flowing into attention_layers.
    #     5. Extract virtual-query-node representations for pos and neg.
    #     6. Score with MLP and compute BCE loss (Paper Eq 13).
    # -----------------------------------------------------------------------
    # Early stopping: stop when val loss hasn't improved for `patience` eval checks.
    # `epochs` is a hard max ceiling (safety net), not the intended stop.
    PATIENCE         = 20           # epochs without improvement before stopping
    early_stop_count = 0            # consecutive eval checks with no improvement
    print("\nStarting Training Loop...")
    best_val_loss    = float('inf')
    window_train_sum = 0.0          # sum of per-epoch losses in current window
    window_train_cnt = 0            # number of epochs in current window
    n_train          = len(train_triples)

    weights_path_mrr = str(output_dir / f"fb20k_ikge_w2v_best_mrr_{ts}.pt")
    report_path      = str(output_dir / f"fb20k_ikge_w2v_evaluation_report_{ts}.pdf")

    diag_pos: list = []   # score-diagnostic accumulators (reset every 10 epochs)
    diag_neg: list = []

    # Reverse index: (h_int, r_int, t_int) -> training fact ID.
    # Used to exclude each positive from its own BFS neighbourhood
    # (prevents self-referential topology leakage; paper Section 5.2.3
    # aggregates *other* training facts' features, not the fact itself).
    triple_to_fact_id = {
        (pos_h_list[i], pos_r_list[i], pos_t_list[i]): i
        for i in range(len(pos_h_list))
    }
    print(f"  triple_to_fact_id built: {len(triple_to_fact_id)} entries.")

    # Pre-generate a FIXED set of val negatives (one neg per val triple, sampled
    # once and reused every epoch).  Re-sampling each epoch introduces large
    # stochastic noise in val_loss (up to 2× between consecutive epochs even when
    # the model hasn't changed), which causes early stopping to fire on noise
    # rather than on a real plateau.
    print("  Pre-generating fixed validation negatives...")
    val_int_fixed = []
    for h, r, t in val_triples:
        if h in ent2id and r in rel2id and t in ent2id:
            val_int_fixed.append((ent2id[h], rel2id[r], ent2id[t]))
    if val_int_fixed:
        _vbh = torch.tensor([h for h, r, t in val_int_fixed], dtype=torch.long)
        _vbr = torch.tensor([r for h, r, t in val_int_fixed], dtype=torch.long)
        _vbt = torch.tensor([t for h, r, t in val_int_fixed], dtype=torch.long)
        val_neg_h_fixed, val_neg_t_fixed = generate_neg_indices(
            _vbh, _vbr, _vbt, positive_set, in_kg_ents,
            rel_tail_type_ents=rel_tail_type_ents,
            rel_head_type_ents=rel_head_type_ents)
        fixed_val_negs = (val_neg_h_fixed, val_neg_t_fixed)
        print(f"  Fixed val negatives: {len(val_int_fixed)} pairs ready.")
    else:
        fixed_val_negs = None
        print("  Warning: no fully in-KG val triples found for fixed negatives.")

    # -----------------------------------------------------------------------
    # Type-matching coverage diagnostic
    # If a large fraction of training triples have zero type validity, gradients
    # are blocked for those triples and the model cannot learn from them.
    # -----------------------------------------------------------------------
    print("  Type-matching diagnostic (checking training triples)...")
    with torch.no_grad():
        _sample = min(2000, len(pos_h_list))
        _bh = pos_h_ids[:_sample].to(device)
        _br = pos_r_ids[:_sample].to(device)
        _bt = pos_t_ids[:_sample].to(device)
        _ht = ent_type[_bh]          # (N, num_types)
        _tt = ent_type[_bt]          # (N, num_types)
        _hno  = (_ht.sum(dim=1) == 0).float().mean().item()
        _tno  = (_tt.sum(dim=1) == 0).float().mean().item()
        _rd   = rel_domain_t[_br]    # (N, num_types)
        _rr   = rel_range_t[_br]     # (N, num_types)
        _rno  = (_rd.sum(dim=1) + _rr.sum(dim=1) == 0).float().mean().item()
        _hmatch = (((_ht * _rd).sum(dim=1) > 0) | (_rd.sum(dim=1) == 0)).float().mean().item()
        _tmatch = (((_tt * _rr).sum(dim=1) > 0) | (_rr.sum(dim=1) == 0)).float().mean().item()
        _valid  = (_hmatch + _tmatch) / 2
        print(f"    Head entities with NO type: {_hno*100:.1f}%")
        print(f"    Tail entities with NO type: {_tno*100:.1f}%")
        print(f"    Relations with NO constraint: {_rno*100:.1f}%")
        print(f"    Head-domain match rate (flat intersection): {_hmatch*100:.1f}%")
        print(f"    Tail-range  match rate (flat intersection): {_tmatch*100:.1f}%")
        print(f"    Triples passing flat type check: {_valid*100:.1f}%  "
              f"(soft gate 0.1 floor ensures gradient flow for remaining {(1-_valid)*100:.1f}%)")

    for epoch in range(epochs):
        epoch_start     = time.time()
        model.train()
        perm            = torch.randperm(n_train)
        mb_indices      = [perm[i:i + train_batch_size]
                           for i in range(0, n_train, train_batch_size)]
        epoch_loss      = 0.0
        epoch_grad_norm = 0.0   # accumulated pre-clip gradient norm (avg over mini-batches)

        for bidx in mb_indices:
            optimizer.zero_grad()
            bh = pos_h_ids[bidx]; br = pos_r_ids[bidx]; bt = pos_t_ids[bidx]
            # Documented deviation from paper Section 5.2.2: restrict negatives
            # to in-KG entities only.  Using all_ent_ids creates a structural
            # shortcut — OOK-entity negatives have empty BFS subgraphs and the
            # MLP trivially learns "rich subgraph = positive" in ~10 epochs,
            # collapsing loss to ln(2) without learning any text/type features.
            neg_h, neg_t = generate_neg_indices(
                bh, br, bt, positive_set, in_kg_ents,
                rel_tail_type_ents=rel_tail_type_ents,
                rel_head_type_ents=rel_head_type_ents)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                # ── Paper Section 5.2 training ───────────────────────────────
                # For each positive (h,r,t) and paired negative (h',r,t'),
                # build a K-hop BFS subgraph of training facts reachable from
                # h and t.  The positive's own fact is excluded from its
                # neighbourhood to prevent self-referential leakage.
                # All nodes in the subgraph + virtual query nodes are passed
                # through the fact feature extractor and K-layer attentive
                # aggregator (Equations 6-11).  Gradients flow end-to-end.
                feat_tensors, edge_index, pq, nq = build_training_batch(
                    bh, br, bt, neg_h, neg_t,
                    entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
                    ent_desc, ent_len, ent_type, ent_names,
                    rel_name_t, rel_domain_t, rel_range_t,
                    rel_domain_words_t, rel_range_words_t,
                    device, K=num_layers, max_neighbor_facts=max_neighbor_facts,
                    triple_to_fact_id=triple_to_fact_id,
                    ook_dropout_p=OOK_DROPOUT_P)

                # CNN on all subgraph nodes → z^(0) (initial fact embeddings)
                all_z = model.extract_fact_features(feat_tensors).float()
                # K-layer attentive aggregation over the line-graph subgraph
                all_z = model.aggregator(all_z, edge_index)

                pos_scores = model(all_z[pq], return_logits=True)
                neg_scores = model(all_z[nq], return_logits=True)

            # Pairwise margin / hinge ranking loss:
            #   L = mean(ReLU(MARGIN - pos_logit + neg_logit))
            # Directly teaches "pos must score at least MARGIN above neg".
            # BCE pushes pos→1 and neg→0 independently; with pos≈0.51 and
            # neg≈0.48 the two gradient terms partially cancel in shared
            # weights and the model stalls near the random-chance plateau.
            # Hinge loss: gradient is non-zero only when the margin is
            # violated (pos_logit - neg_logit < MARGIN), giving a clean
            # directional signal in logit difference space.
            ps_logit = pos_scores.float()
            ns_logit = neg_scores.float()
            loss = F.relu(MARGIN - ps_logit + ns_logit).mean()

            loss.backward()

            # Accumulate score stats for diagnostics (float32, detached, probabilities)
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    ps = torch.sigmoid(ps_logit.detach())
                    ns = torch.sigmoid(ns_logit.detach())
                    diag_pos.append((ps.mean().item(), ps.std().item()))
                    diag_neg.append((ns.mean().item(), ns.std().item()))

            # Record pre-clip gradient norm BEFORE the optimizer touches weights.
            # A consistently tiny norm (<1e-4) indicates vanishing gradients.
            grad_norm_val = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_val += p.grad.data.norm(2).item() ** 2
            epoch_grad_norm += grad_norm_val ** 0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        loss_for_log   = epoch_loss / len(mb_indices)
        avg_grad_norm  = epoch_grad_norm / len(mb_indices)
        epoch_dur      = time.time() - epoch_start
        epoch_ts       = time.strftime("%H:%M:%S")
        scheduler.step()  # cosine annealing advances once per epoch

        # Print score-separation diagnostics every 10 epochs
        if (epoch + 1) % 10 == 0 and diag_pos:
            avg_pos    = np.mean([m for m, _ in diag_pos])
            avg_neg    = np.mean([m for m, _ in diag_neg])
            std_pos    = np.mean([s for _, s in diag_pos])
            margin_gap = avg_pos - avg_neg
            print(f"  [Score diag] pos={avg_pos:.4f}±{std_pos:.4f}  neg={avg_neg:.4f}  "
                  f"gap={margin_gap:+.4f}  (gap>0 = model learning)")
            diag_pos, diag_neg = [], []   # reset for next window

        # Accumulate loss for window average reported at validation time
        window_train_sum += loss_for_log
        window_train_cnt += 1

        # -- Periodic validation: val loss (identical mechanics to training —
        #    never exposes val triples to the optimizer) -----------------------
        if (epoch + 1) % eval_every == 0:
            lr_now          = optimizer.param_groups[0]['lr']
            window_avg_loss = window_train_sum / max(window_train_cnt, 1)
            window_train_sum, window_train_cnt = 0.0, 0   # reset window

            print(f"\n[Epoch {epoch+1}/{epochs}] Running loss validation "
                  f"(window avg train loss {window_avg_loss:.4f}  "
                  f"avg grad norm {avg_grad_norm:.4f})...")

            val_loss, val_pos_mean, val_neg_mean = validate_loss(
                model, val_triples, positive_set,
                entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
                ent_desc, ent_len, ent_type, ent_names,
                rel_name_t, rel_domain_t, rel_range_t,
                rel_domain_words_t, rel_range_words_t,
                ent2id, rel2id, device,
                in_kg_ents=in_kg_ents,
                num_layers=num_layers,
                batch_size=train_batch_size,
                max_neighbor_facts=max_neighbor_facts,
                margin=MARGIN,
                fixed_negs=fixed_val_negs,
                rel_tail_type_ents=rel_tail_type_ents,
                rel_head_type_ents=rel_head_type_ents,
            )
            loss_gap = val_loss - window_avg_loss
            score_gap = val_pos_mean - val_neg_mean

            # Saturation / gradient-death warning
            if abs(score_gap) < 0.02:
                print(f"  ⚠  SCORE SATURATION: pos≈neg≈{val_pos_mean:.3f} — "
                      f"model may not be learning (dead features / gradient failure)")
            if avg_grad_norm < 1e-4:
                print(f"  ⚠  VANISHING GRADIENTS: avg norm = {avg_grad_norm:.2e}")

            print(f"  Val loss: {val_loss:.4f}  |  train loss: {window_avg_loss:.4f}  "
                  f"|  gap: {loss_gap:+.4f}  "
                  f"({'overfit' if loss_gap > 0.2 else 'underfit' if loss_gap < -0.2 else 'ok'})")
            print(f"  Val scores: pos={val_pos_mean:.4f}  neg={val_neg_mean:.4f}  "
                  f"gap={score_gap:+.4f}")

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                early_stop_count = 0
                torch.save(model.state_dict(), weights_path_mrr)
                print(f"  Saved best val-loss weights! val_loss={val_loss:.4f} → {weights_path_mrr}")
                # Also save word2idx so eval scripts can reproduce exact vocab.
                import pickle as _pkl
                _vocab_path = weights_path_mrr.replace('.pt', '_word2idx.pkl')
                with open(_vocab_path, 'wb') as _f:
                    _pkl.dump({'word2idx': word2idx}, _f)
                print(f"  Saved word2idx ({len(word2idx):,} words) → {_vocab_path}")
            else:
                early_stop_count += 1
                print(f"  No improvement ({early_stop_count}/{PATIENCE} patience used)")

            print(f"Epoch {epoch+1:4d}/{epochs} | Train loss: {loss_for_log:.4f} "
                  f"| Val loss: {val_loss:.4f} | GradNorm: {avg_grad_norm:.4f} "
                  f"| LR: {lr_now:.2e} | {epoch_dur:.1f}s @ {epoch_ts}")

            if early_stop_count >= PATIENCE:
                print(f"\n  Early stopping: val loss did not improve for {PATIENCE} "
                      f"consecutive epochs (best={best_val_loss:.4f}).")
                break


    # -----------------------------------------------------------------------
    # 10. Final evaluation on the full test set using the best checkpoint
    # -----------------------------------------------------------------------
    # Prefer best-MRR weights for final eval; fall back to best-loss weights.
    print(f"\nLoading best weights for final test evaluation ({Path(weights_path_mrr).name})...")
    model.load_state_dict(torch.load(weights_path_mrr, map_location=device, weights_only=True))

    # -----------------------------------------------------------------------
    # Test hinge loss – same formula as training, on a fixed set of test
    # negatives.  Lets us compare train_loss / val_loss / test_loss directly
    # and verify the model generalises (test ≈ val) vs overfits (test >> val).
    # -----------------------------------------------------------------------
    print("Computing test hinge loss (apples-to-apples with training loss)...")
    test_int_for_loss = []
    for h, r, t in test_triples:
        if h in ent2id and r in rel2id and t in ent2id:
            test_int_for_loss.append((ent2id[h], rel2id[r], ent2id[t]))
    if test_int_for_loss:
        _tbh = torch.tensor([h for h,r,t in test_int_for_loss], dtype=torch.long)
        _tbr = torch.tensor([r for h,r,t in test_int_for_loss], dtype=torch.long)
        _tbt = torch.tensor([t for h,r,t in test_int_for_loss], dtype=torch.long)
        test_neg_h, test_neg_t = generate_neg_indices(
            _tbh, _tbr, _tbt, positive_set, in_kg_ents,
            rel_tail_type_ents=rel_tail_type_ents,
            rel_head_type_ents=rel_head_type_ents)
        fixed_test_negs = (test_neg_h, test_neg_t)
        test_hinge_loss, test_pos_mean, test_neg_mean = validate_loss(
            model, test_triples, positive_set,
            entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
            ent_desc, ent_len, ent_type, ent_names,
            rel_name_t, rel_domain_t, rel_range_t,
            rel_domain_words_t, rel_range_words_t,
            ent2id, rel2id, device,
            in_kg_ents=in_kg_ents,
            num_layers=num_layers,
            batch_size=train_batch_size,
            max_neighbor_facts=max_neighbor_facts,
            margin=MARGIN,
            fixed_negs=fixed_test_negs,
            rel_tail_type_ents=rel_tail_type_ents,
            rel_head_type_ents=rel_head_type_ents,
        )
        print(f"  Test  hinge loss : {test_hinge_loss:.4f}  "
              f"(val={best_val_loss:.4f}  diff={test_hinge_loss - best_val_loss:+.4f})")
        print(f"  Test  scores     : pos={test_pos_mean:.4f}  neg={test_neg_mean:.4f}  "
              f"gap={test_pos_mean - test_neg_mean:+.4f}")

    print("Running Final Test Evaluation on full test set...")
    test_mrr = evaluate_model(
        model, test_triples, metadata, word2idx, type2idx,
        max_desc_len, device,
        ent2id, rel2id, id2ent, id2rel,
        fact_edge_index, train_triples,
        ent_desc, ent_len, ent_type, ent_names,
        rel_name_t, rel_domain_t, rel_range_t,
        rel_domain_words_t, rel_range_words_t,
        all_triples_for_filter=all_triples,
        train_ent_set=train_ent_set,
        train_rel_set=train_rel_set,
        is_test=True,
        report_filename=report_path
    )

    print("="*80)
    print("Training Complete")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final Test MRR      : {test_mrr:.4f}")
    print(f"Report generated    : {report_path}")
    print(f"Best-loss weights   : {weights_path_mrr}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IKGE on DBPedia50k+ (Wikipedia2Vec)")
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of training triples to use, e.g. 0.1 for 10%% (default: 1.0)"
    )
    parser.add_argument(
        "--run-name", type=str, default="",
        help="Optional label appended to the log filename, e.g. 'debug' or 'frac10'"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Number of training epochs (default: 200)"
    )
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="Validate every N epochs (default: 5)"
    )
    args = parser.parse_args()
    main(fraction=args.fraction, run_name=args.run_name,
         epochs=args.epochs, eval_every=args.eval_every)
