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
from download_glove import setup_glove_for_ikge

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

        # Aggregate graph neighbourhood (K=3 for DBPedia50k+ per paper)
        self.aggregator = AttentiveAggregator(
            fact_embedding_dim=fact_emb_dim,
            num_layers=num_layers,
            device=device
        )

        # Scoring function: 2 FC layers with 512, 256 units → unbounded logit score.
        # Input is the K-layer aggregator output (tanh-saturated, mean≈0, values in (-1,1)).
        # A Dropout at input provides regularisation without killing the signal
        # (LayerNorm would collapse near-constant aggregator outputs to zero).
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
            relation_names=batch_data['rel_name'],
            head_types=batch_data['head_type'],
            tail_types=batch_data['tail_type'],
            relation_domain_types=batch_data['rel_domain'],
            relation_range_types=batch_data['rel_range'],
            head_desc_lengths=batch_data['head_len'],
            tail_desc_lengths=batch_data['tail_len']
        )
        
    def forward(self, features):
        """
        Score facts via Dropout + 2-layer MLP (paper Section 5.3).
        Input is the K-layer aggregator output (tanh-bounded, mean≈0).
        """
        x = self.score_drop(features)
        x = self.score_layer1(x)
        x = self.relu(x)
        x = self.score_layer2(x)
        x = self.relu(x)
        return self.score_out(x).squeeze(-1)


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
        h_words = entity2desc.get(h, 'unknown').lower().split()[:max_desc_len]
        t_words = entity2desc.get(t, 'unknown').lower().split()[:max_desc_len]
        r_words = relation2name.get(r, r.split('/')[-1].replace('_', ' ')).lower().split()[:10]

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

        # Relation type constraint encoding
        if r in rel2domain and rel2domain[r] in type2idx:
            rel_domain[i, type2idx[rel2domain[r]]] = 1.0
        if r in rel2range and rel2range[r] in type2idx:
            rel_range[i, type2idx[rel2range[r]]]  = 1.0

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
                               max_desc_len, num_types):
    """Tokenize every entity description and type vector once into CPU tensors."""
    n = len(entities)
    desc    = torch.zeros(n, max_desc_len, dtype=torch.long)
    lengths = torch.ones(n, dtype=torch.long)
    types   = torch.zeros(n, num_types, dtype=torch.float)
    for i, e in enumerate(entities):
        words = entity2desc.get(e, 'unknown').lower().split()[:max_desc_len]
        for j, w in enumerate(words):
            desc[i, j] = word2idx.get(w, 1)  # 1 = <UNK>
        lengths[i] = max(1, len(words))
        for typ in entity2types.get(e, []):
            if typ in type2idx:
                types[i, type2idx[typ]] = 1.0
    return desc.pin_memory(), lengths.pin_memory(), types.pin_memory()


def precompute_relation_tensors(relations, relation2name, rel2domain, rel2range,
                                 type2idx, word2idx, num_types):
    """Tokenize every relation name and constraint vector once into CPU tensors."""
    n = len(relations)
    rel_name_t   = torch.zeros(n, 10, dtype=torch.long)
    rel_domain_t = torch.zeros(n, num_types, dtype=torch.float)
    rel_range_t  = torch.zeros(n, num_types, dtype=torch.float)
    for i, r in enumerate(relations):
        name_words = relation2name.get(r, r.split('/')[-1].replace('_', ' ')).lower().split()[:10]
        for j, w in enumerate(name_words):
            rel_name_t[i, j] = word2idx.get(w, 1)
        if r in rel2domain and rel2domain[r] in type2idx:
            rel_domain_t[i, type2idx[rel2domain[r]]] = 1.0
        if r in rel2range and rel2range[r] in type2idx:
            rel_range_t[i, type2idx[rel2range[r]]] = 1.0
    return rel_name_t.pin_memory(), rel_domain_t.pin_memory(), rel_range_t.pin_memory()


def build_batch_from_precomputed(h_ids, r_ids, t_ids,
                                  ent_desc, ent_len, ent_type,
                                  rel_name_t, rel_domain_t, rel_range_t,
                                  device):
    """Assemble a batch dict from pre-computed CPU tensors via index selection."""
    return {
        'head_desc':  ent_desc[h_ids].to(device, non_blocking=True),
        'tail_desc':  ent_desc[t_ids].to(device, non_blocking=True),
        'head_len':   ent_len[h_ids].to(device, non_blocking=True),
        'tail_len':   ent_len[t_ids].to(device, non_blocking=True),
        'rel_name':   rel_name_t[r_ids].to(device, non_blocking=True),
        'head_type':  ent_type[h_ids].to(device, non_blocking=True),
        'tail_type':  ent_type[t_ids].to(device, non_blocking=True),
        'rel_domain': rel_domain_t[r_ids].to(device, non_blocking=True),
        'rel_range':  rel_range_t[r_ids].to(device, non_blocking=True),
    }


def generate_neg_indices(h_ids, r_ids, t_ids, positive_set, in_kg_ents):
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
    """
    n = len(h_ids)
    neg_h = h_ids.clone()
    neg_t = t_ids.clone()
    coin  = torch.randint(0, 2, (n,), dtype=torch.bool)
    num_kg = len(in_kg_ents)
    for i in range(n):
        h, r, t = h_ids[i].item(), r_ids[i].item(), t_ids[i].item()
        for _ in range(200):
            c = in_kg_ents[random.randint(0, num_kg - 1)]
            if coin[i]:           # corrupt head
                if (c, r, t) not in positive_set:
                    neg_h[i] = c
                    break
            else:                 # corrupt tail
                if (h, r, c) not in positive_set:
                    neg_t[i] = c
                    break
    return neg_h, neg_t


# ==============================================================================
# Per-batch subgraph aggregation helpers (paper-faithful IKGE training)
# ==============================================================================

def sample_subgraph_for_triple(qh, qt, entity_to_facts,
                                pos_h_list, pos_t_list, K=3, max_facts=8):
    """
    BFS K-hop neighbourhood of query entities (qh, qt) in the training LINE GRAPH
    (nodes = training facts, edges = shared entity).  Returns:
      fact_ids    : list of training fact indices collected during BFS
      virtual_idx : index of the query triple's virtual node = len(fact_ids)
      edge_src / edge_dst : undirected line-graph edges including edges to virtual node
    """
    visited  = {}           # fact_id -> local_index (dict preserves insertion order)
    frontier = {qh, qt}    # entity frontier
    seen_ents = set(frontier)

    for _ in range(K):
        if len(visited) >= max_facts:
            break
        next_ents = set()
        for e in frontier:
            for fi in entity_to_facts.get(e, []):
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
                         ent_desc, ent_len, ent_type,
                         rel_name_t, rel_domain_t, rel_range_t,
                         device, K=3, max_neighbor_facts=8):
    """
    Build a batched DISJOINT-UNION of subgraphs for all pos + neg pairs.

    Each positive (h,r,t) and negative (h',r,t') becomes a 'virtual query node'
    at the end of its own K-hop training-fact subgraph, with CURRENT parameters.

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
            fids, virt, esrc, edst = sample_subgraph_for_triple(
                qh, qt, entity_to_facts, pos_h_list, pos_t_list,
                K, max_neighbor_facts)

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
        ent_desc, ent_len, ent_type,
        rel_name_t, rel_domain_t, rel_range_t, device)

    if all_esrc:
        edge_index = torch.tensor([all_esrc, all_edst], dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    pq = torch.tensor(pos_q, dtype=torch.long, device=device)
    nq = torch.tensor(neg_q, dtype=torch.long, device=device)
    return feat_tensors, edge_index, pq, nq


def evaluate_model(model, eval_triples, metadata, word2idx, type2idx, max_desc_len,
                   device, ent2id, rel2id, id2ent, id2rel,
                   fact_edge_index, train_triples,
                   ent_desc, ent_len, ent_type,
                   rel_name_t, rel_domain_t, rel_range_t,
                   pos_tensors_cached,
                   all_triples_for_filter,
                   train_ent_set, train_rel_set,
                   is_test=False,
                   report_filename="ikge_evaluation_report.pdf"):
    """
    Evaluation dispatcher:
      is_test=False  →  bidirectional filtered MRR on val triples (scheduler signal)
      is_test=True   →  paper-exact 4-group evaluation on test triples

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
    # GPU lookup tables + entity mean-aggregated context for eval.
    # ---------------------------------------------------------------
    model.eval()
    _ed = ent_desc.to(device,     non_blocking=True)
    _el = ent_len.to(device,      non_blocking=True)
    _et = ent_type.to(device,     non_blocking=True)
    _rn = rel_name_t.to(device,   non_blocking=True)
    _rd = rel_domain_t.to(device, non_blocking=True)
    _rr = rel_range_t.to(device,  non_blocking=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # -----------------------------------------------------------------------
    # Multi-pass mean-field approximation of the K-layer attentive aggregator.
    #
    # Paper Equations 9-11:
    #   h_{N(f_u)}^{k+1} = tanh(sum_{v in N(f_u)} a_v * z_v^{(k)})   [Eq 9]
    #   f_tilde_u        = h_{N(f_u)}^{k+1} + f_u                     [Eq 10]
    #   f_u              <- f_tilde_u                                  [Eq 11]
    #
    # At eval we cannot build a per-query subgraph for all 30k candidates.
    # We instead pre-compute per-entity mean features at each layer k:
    #   entity_feat[e]^{(0)} = mean of raw CNN features for training facts with e
    #   entity_feat[e]^{(k+1)} = entity_mean( z^{(0)} + sum_{layers<=k} tanh_neigh )
    #
    # At eval for candidate (h, r, t_cand):
    #   z^{(0)} = extract_fact_features(h, r, t_cand)
    #   for k in 0..K-1:
    #       z += tanh(entity_feat[h]^{(k)} + entity_feat[t_cand]^{(k)})
    #   score = model(z)
    # -----------------------------------------------------------------------
    print("  Computing multi-pass entity context for eval...")
    n_train = len(train_triples)
    # Get original (uncompiled) aggregator for direct weight access.
    _agg_mod = model._orig_mod.aggregator if hasattr(model, '_orig_mod') else model.aggregator
    n_ent    = len(ent2id)

    with torch.no_grad():
        th_dev = torch.tensor([ent2id[h] for h, r, t in train_triples],
                               dtype=torch.long, device=device)
        tr_dev = torch.tensor([rel2id[r] for h, r, t in train_triples],
                               dtype=torch.long, device=device)
        tt_dev = torch.tensor([ent2id[t] for h, r, t in train_triples],
                               dtype=torch.long, device=device)
        CHUNK = 4096

        # Layer 0: raw CNN features for every training fact
        raw_chunks = []
        for s in range(0, n_train, CHUNK):
            e = min(s + CHUNK, n_train)
            ct = build_batch_from_precomputed(
                th_dev[s:e], tr_dev[s:e], tt_dev[s:e],
                _ed, _el, _et, _rn, _rd, _rr, device)
            raw_chunks.append(model.extract_fact_features(ct).float())
        z_facts = torch.cat(raw_chunks)   # [n_train, feat_dim]
        feat_dim = z_facts.shape[1]

        cnt = torch.zeros(n_ent, device=device)
        cnt.scatter_add_(0, th_dev, torch.ones(n_train, device=device))
        cnt.scatter_add_(0, tt_dev, torch.ones(n_train, device=device))
        inv_cnt = (1.0 / cnt.clamp(min=1)).unsqueeze(1)   # [n_ent, 1]

        def _entity_mean(z):
            ema = torch.zeros(n_ent, feat_dim, device=device)
            ema.scatter_add_(0, th_dev.unsqueeze(1).expand(-1, feat_dim), z)
            ema.scatter_add_(0, tt_dev.unsqueeze(1).expand(-1, feat_dim), z)
            return ema * inv_cnt

        # entity_layer_means[k] = per-entity mean of z^(k) over training facts
        entity_layer_means = [_entity_mean(z_facts)]  # k=0: raw CNN mean

        for k in range(_agg_mod.num_layers):
            e_mean = entity_layer_means[k]            # [n_ent, feat_dim]
            # Approximate neighbourhood for each training fact at layer k
            h_mean = e_mean[th_dev].float()           # [n_train, feat_dim]
            t_mean = e_mean[tt_dev].float()
            # Paper Eq 9-10: z_{k+1} = z_k + tanh(h_mean + t_mean)
            agg_neigh = torch.tanh(h_mean + t_mean)
            z_facts = z_facts + agg_neigh
            entity_layer_means.append(_entity_mean(z_facts))

    print(f"  Multi-pass context ready: {_agg_mod.num_layers} layers, "
          f"entity table {entity_layer_means[-1].shape}")

    def score_facts(heads, rels, tails):
        """
        Score each candidate (h, r, t_cand) using the mean-field approximation
        of the K-layer aggregator (Paper Equations 9-11).

          z^{(0)} = relation-specific CNN features
          for k in 0..K-1:
              z += tanh(entity_mean[h]^{(k)} + entity_mean[t_cand]^{(k)})
          score = MLP(z)

        OOK entities have mean=0 vector (identity update).
        """
        SUB_BATCH = 8192
        all_scores = []
        for start in range(0, len(heads), SUB_BATCH):
            h_b = heads[start:start + SUB_BATCH]
            r_b = rels [start:start + SUB_BATCH]
            t_b = tails[start:start + SUB_BATCH]
            with torch.no_grad():
                tensors = build_batch_from_precomputed(
                    h_b, r_b, t_b, _ed, _el, _et, _rn, _rd, _rr, device)
                z = model.extract_fact_features(tensors).float()   # z^(0): relation-specific CNN
                for k in range(_agg_mod.num_layers):
                    e_mean_k  = entity_layer_means[k]
                    # Paper Eq 9-10: z += tanh(mean_h + mean_t)
                    z = z + torch.tanh(e_mean_k[h_b].float() + e_mean_k[t_b].float())
                scores = model(z)
            all_scores.append(scores)
        return torch.cat(all_scores)

    print("Running Evaluation Step...")

    if not is_test:
        # ── Validation: bidirectional filtered MRR ──────────────────────────
        metrics = scorer.evaluate_ranking(
            predict_fn=score_facts,
            test_triples=eval_int,
            num_entities=len(ent2id),
            batch_size=512,
            k_values=[1, 3, 10],
            verbose=True,
            filter_tails=filter_tails,
            filter_heads=filter_heads,
        )
        if report_filename:
            scorer.export_report("IKGE DBPedia50k+ Model", filename=report_filename)
        return metrics['mrr']

    # ── Test: paper-exact 4-group evaluation ────────────────────────────────
    # Classify each test triple by (h∈train, r∈train, t∈train).
    # Category label: O=in-KG (True), X=out-of-KG (False)  →  (h_in, r_in, t_in)
    oot   = []   # O-O-X  (h∈in, r∈in, t∉in)
    xoo   = []   # X-O-O  (h∉in, r∈in, t∈in)
    oxo   = []   # O-X-O  (h∈in, r∉in, t∈in)
    oxx   = []   # O-X-X  (h∈in, r∉in, t∉in)
    xxo   = []   # X-X-O  (h∉in, r∉in, t∈in)
    # O-O-O triples are not used in any paper group

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
        # (True, True, True) → O-O-O: skip

    print(f"  Test triple breakdown:")
    print(f"    O-O-X (tail out):          {len(oot):>5}  → groups 1,3,4")
    print(f"    X-O-O (head out):          {len(xoo):>5}  → groups 2,3,4")
    print(f"    O-X-O (rel out):           {len(oxo):>5}  → groups 1,2")
    print(f"    O-X-X (rel+tail out):      {len(oxx):>5}  → group 1")
    print(f"    X-X-O (head+rel out):      {len(xxo):>5}  → group 2")
    print(f"    O-O-O (all in, not used):  {len(eval_int)-len(oot)-len(xoo)-len(oxo)-len(oxx)-len(xxo):>5}")

    group_results = scorer.evaluate_ikge_groups(
        predict_fn       = score_facts,
        group1_triples   = oot + oxo + oxx,   # head prediction
        group2_triples   = xoo + oxo + xxo,   # tail prediction
        group3_oot_triples = oot,              # tail pred (tail is OOK)
        group3_xoo_triples = xoo,             # head pred (head is OOK)
        group4_triples   = oot + xoo,          # relation prediction
        num_entities     = len(ent2id),
        num_relations    = len(rel2id),
        batch_size       = 512,
        k_values         = [1, 3, 10],
        filter_tails     = filter_tails,
        filter_heads     = filter_heads,
        filter_rels      = filter_rels,
    )

    if report_filename:
        scorer.export_report("IKGE DBPedia50k+ Model", filename=report_filename)

    return group_results.get('overall', {}).get('mrr', 0.0)


def main(fraction: float = 1.0, run_name: str = ""):
    # -----------------------------------------------------------------------
    # Logging – every print also lands in a timestamped log file
    # -----------------------------------------------------------------------
    ts        = time.strftime("%Y%m%d_%H%M%S")
    tag       = f"_{run_name}" if run_name else ""
    log_dir   = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path  = str(log_dir / f"train_{ts}{tag}.log")
    logger    = TeeLogger(log_path)
    sys.stdout = logger
    print(f"Logging to: {log_path}")

    try:
        _main(fraction=fraction, ts=ts)
    finally:
        logger.close()


def _main(fraction: float = 1.0, ts: str = ""):
    if not ts:
        ts = time.strftime("%Y%m%d_%H%M%S")
    print("="*80)
    print("Initializing IKGE DBPedia50k+ Pipeline")
    print("="*80)

    # -----------------------------------------------------------------------
    # Config  (aligned with paper: fact_emb_dim=256, max_desc_len=50, epochs>=200)
    # -----------------------------------------------------------------------
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Enable TF32 Tensor Core matmuls on Ampere/Ada/Blackwell GPUs (free throughput gain)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    word_emb_dim     = 300
    fact_emb_dim     = 256
    conv_channels    = 128
    num_layers       = 3     # K=3 aggregation hops for DBPedia50k+ (paper Section 6)
    dropout          = 0.25  # paper: dropout rate 0.25
    max_desc_len     = 50
    epochs           = 200
    eval_every       = 50    # run validation every N epochs
    # margin not used (BCE loss); kept for reference
    train_batch_size = 256   # mini-batch size for shuffled SGD (paper Section 6)
    print(f"Using device: {device}")

    # Output directory anchored to the script's own location
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(exist_ok=True, parents=True)

    # -----------------------------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------------------------
    data_dir = get_dataset_dir(dataset_dir='/workspace/ikge/ikge/data/DBPedia50k+')

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
    rel2domain = {x[0]: _norm_type(x[1]) for x in rel2constraint_raw if len(x) == 3}
    rel2range  = {x[0]: _norm_type(x[2]) for x in rel2constraint_raw if len(x) == 3}

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
        | set(rel2domain.values())
        | set(rel2range.values())
    )
    type2idx  = {t: i for i, t in enumerate(all_types)}
    num_types = len(type2idx)
    print(f"Entities: {len(ent2id)} | Relations: {len(rel2id)} | Types: {num_types}")

    # -----------------------------------------------------------------------
    # 4. GloVe embeddings
    # -----------------------------------------------------------------------
    descriptions = list(entity2desc.values()) + list(relation2name.values())
    embedding_matrix, word2idx, _ = setup_glove_for_ikge(
        entity_descriptions=descriptions,
        output_dir=str(output_dir / 'embeddings'),
        glove_version='6B',
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
    ent_desc, ent_len, ent_type = precompute_entity_tensors(
        all_entities_sorted, entity2desc, entity2types, type2idx, word2idx,
        max_desc_len, num_types
    )
    rel_name_t, rel_domain_t, rel_range_t = precompute_relation_tensors(
        all_relations_sorted, relation2name, rel2domain, rel2range,
        type2idx, word2idx, num_types
    )

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

    # Sorted list of entity IDs that appear in the training graph.
    # Used for in-KG negative sampling (prevents structural shortcut
    # where OOK negatives have empty subgraphs vs rich subgraphs for positives).
    in_kg_ents = sorted(entity_to_facts.keys())

    # Keep pos_tensors_cached for parameter passing to evaluate_model
    pos_tensors_cached = build_batch_from_precomputed(
        pos_h_ids, pos_r_ids, pos_t_ids,
        ent_desc, ent_len, ent_type,
        rel_name_t, rel_domain_t, rel_range_t,
        device
    )
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

    # torch.compile gives a free ~20-30% JIT speedup (no algorithmic change).
    if device.type == 'cuda':
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as e:
            print(f"torch.compile not available ({e}); continuing without it.")

    # AdamW lr=0.01 per paper. Word embeddings are frozen so requires_grad=False
    # params are excluded to avoid unnecessary optimizer state.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-2, weight_decay=1e-3)
    # Paper: cosine annealing LR scheduler (steps once per epoch).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # BF16 AMP: uses Tensor Cores on RTX 30/40/50-series without loss scaling.
    # BF16 has identical dynamic range to FP32 so training trajectory is unchanged.
    use_amp = device.type == 'cuda'

    metadata = (entity2desc, relation2name, entity2types, rel2domain, rel2range)

    # -----------------------------------------------------------------------
    # 9. Training loop  — paper-faithful per-batch subgraph aggregation
    #
    #   Each mini-batch:
    #     1. For every positive (h,r,t) and its negative (h',r,t'):
    #        a. BFS K-hop neighborhood in the training LINE GRAPH.
    #        b. Append a 'virtual query node' for this triple.
    #        c. Stack all subgraphs into a disjoint-union graph.
    #     2. Single extract_fact_features call on all nodes.
    #     3. Single aggregator call on the disjoint-union graph.
    #     4. Index out virtual-node features for pos and neg.
    #     5. Binary cross-entropy loss (paper Eq 13) and backprop through EVERYTHING.
    #
    #   Both pos and neg go through the SAME aggregation with CURRENT parameters.
    #   No stale features; no distribution shortcut.  The scoring MLP is forced
    #   to learn content-based discrimination, which also generalises at eval time.
    # -----------------------------------------------------------------------
    print("\nStarting Training Loop...")
    best_mrr         = 0.0
    window_best_loss = float('inf')   # best loss in the current eval window
    n_train          = len(train_triples)

    weights_path_mrr = str(output_dir / f"ikge_best_mrr_{ts}.pt")
    report_path      = str(output_dir / f"ikge_evaluation_report_{ts}.pdf")

    diag_pos: list = []   # score-diagnostic accumulators (reset every 10 epochs)
    diag_neg: list = []

    for epoch in range(epochs):
        model.train()
        perm       = torch.randperm(n_train)
        mb_indices = [perm[i:i + train_batch_size]
                      for i in range(0, n_train, train_batch_size)]
        epoch_loss = 0.0

        for bidx in mb_indices:
            optimizer.zero_grad()
            bh = pos_h_ids[bidx]; br = pos_r_ids[bidx]; bt = pos_t_ids[bidx]
            neg_h, neg_t = generate_neg_indices(bh, br, bt, positive_set, in_kg_ents)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                feat_tens, edge_idx, pq_idx, nq_idx = build_training_batch(
                    bh, br, bt, neg_h, neg_t,
                    entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
                    ent_desc, ent_len, ent_type,
                    rel_name_t, rel_domain_t, rel_range_t,
                    device, K=num_layers, max_neighbor_facts=8)

                all_feat   = model.extract_fact_features(feat_tens)
                all_agg    = model.aggregator(all_feat, edge_idx)
                pos_scores = model(all_agg[pq_idx])
                neg_scores = model(all_agg[nq_idx])
                # Paper Eq 13: binary cross-entropy loss
                # y=1 for positive facts, y=0 for negative facts
                ones  = torch.ones_like(pos_scores)
                zeros = torch.zeros_like(neg_scores)
                loss  = (F.binary_cross_entropy_with_logits(pos_scores, ones)
                         + F.binary_cross_entropy_with_logits(neg_scores, zeros))

            loss.backward()

            # Accumulate score stats for diagnostics (float32, detached)
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    ps = pos_scores.float().detach()
                    ns = neg_scores.float().detach()
                    diag_pos.append((ps.mean().item(), ps.std().item()))
                    diag_neg.append((ns.mean().item(), ns.std().item()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        loss_for_log = epoch_loss / len(mb_indices)
        scheduler.step()  # cosine annealing advances once per epoch

        # Print score-separation diagnostics every 10 epochs
        if (epoch + 1) % 10 == 0 and diag_pos:
            avg_pos = np.mean([m for m, _ in diag_pos])
            avg_neg = np.mean([m for m, _ in diag_neg])
            std_pos = np.mean([s for _, s in diag_pos])
            margin_gap = avg_pos - avg_neg
            print(f"  [Score diag] pos={avg_pos:.4f}±{std_pos:.4f}  neg={avg_neg:.4f}  "
                  f"gap={margin_gap:+.4f}  (gap>0 = model learning)")
            diag_pos, diag_neg = [], []   # reset for next window

        # Track best loss in this eval window
        if loss_for_log < window_best_loss:
            window_best_loss = loss_for_log

        # -- Periodic validation (unconditional – always run to catch MRR gains) --
        is_final_epoch = (epoch + 1 == epochs)
        if (epoch + 1) % eval_every == 0 and not is_final_epoch:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch+1}/{epochs}] Running validation "
                  f"(window best loss {window_best_loss:.4f})...")
            window_best_loss = float('inf')   # reset window

            model.eval()
            mrr = evaluate_model(
                model, val_triples[:], metadata, word2idx, type2idx,
                max_desc_len, device,
                ent2id, rel2id, id2ent, id2rel,
                fact_edge_index, train_triples,
                ent_desc, ent_len, ent_type,
                rel_name_t, rel_domain_t, rel_range_t,
                pos_tensors_cached,
                all_triples_for_filter=all_triples,
                train_ent_set=train_ent_set,
                train_rel_set=train_rel_set,
                is_test=False,
                report_filename=None
            )
            model.train()
            if mrr > best_mrr:
                best_mrr = mrr
                torch.save(model.state_dict(), weights_path_mrr)
                print(f"  Saved best MRR weights! MRR: {mrr:.4f} → {weights_path_mrr}")
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss_for_log:.4f} "
                  f"| Val MRR: {mrr:.4f} | LR: {lr_now:.2e}")
        else:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss_for_log:.4f}")

    # -----------------------------------------------------------------------
    # 10. Final evaluation on the full test set using the best checkpoint
    # -----------------------------------------------------------------------
    # Prefer best-MRR weights for final eval; fall back to best-loss weights.
    print(f"\nLoading best weights for final test evaluation ({Path(weights_path_mrr).name})...")
    model.load_state_dict(torch.load(weights_path_mrr, map_location=device, weights_only=True))

    print("Running Final Test Evaluation on full test set...")
    test_mrr = evaluate_model(
        model, test_triples, metadata, word2idx, type2idx,
        max_desc_len, device,
        ent2id, rel2id, id2ent, id2rel,
        fact_edge_index, train_triples,
        ent_desc, ent_len, ent_type,
        rel_name_t, rel_domain_t, rel_range_t,
        pos_tensors_cached,
        all_triples_for_filter=all_triples,
        train_ent_set=train_ent_set,
        train_rel_set=train_rel_set,
        is_test=True,
        report_filename=report_path
    )

    print("="*80)
    print("Training Complete")
    print(f"Best Validation MRR : {best_mrr:.4f}")
    print(f"Final Test MRR      : {test_mrr:.4f}")
    print(f"Report generated    : {report_path}")
    print(f"MRR weights         : {weights_path_mrr}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IKGE on DBPedia50k+")
    parser.add_argument(
        "--fraction", type=float, default=1.0,
        help="Fraction of training triples to use, e.g. 0.1 for 10%% (default: 1.0)"
    )
    parser.add_argument(
        "--run-name", type=str, default="",
        help="Optional label appended to the log filename, e.g. 'debug' or 'frac10'"
    )
    args = parser.parse_args()
    main(fraction=args.fraction, run_name=args.run_name)
