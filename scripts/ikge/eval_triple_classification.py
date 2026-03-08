"""eval_triple_classification.py
==============================
Standalone triple-classification evaluator for the IKGE FB20k+ model.

Usage
-----
  python eval_triple_classification.py
  python eval_triple_classification.py --weights ikge_w2v_best_mrr_XYZ.pt
  python eval_triple_classification.py --seed 7 --batch-size 256 --tau 0.42

What it does
------------
1. Loads the FB20k+ data and the latest (or specified) checkpoint.
2. Loads the frozen word vocab from fb20k_word2idx.pkl.
3. Pre-tokenises all entities and relations (same as eval_fb20k_sampled.py).
4. Builds type-constrained negative-sampling buckets from training facts.
5. Generates (or loads from cache) one hard negative per positive triple for
   both the VALIDATION set and each of the six TEST groups.
   Negatives are saved to TSV files named
       fb20k_triclf_neg_val_seed{S}.tsv
       fb20k_triclf_neg_{group}_seed{S}.tsv
   so results are reproducible.
6. CALIBRATION: scores val positives + val negatives, sweeps τ ∈ [0,1] in
   steps of 0.01, picks the threshold that maximises F1.
7. EVALUATION: scores test positives + test negatives per group, applies τ,
   reports Accuracy / Precision / Recall / F1 per group and overall.
8. Logs everything to logs/fb20k_triclf_{timestamp}.log

No line-graph or AFA aggregation is used — only the raw CNN fact features
(model.extract_fact_features → model.forward).  This is equivalent to using
the zero-hop (local) representation without any neighbourhood context; it is
the cleanest test of whether the text encoder alone can classify triples.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve script directory and make local imports work
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_ikge_w2v import (
    IKGENetwork,
    TeeLogger,
    get_dataset_dir,
    setup_w2v_for_ikge,
    precompute_entity_tensors,
    precompute_relation_tensors,
    build_batch_from_precomputed,
)
from download_w2v import create_embedding_matrix_w2v, build_vocabulary_from_descriptions

# ---------------------------------------------------------------------------
# Constants — must mirror train_ikge_w2v.py hyper-parameters
# ---------------------------------------------------------------------------
_WORD2IDX_CACHE = SCRIPT_DIR / "fb20k_word2idx.pkl"

word_emb_dim   = 300
fact_emb_dim   = 300
conv_channels  = 300
num_layers     = 2
dropout        = 0.1
max_desc_len   = 50

BATCH_SIZE_SCORE = 512   # triples per forward pass during scoring
TAU_SWEEP_STEPS  = 100   # τ sweep resolution (0 to 1 in 1/steps)
NEG_SAMPLE_TRIES = 200   # max attempts per negative sample


# ===========================================================================
# Helpers
# ===========================================================================

def _latest_fb20k_checkpoint() -> str:
    pts = sorted(
        SCRIPT_DIR.glob("fb20k_ikge_w2v_best_mrr_*.pt"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not pts:
        raise FileNotFoundError(
            f"No fb20k_ikge_w2v_best_mrr_*.pt found in {SCRIPT_DIR}"
        )
    return str(pts[0])


def load_txt(path: str) -> list[list[str]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if parts:
                rows.append(parts)
    return rows


def load_triples(path: str) -> list[tuple[str, str, str]]:
    return [(r[0], r[1], r[2]) for r in load_txt(path) if len(r) >= 3]


def _norm_type(t: str) -> str:
    return "dbo:Thing" if "owl#Thing" in t else t


def _neg_cache_path(group: str, seed: int) -> Path:
    return SCRIPT_DIR / f"fb20k_triclf_neg_{group}_seed{seed}.tsv"


def _save_negs(path: Path, negs: list[tuple[str, str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for h, r, t in negs:
            fh.write(f"{h}\t{r}\t{t}\n")
    print(f"  Saved {len(negs):,} negatives → {path.name}")


def _load_negs(path: Path) -> list[tuple[str, str, str]]:
    return load_triples(str(path))


# ===========================================================================
# Type-constrained negative generation
# ===========================================================================

def _build_type_buckets(
    in_kg_ents: list[int],
    ent_type: torch.Tensor,          # [n_ents, num_types]
    rel_domain_t: torch.Tensor,      # [n_rels, num_types]
    rel_range_t: torch.Tensor,       # [n_rels, num_types]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Build per-relation type-constrained entity buckets (CPU, same as training)."""
    et_cpu = ent_type.cpu()
    rd_cpu = rel_domain_t.cpu()
    rr_cpu = rel_range_t.cpu()
    in_kg_tensor = torch.tensor(in_kg_ents, dtype=torch.long)
    et_in_kg = et_cpu[in_kg_tensor]   # [n_in_kg, num_types]

    n_rels = rr_cpu.size(0)
    rel_tail_type_ents: dict[int, list[int]] = {}
    rel_head_type_ents: dict[int, list[int]] = {}

    for r in range(n_rels):
        # tail (range constraint)
        range_mask = rr_cpu[r]
        if range_mask.sum() > 0:
            match = ((et_in_kg * range_mask.unsqueeze(0)).sum(1) > 0).nonzero(as_tuple=True)[0]
            bucket = [in_kg_ents[idx.item()] for idx in match]
            if len(bucket) >= 5:
                rel_tail_type_ents[r] = bucket
        # head (domain constraint)
        domain_mask = rd_cpu[r]
        if domain_mask.sum() > 0:
            match = ((et_in_kg * domain_mask.unsqueeze(0)).sum(1) > 0).nonzero(as_tuple=True)[0]
            bucket = [in_kg_ents[idx.item()] for idx in match]
            if len(bucket) >= 5:
                rel_head_type_ents[r] = bucket

    print(
        f"  Type buckets: {len(rel_tail_type_ents)}/{n_rels} rels have tail buckets, "
        f"{len(rel_head_type_ents)}/{n_rels} rels have head buckets."
    )
    return rel_tail_type_ents, rel_head_type_ents


def _generate_negatives(
    positives: list[tuple[str, str, str]],
    positive_set: set[tuple[str, str, str]],
    ent2id: dict[str, int],
    rel2id: dict[str, int],
    id2ent: dict[int, str],
    in_kg_ents: list[int],
    rel_tail_type_ents: dict[int, list[int]],
    rel_head_type_ents: dict[int, list[int]],
    rng: random.Random,
) -> list[tuple[str, str, str]]:
    """Generate one hard negative per positive triple (type-constrained, in-KG)."""
    negs: list[tuple[str, str, str]] = []
    num_kg = len(in_kg_ents)

    for h_str, r_str, t_str in positives:
        # Skip triples whose entities/relation are outside our vocab
        if h_str not in ent2id or r_str not in rel2id or t_str not in ent2id:
            negs.append((h_str, r_str, t_str))   # placeholder: keep same triple
            continue

        h_id = ent2id[h_str]
        r_id = rel2id[r_str]
        t_id = ent2id[t_str]

        corrupt_head = rng.random() < 0.5

        if corrupt_head:
            bucket = rel_head_type_ents.get(r_id)
            found = False
            if bucket:
                for _ in range(NEG_SAMPLE_TRIES):
                    c = rng.choice(bucket)
                    if (id2ent[c], r_str, t_str) not in positive_set:
                        negs.append((id2ent[c], r_str, t_str))
                        found = True
                        break
            if not found:
                for _ in range(NEG_SAMPLE_TRIES):
                    c = in_kg_ents[rng.randint(0, num_kg - 1)]
                    if (id2ent[c], r_str, t_str) not in positive_set:
                        negs.append((id2ent[c], r_str, t_str))
                        found = True
                        break
            if not found:
                negs.append((h_str, r_str, t_str))   # fallback: same triple
        else:
            bucket = rel_tail_type_ents.get(r_id)
            found = False
            if bucket:
                for _ in range(NEG_SAMPLE_TRIES):
                    c = rng.choice(bucket)
                    if (h_str, r_str, id2ent[c]) not in positive_set:
                        negs.append((h_str, r_str, id2ent[c]))
                        found = True
                        break
            if not found:
                for _ in range(NEG_SAMPLE_TRIES):
                    c = in_kg_ents[rng.randint(0, num_kg - 1)]
                    if (h_str, r_str, id2ent[c]) not in positive_set:
                        negs.append((h_str, r_str, id2ent[c]))
                        found = True
                        break
            if not found:
                negs.append((h_str, r_str, t_str))   # fallback: same triple

    return negs


# ===========================================================================
# Scoring
# ===========================================================================

@torch.no_grad()
def _score_triples(
    triples: list[tuple[str, str, str]],
    model: IKGENetwork,
    ent2id: dict[str, int],
    rel2id: dict[str, int],
    ent_desc: torch.Tensor,
    ent_len: torch.Tensor,
    ent_type: torch.Tensor,
    ent_names: torch.Tensor,
    rel_name_t: torch.Tensor,
    rel_domain_t: torch.Tensor,
    rel_range_t: torch.Tensor,
    rel_domain_words_t: torch.Tensor,
    rel_range_words_t: torch.Tensor,
    device: torch.device,
    batch_size: int = BATCH_SIZE_SCORE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score all triples.  Returns (scores, valid_mask) where valid_mask is True
    for triples whose h/r/t are all in ent2id/rel2id.  Invalid triples get
    score=0; they should be excluded from metrics.
    """
    n = len(triples)
    scores = np.zeros(n, dtype=np.float32)
    valid  = np.zeros(n, dtype=bool)

    # Collect valid indices first
    h_ids_all, r_ids_all, t_ids_all, valid_idxs = [], [], [], []
    for idx, (h, r, t) in enumerate(triples):
        if h in ent2id and r in rel2id and t in ent2id:
            h_ids_all.append(ent2id[h])
            r_ids_all.append(rel2id[r])
            t_ids_all.append(ent2id[t])
            valid_idxs.append(idx)

    if not valid_idxs:
        return scores, valid

    for batch_start in range(0, len(valid_idxs), batch_size):
        batch_end = batch_start + batch_size
        bv = valid_idxs[batch_start:batch_end]
        h_t = torch.tensor(h_ids_all[batch_start:batch_end], dtype=torch.long)
        r_t = torch.tensor(r_ids_all[batch_start:batch_end], dtype=torch.long)
        t_t = torch.tensor(t_ids_all[batch_start:batch_end], dtype=torch.long)

        feat = build_batch_from_precomputed(
            h_t, r_t, t_t,
            ent_desc, ent_len, ent_type, ent_names,
            rel_name_t, rel_domain_t, rel_range_t,
            rel_domain_words_t, rel_range_words_t,
            device,
        )
        z = model.extract_fact_features(feat).float()
        s = model(z).cpu().numpy()   # sigmoid scores, shape [B]

        for local_i, global_i in enumerate(bv):
            scores[global_i] = s[local_i]
            valid[global_i]  = True

    return scores, valid


# ===========================================================================
# Threshold calibration
# ===========================================================================

def _calibrate_threshold(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    steps: int = TAU_SWEEP_STEPS,
) -> tuple[float, float, dict]:
    """Sweep τ ∈ [0, 1], return (best_tau, best_f1, metrics_at_best_tau)."""
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    scores = np.concatenate([pos_scores, neg_scores])

    best_tau = 0.5
    best_f1  = -1.0
    best_metrics: dict = {}

    for step in range(steps + 1):
        tau = step / steps
        preds = (scores >= tau).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        acc   = (tp + tn) / len(labels)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = tau
            best_metrics = dict(tau=tau, acc=acc, prec=prec, rec=rec, f1=f1,
                                tp=tp, fp=fp, tn=tn, fn=fn)

    return best_tau, best_f1, best_metrics


# ===========================================================================
# Per-group evaluation
# ===========================================================================

def _evaluate_group(
    pos_scores: np.ndarray,
    pos_valid: np.ndarray,
    neg_scores: np.ndarray,
    neg_valid: np.ndarray,
    tau: float,
) -> dict:
    """Binary classification metrics for one group given threshold τ."""
    # Only keep triples where BOTH the positive and negative are valid
    mask = pos_valid & neg_valid
    ps = pos_scores[mask]
    ns = neg_scores[mask]

    labels = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    scores = np.concatenate([ps, ns])
    preds  = (scores >= tau).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    return dict(
        n=int(mask.sum()),
        acc=acc, prec=prec, rec=rec, f1=f1,
        tp=tp, fp=fp, tn=tn, fn=fn,
        pos_mean=float(ps.mean()) if len(ps) else float("nan"),
        neg_mean=float(ns.mean()) if len(ns) else float("nan"),
    )


# ===========================================================================
# Main evaluation entry point
# ===========================================================================

def run_triple_classification(
    weights_path: str = "",
    seed: int = 42,
    batch_size: int = BATCH_SIZE_SCORE,
    tau_override: float = -1.0,
    run_name: str = "",
) -> None:
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    log_path = SCRIPT_DIR / "logs" / f"fb20k_triclf_{ts}{suffix}.log"
    log_path.parent.mkdir(exist_ok=True)
    sys.stdout = TeeLogger(str(log_path))

    print("=" * 80)
    print(f"IKGE FB20k+ Triple Classification  |  {ts}")
    print(f"  seed={seed}  batch_size={batch_size}  tau_override={tau_override!r}")
    print("=" * 80)

    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    data_dir   = get_dataset_dir(dataset_dir="/workspace/data/FB20k+")
    output_dir = SCRIPT_DIR.parent / "ikge"   # where embeddings/ lives

    print("\nLoading triples and metadata...")
    train_triples = load_triples(os.path.join(data_dir, "train.txt"))
    val_triples   = load_triples(os.path.join(data_dir, "valid.txt"))

    group_files = {
        "in_KG":  "test.txt",
        "out_T":  "test_out_T.txt",
        "out_H":  "test_out_H.txt",
        "out_R":  "test_out_R.txt",
        "out_RT": "test_out_RT.txt",
        "out_HR": "test_out_HR.txt",
    }
    test_groups: dict[str, list[tuple[str, str, str]]] = {}
    for grp, fname in group_files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            test_groups[grp] = load_triples(fpath)
        else:
            test_groups[grp] = []
            print(f"  WARNING: {fname} not found — group '{grp}' will be empty")

    train_ent_set = set(t[0] for t in train_triples) | set(t[2] for t in train_triples)
    train_rel_set = set(t[1] for t in train_triples)

    print(f"  Train triples  : {len(train_triples):,}")
    print(f"  Val triples    : {len(val_triples):,}")
    for grp, triples in test_groups.items():
        print(f"  Test [{grp:6s}]  : {len(triples):,}")

    entity2desc  = {r[0]: r[1] for r in load_txt(os.path.join(data_dir, "entity2text.txt"))
                    if len(r) >= 2}
    entity2types: dict[str, list[str]] = defaultdict(list)
    for row in load_txt(os.path.join(data_dir, "entity2type.txt")):
        if len(row) >= 2:
            entity2types[row[0]].append(_norm_type(row[1]))

    rel2domain: dict[str, list[str]] = defaultdict(list)
    rel2range:  dict[str, list[str]] = defaultdict(list)
    for x in load_txt(os.path.join(data_dir, "relation2constraint.txt")):
        if len(x) == 3:
            rel2domain[x[0]].append(_norm_type(x[1]))
            rel2range[x[0]].append(_norm_type(x[2]))

    # -----------------------------------------------------------------------
    # 2. Build entity/relation id maps  (same scope as eval_fb20k_sampled.py)
    # -----------------------------------------------------------------------
    vocab_triples        = train_triples + val_triples + test_groups.get("in_KG", [])
    vocab_entities_sorted = sorted(
        set(t[0] for t in vocab_triples) | set(t[2] for t in vocab_triples)
    )
    vocab_relations_sorted = sorted(set(t[1] for t in vocab_triples))

    all_test_flat  = [t for g in test_groups.values() for t in g]
    all_triples    = train_triples + val_triples + all_test_flat

    all_entities_sorted  = sorted(
        set(t[0] for t in all_triples) | set(t[2] for t in all_triples)
    )
    all_relations_sorted = sorted(set(t[1] for t in all_triples))

    ent2id = {e: i for i, e in enumerate(all_entities_sorted)}
    rel2id = {r: i for i, r in enumerate(all_relations_sorted)}
    id2ent = {i: e for e, i in ent2id.items()}
    id2rel = {i: r for r, i in rel2id.items()}

    relation2name = {
        r: r.split("/")[-1].split("#")[-1].replace("_", " ")
        for r in all_relations_sorted
    }

    all_types = sorted(
        set(typ for types in entity2types.values() for typ in types)
        | set(t for ts in rel2domain.values() for t in ts)
        | set(t for ts in rel2range.values() for t in ts)
    )
    type2idx  = {t: i for i, t in enumerate(all_types)}
    num_types = len(type2idx)
    print(f"\nEntities: {len(ent2id):,} | Relations: {len(rel2id):,} | Types: {num_types:,}")

    # -----------------------------------------------------------------------
    # 3. Vocab / embedding matrix
    # -----------------------------------------------------------------------
    if _WORD2IDX_CACHE.exists():
        print(f"\nLoading saved word2idx from {_WORD2IDX_CACHE.name}...")
        with open(_WORD2IDX_CACHE, "rb") as _f:
            _cache = pickle.load(_f)
        word2idx = _cache["word2idx"]

        print("Loading Wikipedia2Vec model for embedding matrix...")
        try:
            from wikipedia2vec import Wikipedia2Vec
            pkl_path = str(SCRIPT_DIR / "embeddings" / "enwiki_20180420_300d.pkl")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                wiki_model = Wikipedia2Vec.load(pkl_path)
            embedding_matrix, _ = create_embedding_matrix_w2v(word2idx, wiki_model, word_emb_dim)
            del wiki_model
        except Exception as _e:
            print(f"  W2V load failed ({_e}), falling back to setup_w2v_for_ikge...")
            vocab_ens = [e.split("/")[-1].replace("_", " ") for e in vocab_entities_sorted]
            vocab_rn  = {r: r.split("/")[-1].split("#")[-1].replace("_", " ")
                         for r in vocab_relations_sorted}
            vocab_tns = [t.split("/")[-1].split("#")[-1].replace("_", " ") for t in all_types]
            descriptions = (list(entity2desc.values()) + list(vocab_rn.values())
                            + vocab_ens + vocab_tns)
            embedding_matrix, word2idx, _ = setup_w2v_for_ikge(
                entity_descriptions=descriptions,
                output_dir=str(SCRIPT_DIR / "embeddings"),
                embedding_dim=word_emb_dim)
    else:
        print(f"\nNo saved vocab at {_WORD2IDX_CACHE.name}; rebuilding (old entity-name style)...")
        vocab_ens = [e.split("/")[-1].replace("_", " ") for e in vocab_entities_sorted]
        vocab_rn  = {r: r.split("/")[-1].split("#")[-1].replace("_", " ")
                     for r in vocab_relations_sorted}
        vocab_tns = [t.split("/")[-1].split("#")[-1].replace("_", " ") for t in all_types]
        descriptions = (list(entity2desc.values()) + list(vocab_rn.values())
                        + vocab_ens + vocab_tns)
        embedding_matrix, word2idx, _ = setup_w2v_for_ikge(
            entity_descriptions=descriptions,
            output_dir=str(SCRIPT_DIR / "embeddings"),
            embedding_dim=word_emb_dim)

    # -----------------------------------------------------------------------
    # 4. Pre-tokenise entities and relations
    # -----------------------------------------------------------------------
    print("\nPre-tokenizing entities and relations...")
    ent_desc, ent_len, ent_type, ent_names = precompute_entity_tensors(
        all_entities_sorted, entity2desc, entity2types, type2idx,
        word2idx, max_desc_len, num_types,
    )
    (rel_name_t, rel_domain_t, rel_range_t,
     rel_domain_words_t, rel_range_words_t) = precompute_relation_tensors(
        all_relations_sorted, relation2name, rel2domain, rel2range,
        type2idx, word2idx, num_types,
    )

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
    print("  Done.")

    # -----------------------------------------------------------------------
    # 5. Load model
    # -----------------------------------------------------------------------
    num_ents = len(all_entities_sorted)
    model = IKGENetwork(
        embedding_matrix=embedding_matrix,
        word_emb_dim=word_emb_dim,
        fact_emb_dim=fact_emb_dim,
        conv_channels=conv_channels,
        num_types=num_types,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    if not weights_path:
        weights_path = _latest_fb20k_checkpoint()
    elif not os.path.isabs(weights_path):
        candidate = SCRIPT_DIR / weights_path
        weights_path = str(candidate) if candidate.exists() else weights_path

    print(f"\nLoading weights: {weights_path}")
    state   = torch.load(weights_path, map_location=device, weights_only=True)
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    print("Weights loaded OK.")

    # -----------------------------------------------------------------------
    # 6. Build in-KG entity list and type-constrained buckets
    # -----------------------------------------------------------------------
    print("\nBuilding in-KG entity set and type buckets...")
    positive_set_train = set(train_triples)
    # positive_set for negative filtering = all known true facts
    positive_set_all = set(train_triples) | set(val_triples) | set(all_test_flat)

    entity_to_facts: dict[int, list[int]] = {}
    for idx, (h, r, t) in enumerate(train_triples):
        if h in ent2id and r in rel2id and t in ent2id:
            for e_id in (ent2id[h], ent2id[t]):
                entity_to_facts.setdefault(e_id, []).append(idx)
    in_kg_ents = sorted(entity_to_facts.keys())

    rel_tail_type_ents, rel_head_type_ents = _build_type_buckets(
        in_kg_ents, ent_type, rel_domain_t, rel_range_t,
    )

    # -----------------------------------------------------------------------
    # 7. Generate / load negatives  (validation + each test group)
    # -----------------------------------------------------------------------
    print("\nPreparing negatives (type-constrained, in-KG, hard negatives)...")

    def _get_negs(group_name: str, positives: list[tuple[str, str, str]]) \
            -> list[tuple[str, str, str]]:
        cache = _neg_cache_path(group_name, seed)
        if cache.exists():
            negs = _load_negs(cache)
            if len(negs) == len(positives):
                print(f"  Loaded cached negatives for '{group_name}' ({len(negs):,})")
                return negs
            else:
                print(f"  Cache size mismatch for '{group_name}' — regenerating...")
        negs = _generate_negatives(
            positives, positive_set_all, ent2id, rel2id, id2ent,
            in_kg_ents, rel_tail_type_ents, rel_head_type_ents, rng,
        )
        _save_negs(cache, negs)
        return negs

    val_negs = _get_negs("val", val_triples)
    test_negs: dict[str, list[tuple[str, str, str]]] = {
        grp: _get_negs(grp, triples)
        for grp, triples in test_groups.items()
        if triples
    }

    # -----------------------------------------------------------------------
    # 8. Calibrate threshold on validation set
    # -----------------------------------------------------------------------
    print("\nScoring validation set for threshold calibration...")
    torch.cuda.empty_cache()
    val_pos_scores, val_pos_valid = _score_triples(
        val_triples, model, ent2id, rel2id,
        ent_desc, ent_len, ent_type, ent_names,
        rel_name_t, rel_domain_t, rel_range_t,
        rel_domain_words_t, rel_range_words_t,
        device, batch_size,
    )
    val_neg_scores, val_neg_valid = _score_triples(
        val_negs, model, ent2id, rel2id,
        ent_desc, ent_len, ent_type, ent_names,
        rel_name_t, rel_domain_t, rel_range_t,
        rel_domain_words_t, rel_range_words_t,
        device, batch_size,
    )

    # Only use triples where both pos and neg are scoreable
    cal_mask = val_pos_valid & val_neg_valid
    cal_pos  = val_pos_scores[cal_mask]
    cal_neg  = val_neg_scores[cal_mask]
    print(f"  Val scoreable triples: {cal_mask.sum():,} / {len(val_triples):,}")
    print(f"  Val pos score mean: {cal_pos.mean():.4f}  neg mean: {cal_neg.mean():.4f}")

    if tau_override >= 0.0:
        tau = tau_override
        print(f"  Using user-specified tau = {tau:.4f} (skipping sweep)")
        best_val = _evaluate_group(val_pos_scores, val_pos_valid,
                                   val_neg_scores, val_neg_valid, tau)
        print(f"  Val Acc={best_val['acc']:.4f}  F1={best_val['f1']:.4f}")
    else:
        print(f"  Sweeping τ in {TAU_SWEEP_STEPS} steps...")
        tau, best_f1, best_val = _calibrate_threshold(cal_pos, cal_neg, TAU_SWEEP_STEPS)
        print(f"\n  Best threshold  : τ = {tau:.4f}")
        print(f"  Val Accuracy    : {best_val['acc']:.4f}")
        print(f"  Val Precision   : {best_val['prec']:.4f}")
        print(f"  Val Recall      : {best_val['rec']:.4f}")
        print(f"  Val F1          : {best_val['f1']:.4f}")
        print(f"  Val TP/FP/TN/FN : {best_val['tp']}/{best_val['fp']}/{best_val['tn']}/{best_val['fn']}")

    # -----------------------------------------------------------------------
    # 9. Evaluate on each test group
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"Test evaluation  (τ = {tau:.4f})")
    print(f"{'='*80}")

    group_results: dict[str, dict] = {}
    for grp, pos_triples in test_groups.items():
        if not pos_triples or grp not in test_negs:
            print(f"  Skipping {grp} (no triples)")
            continue

        print(f"\n  Scoring {grp} ({len(pos_triples):,} triples)...")
        torch.cuda.empty_cache()
        pos_sc, pos_v = _score_triples(
            pos_triples, model, ent2id, rel2id,
            ent_desc, ent_len, ent_type, ent_names,
            rel_name_t, rel_domain_t, rel_range_t,
            rel_domain_words_t, rel_range_words_t,
            device, batch_size,
        )
        neg_sc, neg_v = _score_triples(
            test_negs[grp], model, ent2id, rel2id,
            ent_desc, ent_len, ent_type, ent_names,
            rel_name_t, rel_domain_t, rel_range_t,
            rel_domain_words_t, rel_range_words_t,
            device, batch_size,
        )
        m = _evaluate_group(pos_sc, pos_v, neg_sc, neg_v, tau)
        group_results[grp] = m
        print(f"    n={m['n']:6,}  Acc={m['acc']:.4f}  P={m['prec']:.4f}  "
              f"R={m['rec']:.4f}  F1={m['f1']:.4f}  "
              f"pos_mean={m['pos_mean']:.4f}  neg_mean={m['neg_mean']:.4f}")

    # -----------------------------------------------------------------------
    # 10. Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"TRIPLE CLASSIFICATION SUMMARY  (τ = {tau:.4f})")
    print(f"{'='*80}")
    hdr = f"  {'Group':8s}  {'N':>7s}  {'Acc':>7s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    total_n = total_tp = total_fp = total_tn = total_fn = 0
    for grp in ["in_KG", "out_T", "out_H", "out_R", "out_RT", "out_HR"]:
        m = group_results.get(grp)
        if m is None:
            print(f"  {grp:8s}  {'—':>7s}  {'—':>7s}  {'—':>7s}  {'—':>7s}  {'—':>7s}")
            continue
        print(f"  {grp:8s}  {m['n']:7,}  {m['acc']:7.4f}  {m['prec']:7.4f}  "
              f"{m['rec']:7.4f}  {m['f1']:7.4f}")
        total_n  += m['n']
        total_tp += m['tp']
        total_fp += m['fp']
        total_tn += m['tn']
        total_fn += m['fn']

    # Macro averages
    valid_groups = [group_results[g] for g in group_results]
    macro_acc  = np.mean([m['acc']  for m in valid_groups]) if valid_groups else 0.0
    macro_prec = np.mean([m['prec'] for m in valid_groups]) if valid_groups else 0.0
    macro_rec  = np.mean([m['rec']  for m in valid_groups]) if valid_groups else 0.0
    macro_f1   = np.mean([m['f1']   for m in valid_groups]) if valid_groups else 0.0

    # Micro averages
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)
                  if (micro_prec + micro_rec) > 0 else 0.0)
    micro_acc  = (total_tp + total_tn) / (total_n * 2) if total_n > 0 else 0.0

    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'MACRO':8s}  {total_n:7,}  {macro_acc:7.4f}  {macro_prec:7.4f}  "
          f"{macro_rec:7.4f}  {macro_f1:7.4f}")
    print(f"  {'MICRO':8s}  {total_n:7,}  {micro_acc:7.4f}  {micro_prec:7.4f}  "
          f"{micro_rec:7.4f}  {micro_f1:7.4f}")
    print("=" * 80)
    print(f"  Calibrated threshold τ = {tau:.4f}")
    print(f"  Val F1 at τ            = {best_val['f1']:.4f}")
    print(f"  Val Acc at τ           = {best_val['acc']:.4f}")
    print(f"  Weights                : {weights_path}")
    print(f"  Log                    : {log_path}")
    print("=" * 80)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triple classification evaluation for IKGE on FB20k+")
    parser.add_argument(
        "--weights", default="",
        help="Path to .pt checkpoint. Omit to auto-use latest fb20k_ikge_w2v_best_mrr_*.pt")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for negative sampling (default: 42)")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE_SCORE,
        help=f"Forward-pass batch size (default: {BATCH_SIZE_SCORE})")
    parser.add_argument(
        "--tau", type=float, default=-1.0,
        help="Fixed threshold τ ∈ [0,1]. Omit (or pass -1) to calibrate on val set.")
    parser.add_argument(
        "--run-name", default="",
        help="Optional label appended to the log filename.")
    args = parser.parse_args()
    run_triple_classification(
        weights_path=args.weights,
        seed=args.seed,
        batch_size=args.batch_size,
        tau_override=args.tau,
        run_name=args.run_name,
    )
