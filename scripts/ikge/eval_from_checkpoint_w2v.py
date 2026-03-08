"""
Standalone evaluation script for IKGE  (Wikipedia2Vec variant).
Loads saved weights produced by train_ikge_w2v.py and runs the full
GPU-vectorised paper-exact 4-group test evaluation.

Usage:
    # Latest checkpoint (auto-detected):
    python3 eval_from_checkpoint_w2v.py

    # Specific checkpoint:
    python3 eval_from_checkpoint_w2v.py --weights ikge_w2v_best_mrr_20260302_210719.pt

    # With optional log label:
    python3 eval_from_checkpoint_w2v.py --weights <file>.pt --run-name recheck
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

# ── import helpers from the main training module ─────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ikge_w2v import (
    IKGENetwork,
    TeeLogger,
    evaluate_model,
    get_dataset_dir,
    setup_w2v_for_ikge,
    create_line_graph,
    precompute_entity_tensors,
    precompute_relation_tensors,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _latest_checkpoint() -> str:
    """Return the most recently modified .pt file in the script directory."""
    pts = sorted(SCRIPT_DIR.glob("ikge_w2v_best_mrr_*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(
            f"No ikge_w2v_best_mrr_*.pt checkpoint found in {SCRIPT_DIR}")
    print(f"Auto-detected latest checkpoint: {pts[0].name}")
    return str(pts[0])


def run_eval(weights_path: str = "", run_name: str = ""):
    ts       = time.strftime("%Y%m%d_%H%M%S")
    tag      = f"_{run_name}" if run_name else ""
    log_dir  = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = str(log_dir / f"eval_{ts}{tag}.log")
    logger   = TeeLogger(log_path)
    sys.stdout = logger
    print(f"Logging to: {log_path}")

    try:
        _run_eval(weights_path, ts)
    finally:
        logger.close()


def _run_eval(weights_path: str, ts: str):
    print("=" * 80)
    print("IKGE Evaluation from Checkpoint  (GPU-vectorised full-ranking)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device}")

    # ── Hyperparams (must match training) ────────────────────────────────────
    word_emb_dim  = 300
    fact_emb_dim  = 300   # paper: single d=300 throughout (Section 5.2.4)
    conv_channels = 300   # must equal word_emb_dim
    num_layers    = 2     # K=2 matches training
    dropout       = 0.1   # matches training
    max_desc_len  = 50

    output_dir = SCRIPT_DIR
    data_dir   = get_dataset_dir(dataset_dir='/workspace/data/DBPedia50k+')

    def load_txt(path):
        with open(path, 'r') as f:
            return [line.strip().split('\t') for line in f if line.strip()]

    print("Loading dataset...")
    train_triples = load_txt(os.path.join(data_dir, 'train.txt'))
    val_triples   = load_txt(os.path.join(data_dir, 'valid.txt'))
    test_triples  = load_txt(os.path.join(data_dir, 'test.txt'))
    print(f"  train: {len(train_triples):,}  val: {len(val_triples):,}  "
          f"test: {len(test_triples):,}")

    train_ent_set = set(t[0] for t in train_triples) | set(t[2] for t in train_triples)
    train_rel_set = set(t[1] for t in train_triples)
    print(f"In-KG entities: {len(train_ent_set):,}  |  In-KG relations: {len(train_rel_set):,}")

    entity2desc_raw = load_txt(os.path.join(data_dir, 'entity2text.txt'))
    entity2desc     = {x[0]: x[1] for x in entity2desc_raw if len(x) == 2}

    entity2types = defaultdict(list)
    for x in load_txt(os.path.join(data_dir, 'entity2type.txt')):
        if len(x) == 2:
            typ = 'dbo:Thing' if 'owl#Thing' in x[1] else x[1]
            entity2types[x[0]].append(typ)

    rel2constraint_raw = load_txt(os.path.join(data_dir, 'relation2constraint.txt'))
    def _norm(t): return 'dbo:Thing' if 'owl#Thing' in t else t
    rel2domain = defaultdict(list)
    rel2range  = defaultdict(list)
    for x in rel2constraint_raw:
        if len(x) == 3:
            rel2domain[x[0]].append(_norm(x[1]))
            rel2range[x[0]].append(_norm(x[2]))

    all_triples          = train_triples + val_triples + test_triples
    all_entities_sorted  = sorted(set(t[0] for t in all_triples) | set(t[2] for t in all_triples))
    all_relations_sorted = sorted(set(t[1] for t in all_triples))

    ent2id = {e: i for i, e in enumerate(all_entities_sorted)}
    rel2id = {r: i for i, r in enumerate(all_relations_sorted)}
    id2ent = {i: e for e, i in ent2id.items()}
    id2rel = {i: r for r, i in rel2id.items()}

    relation2name = {
        r: r.split('/')[-1].split('#')[-1].replace('_', ' ')
        for r in all_relations_sorted
    }

    all_types = sorted(
        set(typ for types in entity2types.values() for typ in types)
        | set(t for ts in rel2domain.values() for t in ts)
        | set(t for ts in rel2range.values() for t in ts)
    )
    type2idx  = {t: i for i, t in enumerate(all_types)}
    num_types = len(type2idx)
    print(f"Entities: {len(ent2id)} | Relations: {len(rel2id)} | Types: {num_types}")

    entity_name_strings = [
        e.split('/')[-1].replace('_', ' ') for e in all_entities_sorted
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

    print("\nBuilding Line Graph...")
    id_train_triples    = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in train_triples]
    train_triple_tensor = torch.tensor(id_train_triples, dtype=torch.long)
    fact_edge_index, _  = create_line_graph(train_triple_tensor)
    fact_edge_index     = fact_edge_index.to(device)

    print("Pre-tokenizing entities and relations...")
    ent_desc, ent_len, ent_type, ent_names = precompute_entity_tensors(
        all_entities_sorted, entity2desc, entity2types, type2idx, word2idx, max_desc_len, num_types
    )
    rel_name_t, rel_domain_t, rel_range_t, rel_domain_words_t, rel_range_words_t = precompute_relation_tensors(
        all_relations_sorted, relation2name, rel2domain, rel2range, type2idx, word2idx, num_types
    )

    # Move all lookup tables to GPU once (mirrors train_ikge_w2v._main)
    ent_desc         = ent_desc.to(device)
    ent_len          = ent_len.to(device)
    ent_type         = ent_type.to(device)
    ent_names        = ent_names.to(device)
    rel_name_t       = rel_name_t.to(device)
    rel_domain_t     = rel_domain_t.to(device)
    rel_range_t      = rel_range_t.to(device)
    rel_domain_words_t = rel_domain_words_t.to(device)
    rel_range_words_t  = rel_range_words_t.to(device)

    pos_h_ids = torch.tensor([ent2id[h] for h, r, t in train_triples], dtype=torch.long)
    pos_r_ids = torch.tensor([rel2id[r] for h, r, t in train_triples], dtype=torch.long)
    pos_t_ids = torch.tensor([ent2id[t] for h, r, t in train_triples], dtype=torch.long)

    # ── Build model ──────────────────────────────────────────────────────────
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

    # Resolve path: absolute, relative to cwd, or relative to script dir
    if not weights_path:
        weights_path = _latest_checkpoint()
    elif not os.path.isabs(weights_path):
        candidate = SCRIPT_DIR / weights_path
        weights_path = str(candidate) if candidate.exists() else weights_path

    print(f"\nLoading weights: {weights_path}")
    state   = torch.load(weights_path, map_location=device, weights_only=True)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    print("Weights loaded OK.")

    metadata    = (entity2desc, relation2name, entity2types, rel2domain, rel2range)
    report_path = str(output_dir / f"ikge_w2v_evaluation_report_{ts}.pdf")

    print("\nRunning full-ranking test evaluation...")
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

    print("=" * 80)
    print(f"Test MRR (overall) : {test_mrr:.4f}")
    print(f"Weights            : {weights_path}")
    print(f"PDF report         : {report_path}")
    print(f"Log                : {sys.stdout.log_path if hasattr(sys.stdout, 'log_path') else 'N/A'}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate IKGE from a saved checkpoint (GPU-vectorised full ranking)")
    parser.add_argument(
        "--weights", default="",
        help="Path to .pt file (absolute, relative to cwd, or filename in script dir). "
             "Omit to auto-use the latest checkpoint.")
    parser.add_argument(
        "--run-name", default="",
        help="Optional label appended to the log filename.")
    args = parser.parse_args()
    run_eval(args.weights, args.run_name)
