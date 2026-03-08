"""
Standalone evaluation script for IKGE.
Loads saved weights and regenerates the full test report
without retraining.

Usage:
    python3 eval_from_checkpoint.py --weights ikge_best_mrr_20260302_043039.pt
    python3 eval_from_checkpoint.py --weights ikge_best_mrr_20260302_043039.pt --run-name recheck
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
from train_ikge import (
    IKGENetwork,
    TeeLogger,
    evaluate_model,
    get_dataset_dir,
    setup_glove_for_ikge,
    create_line_graph,
    precompute_entity_tensors,
    precompute_relation_tensors,
    build_batch_from_precomputed,
)


def run_eval(weights_path: str, run_name: str = ""):
    ts      = time.strftime("%Y%m%d_%H%M%S")
    tag     = f"_{run_name}" if run_name else ""
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = str(log_dir / f"eval_{ts}{tag}.log")
    logger   = TeeLogger(log_path)
    sys.stdout = logger
    print(f"Logging to: {log_path}")

    try:
        _run_eval(weights_path)
    finally:
        logger.close()


def _run_eval(weights_path: str):
    print("=" * 80)
    print("IKGE Evaluation from Checkpoint")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device}")

    # ── Hyperparams (must match training) ────────────────────────────────────
    word_emb_dim  = 300
    fact_emb_dim  = 256
    conv_channels = 128
    num_layers    = 3
    dropout       = 0.25
    max_desc_len  = 50

    output_dir = Path(__file__).resolve().parent
    data_dir   = get_dataset_dir(dataset_dir='/workspace/ikge/ikge/data/DBPedia50k+')

    def load_txt(path):
        with open(path, 'r') as f:
            return [line.strip().split('\t') for line in f if line.strip()]

    train_triples = load_txt(os.path.join(data_dir, 'train.txt'))
    val_triples   = load_txt(os.path.join(data_dir, 'valid.txt'))
    test_triples  = load_txt(os.path.join(data_dir, 'test.txt'))

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
    def _norm_type(t):
        return 'dbo:Thing' if 'owl#Thing' in t else t
    rel2domain = {x[0]: _norm_type(x[1]) for x in rel2constraint_raw if len(x) == 3}
    rel2range  = {x[0]: _norm_type(x[2]) for x in rel2constraint_raw if len(x) == 3}

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
        | set(rel2domain.values())
        | set(rel2range.values())
    )
    type2idx  = {t: i for i, t in enumerate(all_types)}
    num_types = len(type2idx)
    print(f"Entities: {len(ent2id)} | Relations: {len(rel2id)} | Types: {num_types}")

    descriptions     = list(entity2desc.values()) + list(relation2name.values())
    embedding_matrix, word2idx, _ = setup_glove_for_ikge(
        entity_descriptions=descriptions,
        output_dir=str(output_dir / 'embeddings'),
        glove_version='6B',
        embedding_dim=word_emb_dim
    )

    print("\nBuilding Line Graph...")
    id_train_triples     = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in train_triples]
    train_triple_tensor  = torch.tensor(id_train_triples, dtype=torch.long)
    fact_edge_index, _   = create_line_graph(train_triple_tensor)
    fact_edge_index      = fact_edge_index.to(device)

    print("Pre-tokenizing entities and relations...")
    ent_desc, ent_len, ent_type = precompute_entity_tensors(
        all_entities_sorted, entity2desc, entity2types, type2idx, word2idx, max_desc_len, num_types
    )
    rel_name_t, rel_domain_t, rel_range_t = precompute_relation_tensors(
        all_relations_sorted, relation2name, rel2domain, rel2range, type2idx, word2idx, num_types
    )

    pos_h_ids = torch.tensor([ent2id[h] for h, r, t in train_triples], dtype=torch.long)
    pos_r_ids = torch.tensor([rel2id[r] for h, r, t in train_triples], dtype=torch.long)
    pos_t_ids = torch.tensor([ent2id[t] for h, r, t in train_triples], dtype=torch.long)

    pos_tensors_cached = build_batch_from_precomputed(
        pos_h_ids, pos_r_ids, pos_t_ids,
        ent_desc, ent_len, ent_type,
        rel_name_t, rel_domain_t, rel_range_t,
        device
    )

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

    # Load checkpoint (compiled or plain)
    weights_path = str(Path(__file__).resolve().parent / weights_path) \
        if not os.path.isabs(weights_path) else weights_path
    print(f"\nLoading weights from: {weights_path}")
    state = torch.load(weights_path, map_location=device, weights_only=True)
    # Strip _orig_mod. prefix if checkpoint was saved from torch.compile model
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    print("Weights loaded successfully.")

    metadata   = (entity2desc, relation2name, entity2types, rel2domain, rel2range)
    ts         = time.strftime("%Y%m%d_%H%M%S")
    report_path = str(output_dir / f"ikge_evaluation_report_{ts}.pdf")

    print("\nRunning Final Test Evaluation on full test set...")
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

    print("=" * 80)
    print(f"Test MRR   : {test_mrr:.4f}")
    print(f"Report     : {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IKGE from saved checkpoint")
    parser.add_argument("--weights", required=True,
                        help="Path to .pt weights file (absolute or relative to script dir)")
    parser.add_argument("--run-name", type=str, default="",
                        help="Optional label appended to the log filename")
    args = parser.parse_args()
    run_eval(args.weights, args.run_name)
