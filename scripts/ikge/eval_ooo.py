"""
Sanity-check evaluation on O-O-O test triples only (all-in-KG / closed-world).

WHY: The full 4-group eval gives MRR ~0.10 against ~1200 T2 candidates.
     O-O-O triples all hit T1 (avg 1.2 candidates), so if the model works
     at all MRR here should be very high (>0.8).
     If it ISN'T, the scoring function itself is broken.
     If it IS, the MRR gap in G1/G2 is entirely explained by candidate set size.

Reports:
  1. Hinge loss on O-O-O triples  (same fixed-neg formula as training)
  2. MRR/H@k against T1 tail candidates  (pair_tail_cands, avg ~1.2)
  3. MRR/H@k against T1 head candidates  (pair_head_cands, avg ~2.2)
  4. MRR/H@k against ALL n_ents (~30k) — the "hard" oracle comparison

Usage:
    python3 eval_ooo.py                        # auto latest checkpoint
    python3 eval_ooo.py --weights <file>.pt
"""

import argparse
import os
import sys
import time
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ikge_w2v import (
    IKGENetwork,
    TeeLogger,
    generate_neg_indices,
    validate_loss,
    get_dataset_dir,
    setup_w2v_for_ikge,
    create_line_graph,
    precompute_entity_tensors,
    precompute_relation_tensors,
    build_batch_from_precomputed,
)

SCRIPT_DIR = Path(__file__).resolve().parent
NEG_INF = float('-inf')


def _latest_checkpoint() -> str:
    pts = sorted(SCRIPT_DIR.glob("ikge_w2v_best_mrr_*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(f"No ikge_w2v_best_mrr_*.pt found in {SCRIPT_DIR}")
    print(f"Auto-detected checkpoint: {pts[0].name}")
    return str(pts[0])


def _mrr_stats(ranks: list) -> dict:
    a = np.array(ranks, dtype=float)
    return dict(n=len(a), mrr=float(np.mean(1.0/a)),
                h1=float(np.mean(a<=1)), h3=float(np.mean(a<=3)),
                h10=float(np.mean(a<=10)), mean_rank=float(np.mean(a)))


def _print_stats(label: str, s: dict):
    print(f"  {label:40s}  n={s['n']:5d}  MRR={s['mrr']:.4f}  "
          f"H@1={s['h1']:.4f}  H@3={s['h3']:.4f}  H@10={s['h10']:.4f}  "
          f"MeanRank={s['mean_rank']:.1f}")


def run(weights_path: str = ""):
    ts      = time.strftime("%Y%m%d_%H%M%S")
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = str(log_dir / f"eval_ooo_{ts}.log")
    logger   = TeeLogger(log_path)
    sys.stdout = logger
    print(f"Logging to: {log_path}")
    try:
        _run(weights_path, ts)
    finally:
        logger.close()


def _run(weights_path: str, ts: str):
    print("=" * 72)
    print("O-O-O Closed-World Sanity-Check Evaluation")
    print("=" * 72)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device}")

    # ── Hyperparams — must match training ────────────────────────────────────
    word_emb_dim  = 300
    fact_emb_dim  = 300
    conv_channels = 300
    num_layers    = 2
    dropout       = 0.1
    max_desc_len  = 50
    MARGIN        = 0.5

    data_dir = get_dataset_dir(dataset_dir='/workspace/data/DBPedia50k+')

    def load_txt(p):
        with open(p) as f:
            return [ln.strip().split('\t') for ln in f if ln.strip()]

    print("Loading dataset...")
    train_triples = load_txt(os.path.join(data_dir, 'train.txt'))
    val_triples   = load_txt(os.path.join(data_dir, 'valid.txt'))
    test_triples  = load_txt(os.path.join(data_dir, 'test.txt'))

    train_ent_set = set(t[0] for t in train_triples) | set(t[2] for t in train_triples)
    train_rel_set = set(t[1] for t in train_triples)

    # O-O-O: head, relation, tail all in training KG
    ooo_triples = [t for t in test_triples
                   if t[0] in train_ent_set and t[1] in train_rel_set and t[2] in train_ent_set]
    print(f"  Total test triples: {len(test_triples):,}")
    print(f"  O-O-O (all in-KG) : {len(ooo_triples):,}")
    if not ooo_triples:
        print("No O-O-O triples found — nothing to evaluate.")
        return

    entity2desc_raw = load_txt(os.path.join(data_dir, 'entity2text.txt'))
    entity2desc     = {x[0]: x[1] for x in entity2desc_raw if len(x) == 2}

    entity2types = defaultdict(list)
    for x in load_txt(os.path.join(data_dir, 'entity2type.txt')):
        if len(x) == 2:
            t_ = 'dbo:Thing' if 'owl#Thing' in x[1] else x[1]
            entity2types[x[0]].append(t_)

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

    relation2name = {r: r.split('/')[-1].replace('_', ' ') for r in all_relations_sorted}

    all_types = sorted(
        set(tp for ts in entity2types.values() for tp in ts)
        | set(t for ts in rel2domain.values() for t in ts)
        | set(t for ts in rel2range.values() for t in ts))
    type2idx  = {t: i for i, t in enumerate(all_types)}
    num_types = len(type2idx)

    entity_name_strings = [e.split('/')[-1].replace('_', ' ') for e in all_entities_sorted]
    type_name_strings   = [t.split('/')[-1].replace('_', ' ') for t in all_types]
    descriptions = (list(entity2desc.values()) + list(relation2name.values())
                    + entity_name_strings + type_name_strings)

    output_dir = SCRIPT_DIR
    embedding_matrix, word2idx, _ = setup_w2v_for_ikge(
        entity_descriptions=descriptions,
        output_dir=str(output_dir / 'embeddings'),
        embedding_dim=word_emb_dim
    )

    print("Building line graph...")
    id_train = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in train_triples]
    fact_edge_index, _ = create_line_graph(
        torch.tensor(id_train, dtype=torch.long))
    fact_edge_index = fact_edge_index.to(device)

    print("Pre-tokenizing...")
    ent_desc, ent_len, ent_type, ent_names = precompute_entity_tensors(
        all_entities_sorted, entity2desc, entity2types, type2idx, word2idx, max_desc_len, num_types)
    rel_name_t, rel_domain_t, rel_range_t, rel_domain_words_t, rel_range_words_t = \
        precompute_relation_tensors(
            all_relations_sorted, relation2name, rel2domain, rel2range, type2idx, word2idx, num_types)

    for t_ in [ent_desc, ent_len, ent_type, ent_names,
               rel_name_t, rel_domain_t, rel_range_t, rel_domain_words_t, rel_range_words_t]:
        t_.to(device)  # in-place for non-floating, .to returns new for float — re-assign
    ent_desc          = ent_desc.to(device)
    ent_len           = ent_len.to(device)
    ent_type          = ent_type.to(device)
    ent_names         = ent_names.to(device)
    rel_name_t        = rel_name_t.to(device)
    rel_domain_t      = rel_domain_t.to(device)
    rel_range_t       = rel_range_t.to(device)
    rel_domain_words_t = rel_domain_words_t.to(device)
    rel_range_words_t  = rel_range_words_t.to(device)

    # ── Build model ───────────────────────────────────────────────────────────
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

    if not weights_path:
        weights_path = _latest_checkpoint()
    elif not os.path.isabs(weights_path):
        candidate = SCRIPT_DIR / weights_path
        weights_path = str(candidate) if candidate.exists() else weights_path

    print(f"Loading weights: {weights_path}")
    state   = torch.load(weights_path, map_location=device, weights_only=True)
    cleaned = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    print("Weights loaded OK.\n")

    # ── Candidate lookup tables (T1 + T2 + global filter) ────────────────────
    pos_h = [ent2id[h] for h, r, t in train_triples]
    pos_r = [rel2id[r] for h, r, t in train_triples]
    pos_t = [ent2id[t] for h, r, t in train_triples]

    pair_tail_cands: dict = {}   # (h,r) -> sorted list of seen tails
    pair_head_cands: dict = {}   # (r,t) -> sorted list of seen heads
    filter_tails:   dict = {}   # (h,r) -> all correct tails (train+val+test)
    filter_heads:   dict = {}   # (r,t) -> all correct heads

    for h_i, r_i, t_i in zip(pos_h, pos_r, pos_t):
        pair_tail_cands.setdefault((h_i, r_i), set()).add(t_i)
        pair_head_cands.setdefault((r_i, t_i), set()).add(h_i)

    pair_tail_cands = {k: sorted(v) for k, v in pair_tail_cands.items()}
    pair_head_cands = {k: sorted(v) for k, v in pair_head_cands.items()}

    for h_s, r_s, t_s in all_triples:
        if h_s in ent2id and r_s in rel2id and t_s in ent2id:
            h_i, r_i, t_i = ent2id[h_s], rel2id[r_s], ent2id[t_s]
            filter_tails.setdefault((h_i, r_i), []).append(t_i)
            filter_heads.setdefault((r_i, t_i), []).append(h_i)

    in_kg_ents = torch.tensor(
        [ent2id[e] for e in train_ent_set if e in ent2id], dtype=torch.long)

    positive_set = set((ent2id[h], rel2id[r], ent2id[t])
                       for h, r, t in all_triples
                       if h in ent2id and r in rel2id and t in ent2id)

    pos_h_list = pos_h
    pos_r_list = pos_r
    pos_t_list = pos_t

    # entity -> list of training fact indices (for BFS subgraph in validate_loss)
    _e2f: dict = defaultdict(list)
    for fi, (h_i_t, t_i_t) in enumerate(zip(pos_h_list, pos_t_list)):
        _e2f[h_i_t].append(fi)
        _e2f[t_i_t].append(fi)
    entity_to_facts = dict(_e2f)

    # ── 1. Hinge loss on O-O-O triples ───────────────────────────────────────
    print("─" * 72)
    print("1. HINGE LOSS on O-O-O test triples (same formula as training)")
    print("─" * 72)

    ooo_int = [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in ooo_triples]
    _bh = torch.tensor([h for h, r, t in ooo_int], dtype=torch.long)
    _br = torch.tensor([r for h, r, t in ooo_int], dtype=torch.long)
    _bt = torch.tensor([t for h, r, t in ooo_int], dtype=torch.long)
    neg_h_fix, neg_t_fix = generate_neg_indices(_bh, _br, _bt, positive_set, in_kg_ents)
    fixed_ooo_negs = (neg_h_fix, neg_t_fix)

    ooo_loss, ooo_pos_mean, ooo_neg_mean = validate_loss(
        model, ooo_triples, positive_set,
        entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
        ent_desc, ent_len, ent_type, ent_names,
        rel_name_t, rel_domain_t, rel_range_t,
        rel_domain_words_t, rel_range_words_t,
        ent2id, rel2id, device,
        in_kg_ents=in_kg_ents,
        num_layers=num_layers,
        batch_size=256,
        max_neighbor_facts=32,
        margin=MARGIN,
        fixed_negs=fixed_ooo_negs,
    )
    print(f"  O-O-O hinge loss : {ooo_loss:.4f}")
    print(f"  pos score mean   : {ooo_pos_mean:.4f}")
    print(f"  neg score mean   : {ooo_neg_mean:.4f}")
    print(f"  score gap        : {ooo_pos_mean - ooo_neg_mean:+.4f}\n")

    # ── 2. Pre-compute CNN features for all training facts ────────────────────
    print("─" * 72)
    print("2. MRR RANKING on O-O-O triples")
    print("─" * 72)

    print("Pre-computing z_train_init...")
    CHUNK = 2048
    n_train_fac = len(pos_h_list)
    z_parts = []
    with torch.no_grad():
        for cs in range(0, n_train_fac, CHUNK):
            ce   = min(cs + CHUNK, n_train_fac)
            bh_t = torch.tensor(pos_h_list[cs:ce], dtype=torch.long, device=device)
            br_t = torch.tensor(pos_r_list[cs:ce], dtype=torch.long, device=device)
            bt_t = torch.tensor(pos_t_list[cs:ce], dtype=torch.long, device=device)
            feat = build_batch_from_precomputed(
                bh_t, br_t, bt_t,
                ent_desc, ent_len, ent_type, ent_names,
                rel_name_t, rel_domain_t, rel_range_t,
                rel_domain_words_t, rel_range_words_t, device)
            z_parts.append(model.extract_fact_features(feat).float().cpu())
    z_train = torch.cat(z_parts, dim=0)  # [n_train_fac, d]

    HALF = 16
    d    = z_train.shape[1]
    n_ents = len(ent2id)

    # Entity→training-fact neighbour table  (reuse entity_to_facts built above)
    ent_nbrs_cpu = torch.full((n_ents, HALF), n_train_fac, dtype=torch.long)
    for e, fids in entity_to_facts.items():
        k = min(HALF, len(fids))
        ent_nbrs_cpu[e, :k] = torch.tensor(fids[:k], dtype=torch.long)

    z_pad      = torch.cat([z_train.to(device), torch.zeros(1, d, device=device)], 0)
    ent_nbrs_g = ent_nbrs_cpu.to(device)
    nbr_feats  = z_pad[ent_nbrs_g]                                  # [n_ents, HALF, d]
    nbr_mask   = (ent_nbrs_cpu != n_train_fac).to(device)           # [n_ents, HALF]
    agg        = model.aggregator

    def score_candidates(h_i: int, r_i: int, cand_ids: list) -> torch.Tensor:
        """Score (h_i, r_i, cand) for each cand — returns [len(cand_ids)] tensor."""
        cands  = torch.tensor(cand_ids, dtype=torch.long, device=device)
        C      = len(cand_ids)

        with torch.no_grad():
            # CNN features for each (h_i, r_i, cand_t)
            bh_t = torch.full((C,), h_i, dtype=torch.long, device=device)
            br_t = torch.full((C,), r_i, dtype=torch.long, device=device)
            feat = build_batch_from_precomputed(
                bh_t, br_t, cands,
                ent_desc, ent_len, ent_type, ent_names,
                rel_name_t, rel_domain_t, rel_range_t,
                rel_domain_words_t, rel_range_words_t, device)
            z = model.extract_fact_features(feat).float()   # [C, d]

            fix_nf = nbr_feats[h_i].unsqueeze(0).expand(C, -1, -1)   # [C, HALF, d]
            fix_nm = nbr_mask[h_i].unsqueeze(0).expand(C, -1)        # [C, HALF]
            can_nf = nbr_feats[cands]                                  # [C, HALF, d]
            can_nm = nbr_mask[cands]                                   # [C, HALF]

            for layer in agg.attention_layers:
                W      = layer.weight
                z_proj = torch.matmul(z, W.T)                         # [C, d]
                z_p    = z_proj.unsqueeze(1)                           # [C, 1, d]
                a_fix  = (fix_nf * z_p).sum(-1)                       # [C, HALF]
                a_can  = (can_nf * z_p).sum(-1)                       # [C, HALF]
                a_fix.masked_fill_(~fix_nm, NEG_INF)
                a_can.masked_fill_(~can_nm, NEG_INF)
                att_w  = torch.softmax(torch.cat([a_fix, a_can], dim=1), dim=1)  # [C, 2*HALF]
                agg_f  = (att_w[:, :HALF].unsqueeze(-1) * fix_nf).sum(1)
                agg_c  = (att_w[:, HALF:].unsqueeze(-1) * can_nf).sum(1)
                z      = z + torch.tanh(agg_f + agg_c)

            return model(z).float().cpu()   # sigmoid scores [C]

    def score_head_candidates(t_i: int, r_i: int, cand_ids: list) -> torch.Tensor:
        """Score (cand_h, r_i, t_i) for each cand — returns [len(cand_ids)] tensor."""
        cands  = torch.tensor(cand_ids, dtype=torch.long, device=device)
        C      = len(cand_ids)

        with torch.no_grad():
            bt_t = torch.full((C,), t_i, dtype=torch.long, device=device)
            br_t = torch.full((C,), r_i, dtype=torch.long, device=device)
            feat = build_batch_from_precomputed(
                cands, br_t, bt_t,
                ent_desc, ent_len, ent_type, ent_names,
                rel_name_t, rel_domain_t, rel_range_t,
                rel_domain_words_t, rel_range_words_t, device)
            z = model.extract_fact_features(feat).float()

            fix_nf = nbr_feats[t_i].unsqueeze(0).expand(C, -1, -1)
            fix_nm = nbr_mask[t_i].unsqueeze(0).expand(C, -1)
            can_nf = nbr_feats[cands]
            can_nm = nbr_mask[cands]

            for layer in agg.attention_layers:
                W      = layer.weight
                z_proj = torch.matmul(z, W.T)
                z_p    = z_proj.unsqueeze(1)
                a_fix  = (fix_nf * z_p).sum(-1)
                a_can  = (can_nf * z_p).sum(-1)
                a_fix.masked_fill_(~fix_nm, NEG_INF)
                a_can.masked_fill_(~can_nm, NEG_INF)
                att_w  = torch.softmax(torch.cat([a_fix, a_can], dim=1), dim=1)
                agg_f  = (att_w[:, :HALF].unsqueeze(-1) * fix_nf).sum(1)
                agg_c  = (att_w[:, HALF:].unsqueeze(-1) * can_nf).sum(1)
                z      = z + torch.tanh(agg_f + agg_c)

            return model(z).float().cpu()

    all_ents = list(range(n_ents))

    t1_tail_ranks, t1_head_ranks = [], []
    full_tail_ranks, full_head_ranks = [], []
    t1_tail_sizes, t1_head_sizes = [], []

    print(f"Scoring {len(ooo_int)} O-O-O triples...")
    t0_rank = time.time()
    for idx, (h_i, r_i, t_i) in enumerate(ooo_int):
        if (idx + 1) % 200 == 0 or idx == len(ooo_int) - 1:
            elapsed = time.time() - t0_rank
            eta     = elapsed / (idx + 1) * (len(ooo_int) - idx - 1)
            print(f"  {idx+1}/{len(ooo_int)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

        # ── Tail prediction ───────────────────────────────────────────────────
        # T1 candidate set
        t1_tail = list(set(pair_tail_cands.get((h_i, r_i), [])) | {t_i})
        t1_tail_sizes.append(len(t1_tail))
        sc_t1_tail = score_candidates(h_i, r_i, t1_tail)
        known_tail = filter_tails.get((h_i, r_i), [])
        row = sc_t1_tail.clone()
        local = {e: i for i, e in enumerate(t1_tail)}
        for e in known_tail:
            if e != t_i and e in local:
                row[local[e]] = NEG_INF
        t1_tail_ranks.append(int((row > row[local[t_i]]).sum().item()) + 1)

        # Full ranking (all n_ents)
        sc_full_tail = score_candidates(h_i, r_i, all_ents)
        row_f = sc_full_tail.clone()
        for e in known_tail:
            if e != t_i:
                row_f[e] = NEG_INF
        full_tail_ranks.append(int((row_f > row_f[t_i]).sum().item()) + 1)

        # ── Head prediction ───────────────────────────────────────────────────
        t1_head = list(set(pair_head_cands.get((r_i, t_i), [])) | {h_i})
        t1_head_sizes.append(len(t1_head))
        sc_t1_head = score_head_candidates(t_i, r_i, t1_head)
        known_head = filter_heads.get((r_i, t_i), [])
        row_h = sc_t1_head.clone()
        local_h = {e: i for i, e in enumerate(t1_head)}
        for e in known_head:
            if e != h_i and e in local_h:
                row_h[local_h[e]] = NEG_INF
        t1_head_ranks.append(int((row_h > row_h[local_h[h_i]]).sum().item()) + 1)

        # Full head ranking
        sc_full_head = score_head_candidates(t_i, r_i, all_ents)
        row_fh = sc_full_head.clone()
        for e in known_head:
            if e != h_i:
                row_fh[e] = NEG_INF
        full_head_ranks.append(int((row_fh > row_fh[h_i]).sum().item()) + 1)

    total_time = time.time() - t0_rank
    print(f"\nRanking complete in {total_time:.1f}s")

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("RESULTS — O-O-O CLOSED-WORLD TRIPLES")
    print("=" * 72)
    print(f"\n  Hinge loss (fixed negs, same as training formula):")
    print(f"    O-O-O loss = {ooo_loss:.4f}  |  pos={ooo_pos_mean:.4f}  "
          f"neg={ooo_neg_mean:.4f}  gap={ooo_pos_mean - ooo_neg_mean:+.4f}")
    print(f"\n  Candidate set sizes:")
    print(f"    T1 tail cands: avg={np.mean(t1_tail_sizes):.2f}  "
          f"min={min(t1_tail_sizes)}  max={max(t1_tail_sizes)}")
    print(f"    T1 head cands: avg={np.mean(t1_head_sizes):.2f}  "
          f"min={min(t1_head_sizes)}  max={max(t1_head_sizes)}")

    print(f"\n  {'Metric':<44} {'n':>5}  {'MRR':>6}  {'H@1':>6}  {'H@3':>6}  "
          f"{'H@10':>6}  {'MeanRank':>10}")
    print("  " + "-" * 90)
    _print_stats("Tail T1 (pair_tail_cands, avg ~1.2)", _mrr_stats(t1_tail_ranks))
    _print_stats("Head T1 (pair_head_cands, avg ~2.2)", _mrr_stats(t1_head_ranks))
    _print_stats("Tail FULL (~30k, filtered)",          _mrr_stats(full_tail_ranks))
    _print_stats("Head FULL (~30k, filtered)",          _mrr_stats(full_head_ranks))
    print()
    print("  INTERPRETATION:")
    print("  - If T1 MRR >> 0.5  → scoring works, large cand sets explain MRR gap")
    print("  - If T1 MRR << 0.5  → scoring function or architecture is broken")
    print("  - Full MRR ≈ T2 MRR in main eval (~0.09/0.14)  → problem IS cand size")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="",
                        help=".pt file path (omit for latest auto-detected)")
    args = parser.parse_args()
    run(weights_path=args.weights)
