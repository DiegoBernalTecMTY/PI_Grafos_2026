"""
Sampled evaluation script for IKGE on FB20k+  (Wikipedia2Vec variant).

Loads FB20k+ weights produced by train_ikge_w2v.py and runs the full
GPU-vectorised paper-exact 4-group test evaluation on a STRATIFIED SAMPLE
of N_SAMPLE test triples — matching the DBPedia50k+ test set size (10,267)
so both datasets can be compared on equal footing.

Sampling is stratified across the 6 test-file groups:
    test.txt          → in-KG
    test_out_T.txt    → tail OOK
    test_out_H.txt    → head OOK
    test_out_R.txt    → relation OOK
    test_out_RT.txt   → relation + tail OOK
    test_out_HR.txt   → head + relation OOK

Usage:
    # Latest FB20k+ checkpoint (auto-detected):
    python3 eval_fb20k_sampled.py

    # Specific checkpoint:
    python3 eval_fb20k_sampled.py --weights fb20k_ikge_w2v_best_mrr_<ts>.pt

    # Fix the random seed for reproducibility:
    python3 eval_fb20k_sampled.py --seed 42

    # Override sample size:
    python3 eval_fb20k_sampled.py --n-sample 10267

    # Optional log label:
    python3 eval_fb20k_sampled.py --run-name quick_check
"""

import argparse
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ikge_w2v import (
    IKGENetwork,
    TeeLogger,
    evaluate_model,
    validate_loss,
    get_dataset_dir,
    setup_w2v_for_ikge,
    create_line_graph,
    precompute_entity_tensors,
    precompute_relation_tensors,
    build_batch_from_precomputed,
)
from download_w2v import create_embedding_matrix_w2v, build_vocabulary_from_descriptions

SCRIPT_DIR = Path(__file__).resolve().parent
_WORD2IDX_CACHE = SCRIPT_DIR / 'fb20k_word2idx.pkl'

# Match DBPedia50k+ test set size exactly:
#   3434 + 2919 + 122 + 153 + 452 + 1372 + 1815 = 10,267
DBPEDIA_TEST_SIZE = 10_267


def _latest_fb20k_checkpoint() -> str:
    pts = sorted(SCRIPT_DIR.glob("fb20k_ikge_w2v_best_mrr_*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(
            f"No fb20k_ikge_w2v_best_mrr_*.pt checkpoint found in {SCRIPT_DIR}")
    print(f"Auto-detected latest FB20k+ checkpoint: {pts[0].name}")
    return str(pts[0])


def _load_triples(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        rows = [ln.strip().split('\t') for ln in f if ln.strip()]
    return [r for r in rows if len(r) == 3]


def _stratified_sample(groups: dict, n_total: int, seed: int) -> list:
    """
    Stratified sample of n_total triples, drawn proportionally from each group.
    groups : {group_name: [triples]}
    Returns a flat list of (h, r, t) and prints the per-group breakdown.
    """
    rng = random.Random(seed)
    total_available = sum(len(v) for v in groups.values())
    n_sample = min(n_total, total_available)
    if n_sample < n_total:
        print(f"  WARNING: only {total_available:,} triples available; "
              f"sampling all of them (< requested {n_total:,}).")

    sampled = []
    remaining = n_sample
    # Sort groups so allocation is deterministic
    sorted_groups = sorted(groups.items(), key=lambda x: x[0])
    total_for_proportion = total_available

    print(f"\n  Stratified sampling ({n_sample:,} / {total_available:,} total):")
    for i, (name, triples) in enumerate(sorted_groups):
        is_last = (i == len(sorted_groups) - 1)
        if is_last:
            k = remaining
        else:
            proportion = len(triples) / total_for_proportion
            k = min(round(n_sample * proportion), len(triples), remaining)
        k = max(0, k)
        chosen = rng.sample(triples, k) if k < len(triples) else list(triples)
        sampled.extend(chosen)
        remaining -= len(chosen)
        print(f"    {name:20s}  avail={len(triples):6,}  sampled={len(chosen):5,}  "
              f"({100*len(chosen)/max(len(triples),1):.1f}%)")

    print(f"  Total sampled: {len(sampled):,}")
    return sampled


def run_eval(weights_path: str = "", run_name: str = "",
             n_sample: int = DBPEDIA_TEST_SIZE, seed: int = 42):
    ts       = time.strftime("%Y%m%d_%H%M%S")
    tag      = f"_{run_name}" if run_name else ""
    log_dir  = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = str(log_dir / f"fb20k_eval_{ts}{tag}.log")
    logger   = TeeLogger(log_path)
    sys.stdout = logger
    print(f"Logging to: {log_path}")
    try:
        _run_eval(weights_path, ts, n_sample, seed)
    finally:
        logger.close()


def _run_eval(weights_path: str, ts: str, n_sample: int, seed: int):
    print("=" * 80)
    print("IKGE FB20k+ Evaluation  —  Sampled (GPU-vectorised full-ranking)")
    print(f"  Sample size : {n_sample:,}  (DBPedia50k+ parity)")
    print(f"  Random seed : {seed}")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
    print(f"Device: {device}")

    # ── Hyperparams must match training ─────────────────────────────────────
    word_emb_dim  = 300
    fact_emb_dim  = 300
    conv_channels = 300
    num_layers    = 2
    dropout       = 0.1
    max_desc_len  = 50

    output_dir = SCRIPT_DIR
    data_dir   = get_dataset_dir(dataset_dir='/workspace/data/FB20k+')

    def load_txt(path):
        """Load any TSV file, returning all rows regardless of column count."""
        if not os.path.exists(path):
            print(f"  WARNING: missing file: {path}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            rows = [ln.strip().split('\t') for ln in f if ln.strip()]
        if not rows:
            print(f"  WARNING: empty file: {path}")
        return rows

    def load_triples(path):
        """Load a triple file (exactly 3 tab-separated columns)."""
        return [r for r in load_txt(path) if len(r) == 3]

    # ── Load train / val (needed for vocab, graph, filter) ───────────────────
    print("\nLoading dataset...")
    train_triples = load_triples(os.path.join(data_dir, 'train.txt'))
    val_triples   = load_triples(os.path.join(data_dir, 'valid.txt'))

    # ── Load all test groups ─────────────────────────────────────────────────
    test_groups = {
        'in_KG':  load_triples(os.path.join(data_dir, 'test.txt')),
        'out_T':  load_triples(os.path.join(data_dir, 'test_out_T.txt')),
        'out_H':  load_triples(os.path.join(data_dir, 'test_out_H.txt')),
        'out_R':  load_triples(os.path.join(data_dir, 'test_out_R.txt')),
        'out_RT': load_triples(os.path.join(data_dir, 'test_out_RT.txt')),
        'out_HR': load_triples(os.path.join(data_dir, 'test_out_HR.txt')),
    }
    total_test = sum(len(v) for v in test_groups.values())
    print(f"  train: {len(train_triples):,}  val: {len(val_triples):,}")
    print(f"  Test files loaded:")
    for gname, gtriples in sorted(test_groups.items()):
        print(f"    {gname:20s}: {len(gtriples):,}")
    print(f"  Total test available: {total_test:,}")

    # ── Stratified sample ────────────────────────────────────────────────────
    test_triples_sampled = _stratified_sample(test_groups, n_sample, seed)

    train_ent_set = set(t[0] for t in train_triples) | set(t[2] for t in train_triples)
    train_rel_set = set(t[1] for t in train_triples)
    print(f"\nIn-KG entities: {len(train_ent_set):,}  |  In-KG relations: {len(train_rel_set):,}")

    # ── Metadata ─────────────────────────────────────────────────────────────
    entity2desc = {}
    for x in load_txt(os.path.join(data_dir, 'entity2text.txt')):
        if len(x) == 2:
            entity2desc[x[0]] = x[1]

    # Normalize owl#Thing → dbo:Thing  (must match training exactly)
    def _norm_type(t: str) -> str:
        return 'dbo:Thing' if 'owl#Thing' in t else t

    entity2types = defaultdict(list)
    for x in load_txt(os.path.join(data_dir, 'entity2type.txt')):
        if len(x) == 2:
            entity2types[x[0]].append(_norm_type(x[1]))

    rel2domain = defaultdict(list)
    rel2range  = defaultdict(list)
    for x in load_txt(os.path.join(data_dir, 'relation2constraint.txt')):
        if len(x) == 3:
            rel2domain[x[0]].append(_norm_type(x[1]))
            rel2range[x[0]].append(_norm_type(x[2]))

    # ── Vocab-building triples: train + val + test.txt ONLY ──────────────────
    # This MUST match the training script exactly (which only loaded test.txt,
    # not the out-of-KG test files).  The word embedding matrix shape is fixed
    # by the vocabulary that was built during training, so we must reproduce
    # the exact same description corpus → same vocab → same embedding shape.
    vocab_triples = train_triples + val_triples + test_groups['in_KG']
    vocab_entities_sorted  = sorted(set(t[0] for t in vocab_triples)
                                    | set(t[2] for t in vocab_triples))
    vocab_relations_sorted = sorted(set(t[1] for t in vocab_triples))

    # ── Eval-scope entities: ALL 6 test groups ───────────────────────────────
    # ent2id must cover OOK entities so we can evaluate on them.  Their
    # description words are all inside entity2desc.values() which was already
    # included in the training vocab, so no word-embedding index goes OOB.
    all_test_flat = [t for g in test_groups.values() for t in g]
    all_triples   = train_triples + val_triples + all_test_flat

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
    print(f"Entities: {len(ent2id):,} | Relations: {len(rel2id):,} | Types: {num_types:,}")

    # ── Word2Vec / embedding matrix ──────────────────────────────────────────
    def _entity_name(e: str, desc: str = '') -> str:
        seg = e.split('/')[-1].split('#')[-1].replace('_', ' ')
        non_alpha = sum(1 for c in seg if not c.isalpha() and c != ' ')
        if seg and non_alpha > len(seg) * 0.4:
            words = desc.split()[:4]
            return ' '.join(words) if words else seg
        return seg

    # ── Vocab / embedding matrix ─────────────────────────────────────────────
    # Training built the vocab with the OLD entity-name style (raw URI segment,
    # no MID heuristic): e.split('/')[-1].replace('_',' ').  The MID-heuristic
    # fix was applied to train_ikge_w2v.py AFTER the training process started,
    # so the running process used the old code → vocab = 105,129 words.
    # We reproduce that exact vocab here (or load it from the pre-built pkl).
    if _WORD2IDX_CACHE.exists():
        print(f"Loading saved word2idx from {_WORD2IDX_CACHE.name}...")
        with open(_WORD2IDX_CACHE, 'rb') as _f:
            _cache = pickle.load(_f)
        word2idx = _cache['word2idx']
        # Build embedding matrix using the loaded vocab (W2V lookup only)
        # Must still load the W2V model — use setup_w2v_for_ikge with a minimal
        # single-description list and then rebuild matrix with the real word2idx.
        print("Loading Wikipedia2Vec model for embedding matrix...")
        try:
            from wikipedia2vec import Wikipedia2Vec
            import warnings
            pkl_path = str(output_dir / 'embeddings' / 'enwiki_20180420_300d.pkl')
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                wiki_model = Wikipedia2Vec.load(pkl_path)
            embedding_matrix, _ = create_embedding_matrix_w2v(word2idx, wiki_model, word_emb_dim)
            del wiki_model  # free 10 GB
        except Exception as _e:
            print(f"  W2V load failed ({_e}), falling back to setup_w2v_for_ikge...")
            # Fallback: rebuild vocab the old way and call setup
            vocab_ens = [e.split('/')[-1].replace('_', ' ') for e in vocab_entities_sorted]
            vocab_rn  = {r: r.split('/')[-1].split('#')[-1].replace('_', ' ')
                         for r in vocab_relations_sorted}
            vocab_tns = [t.split('/')[-1].split('#')[-1].replace('_', ' ') for t in all_types]
            descriptions = (list(entity2desc.values()) + list(vocab_rn.values())
                            + vocab_ens + vocab_tns)
            embedding_matrix, word2idx, _ = setup_w2v_for_ikge(
                entity_descriptions=descriptions,
                output_dir=str(output_dir / 'embeddings'),
                embedding_dim=word_emb_dim)
    else:
        print(f"No saved vocab found at {_WORD2IDX_CACHE.name}; rebuilding with "
              f"OLD entity-name style to match training checkpoint...")
        # OLD entity name computation (matching training's state at run time):
        vocab_ens = [e.split('/')[-1].replace('_', ' ') for e in vocab_entities_sorted]
        vocab_rn  = {r: r.split('/')[-1].split('#')[-1].replace('_', ' ')
                     for r in vocab_relations_sorted}
        vocab_tns = [t.split('/')[-1].split('#')[-1].replace('_', ' ') for t in all_types]
        descriptions = (list(entity2desc.values()) + list(vocab_rn.values())
                        + vocab_ens + vocab_tns)
        print(f"  Descriptions: {len(descriptions):,} (expect 40,396)")
        embedding_matrix, word2idx, _ = setup_w2v_for_ikge(
            entity_descriptions=descriptions,
            output_dir=str(output_dir / 'embeddings'),
            embedding_dim=word_emb_dim)

    # ── Line graph (training facts only) ────────────────────────────────────
    print("\nBuilding Line Graph...")
    id_train_triples    = [(ent2id[h], rel2id[r], ent2id[t])
                           for h, r, t in train_triples
                           if h in ent2id and r in rel2id and t in ent2id]
    train_triple_tensor = torch.tensor(id_train_triples, dtype=torch.long)
    fact_edge_index, _  = create_line_graph(train_triple_tensor)
    fact_edge_index     = fact_edge_index.to(device)

    # ── Precompute & move lookup tables to GPU ───────────────────────────────
    print("Pre-tokenizing entities and relations...")
    ent_desc, ent_len, ent_type, ent_names = precompute_entity_tensors(
        all_entities_sorted, entity2desc, entity2types, type2idx,
        word2idx, max_desc_len, num_types
    )
    rel_name_t, rel_domain_t, rel_range_t, rel_domain_words_t, rel_range_words_t = \
        precompute_relation_tensors(
            all_relations_sorted, relation2name, rel2domain, rel2range,
            type2idx, word2idx, num_types
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
    print("  Lookup tables on GPU.")

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

    if not weights_path:
        weights_path = _latest_fb20k_checkpoint()
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
    report_path = str(output_dir / f"fb20k_eval_report_{ts}.pdf")

    # ── BCE loss on sampled test triples (uses same hinge logic as training) ──
    print("\n--- BCE / Hinge Loss on sampled test triples ---")
    positive_set = set((h, r, t) for h, r, t in train_triples)

    pos_h_list = [ent2id[h] for h, r, t in train_triples
                  if h in ent2id and r in rel2id and t in ent2id]
    pos_r_list = [rel2id[r] for h, r, t in train_triples
                  if h in ent2id and r in rel2id and t in ent2id]
    pos_t_list = [ent2id[t] for h, r, t in train_triples
                  if h in ent2id and r in rel2id and t in ent2id]
    entity_to_facts: dict = {}
    for _i in range(len(pos_h_list)):
        for _e in (pos_h_list[_i], pos_t_list[_i]):
            entity_to_facts.setdefault(_e, []).append(_i)
    in_kg_ents = sorted(entity_to_facts.keys())

    # validate_loss reuses the exact same hinge + BFS logic as the training loop
    # Use small batch_size + max_neighbor_facts to avoid OOM on FB20k+ line graph
    # (485 M edges occupy most of GPU VRAM; the conv op needs the remainder).
    torch.cuda.empty_cache()
    hinge_loss, pos_mean, neg_mean = validate_loss(
        model, test_triples_sampled, positive_set,
        entity_to_facts, pos_h_list, pos_r_list, pos_t_list,
        ent_desc, ent_len, ent_type, ent_names,
        rel_name_t, rel_domain_t, rel_range_t,
        rel_domain_words_t, rel_range_words_t,
        ent2id, rel2id, device,
        in_kg_ents=in_kg_ents,
        num_layers=num_layers,
        batch_size=32,
        max_neighbor_facts=16,
        margin=1.0,
    )
    print(f"  Hinge loss (test) : {hinge_loss:.4f}")
    print(f"  Pos score mean    : {pos_mean:.4f}")
    print(f"  Neg score mean    : {neg_mean:.4f}")
    print(f"  Score gap         : {pos_mean - neg_mean:+.4f}")

    # ── Full-ranking MRR evaluation ──────────────────────────────────────────
    print("\nRunning Full-Ranking Test Evaluation on sampled test set...")
    test_mrr = evaluate_model(
        model, test_triples_sampled, metadata, word2idx, type2idx,
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
    print("FB20k+ Sampled Evaluation Complete")
    print(f"  Sample size     : {len(test_triples_sampled):,} / {total_test:,} total "
          f"(DBPedia parity = {n_sample:,})")
    print(f"  Hinge loss (test): {hinge_loss:.4f}")
    print(f"  Score gap        : {pos_mean - neg_mean:+.4f}  "
          f"(pos={pos_mean:.4f}  neg={neg_mean:.4f})")
    print(f"  Test MRR        : {test_mrr:.4f}")
    print(f"  Weights         : {weights_path}")
    print(f"  PDF report      : {report_path}")
    print(f"  Log             : {sys.stdout.log_path if hasattr(sys.stdout, 'log_path') else 'N/A'}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate IKGE on FB20k+ (sampled to DBPedia50k+ test-set size)")
    parser.add_argument(
        "--weights", default="",
        help="Path to .pt file. Omit to auto-use latest fb20k_ikge_w2v_best_mrr_*.pt")
    parser.add_argument(
        "--n-sample", type=int, default=DBPEDIA_TEST_SIZE,
        help=f"Number of test triples to sample (default: {DBPEDIA_TEST_SIZE})")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified sampling (default: 42)")
    parser.add_argument(
        "--run-name", default="",
        help="Optional label appended to the log filename.")
    args = parser.parse_args()
    run_eval(args.weights, args.run_name, args.n_sample, args.seed)
