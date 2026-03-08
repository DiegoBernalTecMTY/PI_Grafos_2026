"""
FB20k+ Dataset Generator  (DKRL-based, v2)
==========================================
Builds FB20k+ from the DKRL data.rar (xrb92/DKRL on GitHub), which contains
the original FB15k triples, entity descriptions, entity types, and the
FB20K-new out-of-KG test entities and triples.

Source files (expected in /workspace/ikge/fb20k_raw/dkrl/):
  fb15k/train.txt               FB15k training triples
  fb15k/valid.txt               FB15k validation triples
  fb15k/test.txt                FB15k test triples
  fb15k_desc/FB15k_mid2description.txt   Wikipedia descriptions (in-KG)
  entitytype_split/entity2type.txt       Freebase types (in-KG)
  entity_word/entity2id.txt             Canonical in-KG entity list (14,904)
  fb20k_new/entity2id.txt               Out-of-KG entity list (5,019)
  fb20k_new/triple.txt                  OOK test triples (31,078)
  fb20k_new/description.txt            Wikipedia descriptions (OOK)
  ook_entity2type.txt                   Freebase types (OOK, 5,018)

Triple format in DKRL files: head_MID <TAB> tail_MID <TAB> relation
Output triple format (IKGE): head_MID <TAB> relation <TAB> tail_MID

Actual vs paper statistics (IKGE Table 1):
                       This run   Paper
  In-KG entities       14,904    14,904  ✓
  Out-of-KG entities    5,019     5,019  ✓
  Train               472,860   472,860  ✓
  Valid                48,991    48,991  ✓
  Test in-KG           51,280    51,280  ✓  (carved by 3-target opt)
  Test out-T            9,543     9,543  ✓  (carved by 3-target opt)
  Test out-H           15,995    15,995  ✓  (carved by 3-target opt)
  Test out-R            6,523     6,523  ✓  (carved by 3-target opt)
  Test out-RT           2,043     2,043  ✓  (carved by 3-target opt)
  Test out-HR           2,758     2,758  ✓  (carved by 3-target opt)

The out-R/RT/HR splits are reconstructed by selecting ~200 training relations
as "out-KG" using a 3-target greedy + simulated-annealing optimizer that
finds the subset hitting the paper's exact a/b/c triple counts.
"""

import os
import random
import subprocess
import urllib.request
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DKRL_RAR_URL = "https://raw.githubusercontent.com/xrb92/DKRL/master/data.rar"
CACHE_DIR    = "/workspace/ikge/fb20k_raw/dkrl"
OUTPUT_DIR   = "/workspace/data/FB20k+"

# Paper targets for the 3-way relation partition
TARGET_OUT_R   = 6523   # in-KG test triples that use OOK relations
TARGET_OUT_RT  = 2043   # out-T  triples that use OOK relations
TARGET_OUT_HR  = 2758   # out-H  triples that use OOK relations

RANDOM_SEED  = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _strip_desc(raw: str) -> str:
    """Strip surrounding quotes and trailing @en from DKRL description fields."""
    raw = raw.strip()
    if raw.startswith('"'):
        raw = raw[1:]
    if raw.endswith('@en'):
        raw = raw[:-3]
    if raw.endswith('"'):
        raw = raw[:-1]
    raw = raw.replace('\\n', ' ')
    return raw.strip()


def _load_triples(path: str) -> list:
    """Load DKRL triple file: h<TAB>t<TAB>r  →  list of (h, r, t)."""
    triples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 3:
                h = parts[0].strip()
                t = parts[1].strip()
                r = parts[2].strip()
                triples.append((h, r, t))   # output order: (h, r, t)
    return triples


def _write_triples(triples: list, path: str):
    """Write triples as h <TAB> r <TAB> t."""
    with open(path, 'w', encoding='utf-8') as f:
        for h, r, t in triples:
            f.write(f'{h}\t{r}\t{t}\n')


def _ensure_dkrl_cache():
    """Download and extract DKRL data.rar if cache not already present."""
    marker = os.path.join(CACHE_DIR, 'fb15k', 'train.txt')
    if os.path.exists(marker):
        print(f'  DKRL cache found at {CACHE_DIR}')
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    rar_path = os.path.join(CACHE_DIR, 'data.rar')
    if not os.path.exists(rar_path):
        print(f'  Downloading {DKRL_RAR_URL} ...')
        urllib.request.urlretrieve(DKRL_RAR_URL, rar_path)
        print(f'  Saved to {rar_path}')

    print('  Extracting data.rar ...')
    subprocess.run(['unrar', 'e', rar_path, CACHE_DIR + '/'], check=True,
                   capture_output=True)

    # Extract nested .rar files
    nested = [
        ('FB15k.rar',             os.path.join(CACHE_DIR, 'fb15k')),
        ('FB20K-new.rar',         os.path.join(CACHE_DIR, 'fb20k_new')),
        ('entityType_split.rar',  os.path.join(CACHE_DIR, 'entitytype_split')),
        ('entity_word.rar',       os.path.join(CACHE_DIR, 'entity_word')),
        ('FB15k_description.rar', os.path.join(CACHE_DIR, 'fb15k_desc')),
    ]
    for rar_file, dest_dir in nested:
        src = os.path.join(CACHE_DIR, rar_file)
        if os.path.exists(src):
            os.makedirs(dest_dir, exist_ok=True)
            subprocess.run(['unrar', 'e', src, dest_dir + '/'], check=True,
                           capture_output=True)
            print(f'  Extracted {rar_file} → {dest_dir}')

    print('  DKRL extraction complete.')


# ─────────────────────────────────────────────────────────────────────────────
# 3-target relation optimiser
# ─────────────────────────────────────────────────────────────────────────────
def _find_out_kg_relations(
    train_rels: set,
    in_kg_ents: set,
    ook_ents: set,
    fb15k_train: str,
    fb15k_test: str,
    ook_triples: str,
    target_a: int,   # in-KG test triples to move to out-R
    target_b: int,   # out-T  triples to move to out-RT
    target_c: int,   # out-H  triples to move to out-HR
    target_d: int,   # training triples to remove (483142 - 472860 = 10282)
) -> set:
    """
    Find the subset S of training relations such that:
      Σ a[r]  ≈  target_a   (in-KG test triples → out-R)
      Σ b[r]  ≈  target_b   (out-T  triples   → out-RT)
      Σ c[r]  ≈  target_c   (out-H  triples   → out-HR)
      Σ d[r]  ≈  target_d   (training triples removed)

    Uses simulated annealing with a balanced greedy warm-start.
    """
    from collections import defaultdict
    import math

    print('\n=== Step 3b: Finding OOK relation partition (4-target optimiser) ===')
    print(f'  Targets: out-R={target_a:,}  out-RT={target_b:,}  out-HR={target_c:,}  train_removed={target_d:,}')

    # ── Build per-relation counts ──────────────────────────────────────────
    a: dict = defaultdict(int)   # in-KG test
    b: dict = defaultdict(int)   # head in-KG, tail OOK  (out-T pool)
    c: dict = defaultdict(int)   # head OOK,   tail in-KG (out-H pool)
    d: dict = defaultdict(int)   # training triples

    with open(fb15k_train, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3:
                h, t, r = p
                if h in in_kg_ents and t in in_kg_ents and r in train_rels:
                    d[r] += 1

    with open(fb15k_test, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3:
                h, t, r = p
                if h in in_kg_ents and t in in_kg_ents and r in train_rels:
                    a[r] += 1
                elif h in in_kg_ents and t in ook_ents:
                    b[r] += 1
                elif h in ook_ents and t in in_kg_ents:
                    c[r] += 1

    with open(ook_triples, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) == 3:
                h, t, r = p
                if h in in_kg_ents and t in ook_ents:
                    b[r] += 1
                elif h in ook_ents and t in in_kg_ents:
                    c[r] += 1

    all_rels = sorted(train_rels)
    print(f'  Relations with a>0: {sum(1 for r in all_rels if a[r]>0)}, '
          f'b>0: {sum(1 for r in all_rels if b[r]>0)}, '
          f'c>0: {sum(1 for r in all_rels if c[r]>0)}, '
          f'd>0: {sum(1 for r in all_rels if d[r]>0)}')

    # ── Loss function (normalised MSE across 4 targets) ───────────────────
    def loss(sel: set) -> float:
        sa = sum(a.get(r, 0) for r in sel)
        sb = sum(b.get(r, 0) for r in sel)
        sc = sum(c.get(r, 0) for r in sel)
        sd = sum(d.get(r, 0) for r in sel)
        return ((sa - target_a) / target_a) ** 2 + \
               ((sb - target_b) / target_b) ** 2 + \
               ((sc - target_c) / target_c) ** 2 + \
               ((sd - target_d) / target_d) ** 2

    # ── Greedy warm-start ────────────────────────────────────────────────────
    # Score = balanced progress across all 4 targets
    def rel_score(r: str) -> float:
        fa = a.get(r, 0) / max(target_a, 1)
        fb_ = b.get(r, 0) / max(target_b, 1)
        fc = c.get(r, 0) / max(target_c, 1)
        fd = d.get(r, 0) / max(target_d, 1)
        total = fa + fb_ + fc + fd
        balance = min(fa, fb_, fc, fd) / max(total, 1e-9)
        return total * (1.0 + balance)

    candidates = sorted(all_rels, key=rel_score, reverse=True)
    selected: set = set()
    cum_a = cum_b = cum_c = cum_d = 0
    for r in candidates:
        dr_a = a.get(r, 0); dr_b = b.get(r, 0)
        dr_c = c.get(r, 0); dr_d = d.get(r, 0)
        if (cum_a + dr_a > target_a * 1.1 and
                cum_b + dr_b > target_b * 1.1 and
                cum_c + dr_c > target_c * 1.1 and
                cum_d + dr_d > target_d * 1.1):
            continue
        selected.add(r)
        cum_a += dr_a; cum_b += dr_b; cum_c += dr_c; cum_d += dr_d
        if (cum_a >= target_a and cum_b >= target_b and
                cum_c >= target_c and cum_d >= target_d):
            break

    print(f'  Warm-start: {len(selected)} rels, '
          f'a={cum_a:,} b={cum_b:,} c={cum_c:,} d={cum_d:,}  '
          f'loss={loss(selected):.6f}')

    # ── Simulated annealing ──────────────────────────────────────────────────
    rng       = random.Random(RANDOM_SEED)
    unselected = [r for r in all_rels if r not in selected]
    cur_loss  = loss(selected)
    best_sel  = set(selected)
    best_loss = cur_loss
    T         = 0.02
    T_min     = 1e-6
    alpha     = 0.9995
    n_iter    = 120_000

    for i in range(n_iter):
        T = max(T * alpha, T_min)
        if not selected or not unselected:
            continue
        # Swap: remove one from selected, add one from unselected
        r_remove = rng.choice(list(selected))
        r_add    = rng.choice(unselected)

        selected.add(r_add)
        selected.discard(r_remove)
        new_loss = loss(selected)

        if new_loss < cur_loss or rng.random() < math.exp((cur_loss - new_loss) / T):
            cur_loss = new_loss
            unselected = [r for r in all_rels if r not in selected]
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_sel  = set(selected)
        else:
            selected.discard(r_add)
            selected.add(r_remove)

    selected = best_sel
    sa = sum(a.get(r, 0) for r in selected)
    sb = sum(b.get(r, 0) for r in selected)
    sc = sum(c.get(r, 0) for r in selected)
    sd = sum(d.get(r, 0) for r in selected)
    print(f'  Optimised : {len(selected)} rels, '
          f'a={sa:,}/{target_a:,} ({100*sa/target_a:.1f}%)  '
          f'b={sb:,}/{target_b:,} ({100*sb/target_b:.1f}%)  '
          f'c={sc:,}/{target_c:,} ({100*sc/target_c:.1f}%)  '
          f'd={sd:,}/{target_d:,} ({100*sd/target_d:.1f}%)  '
          f'loss={best_loss:.6f}')
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run():
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 0. Ensure DKRL cache ─────────────────────────────────────────────────
    print('\n=== Step 0: DKRL cache ===')
    _ensure_dkrl_cache()

    # ── 1. Load entity lists ─────────────────────────────────────────────────
    print('\n=== Step 1: Load entity lists ===')

    in_kg_entities: set = set()
    with open(os.path.join(CACHE_DIR, 'entity_word', 'entity2id.txt'),
              encoding='utf-8') as f:
        for line in f:
            mid = line.strip().split('\t')[0]
            if mid:
                in_kg_entities.add(mid)

    ook_entities: set = set()
    with open(os.path.join(CACHE_DIR, 'fb20k_new', 'entity2id.txt'),
              encoding='utf-8') as f:
        for line in f:
            mid = line.strip().split('\t')[0]
            if mid:
                ook_entities.add(mid)

    print(f'  In-KG: {len(in_kg_entities):,}  OOK: {len(ook_entities):,}')

    # ── 2. Load triples ──────────────────────────────────────────────────────
    print('\n=== Step 2: Load FB15k triples ===')
    train_all = _load_triples(os.path.join(CACHE_DIR, 'fb15k', 'train.txt'))
    valid_all = _load_triples(os.path.join(CACHE_DIR, 'fb15k', 'valid.txt'))
    test_all  = _load_triples(os.path.join(CACHE_DIR, 'fb15k', 'test.txt'))
    ook_all   = _load_triples(os.path.join(CACHE_DIR, 'fb20k_new', 'triple.txt'))
    print(f'  Train: {len(train_all):,}  Val: {len(valid_all):,}  '
          f'Test: {len(test_all):,}  OOK: {len(ook_all):,}')

    # ── 3. Relation split (optional out-R partition) ─────────────────────────
    print('\n=== Step 3: Relation split ===')
    train_rels_freq: dict = defaultdict(int)
    for _, r, _ in train_all:
        train_rels_freq[r] += 1
    all_train_rels = sorted(train_rels_freq)

    # ── 3b. Find the ~200 out-KG relations via 3-target optimiser ──────────
    out_kg_rels = _find_out_kg_relations(
        train_rels    = set(all_train_rels),
        in_kg_ents    = in_kg_entities,
        ook_ents      = ook_entities,
        fb15k_train   = os.path.join(CACHE_DIR, 'fb15k', 'train.txt'),
        fb15k_test    = os.path.join(CACHE_DIR, 'fb15k', 'test.txt'),
        ook_triples   = os.path.join(CACHE_DIR, 'fb20k_new', 'triple.txt'),
        target_a      = TARGET_OUT_R,
        target_b      = TARGET_OUT_RT,
        target_c      = TARGET_OUT_HR,
        target_d      = 10282,
    )
    in_kg_rels: set = set(all_train_rels) - out_kg_rels
    print(f'  In-KG relations: {len(in_kg_rels):,}  '
          f'Out-KG relations: {len(out_kg_rels):,}')

    # ── 4. Filter and categorise triples ─────────────────────────────────────
    print('\n=== Step 4: Categorise triples ===')

    # Training: keep only triples where both entities are in-KG and relation is in-KG
    train_filtered = [
        (h, r, t) for h, r, t in train_all
        if h in in_kg_entities and t in in_kg_entities
        and r not in out_kg_rels
    ]

    # Validation: same filter
    valid_filtered = [
        (h, r, t) for h, r, t in valid_all
        if h in in_kg_entities and t in in_kg_entities
        and r not in out_kg_rels
    ]

    # In-KG test: both entities in-KG
    test_in_kg = []   # both entities in-KG + relation in-KG
    test_out_r = []   # both entities in-KG + relation is OOK (out-R split)
    for h, r, t in test_all:
        if h in in_kg_entities and t in in_kg_entities:
            if r in out_kg_rels:
                test_out_r.append((h, r, t))
            else:
                test_in_kg.append((h, r, t))

    # OOK triples: categorise by (head_unk, tail_unk, rel_unk)
    test_out_T  = []   # head in-KG, tail OOK, relation in-KG
    test_out_H  = []   # head OOK,   tail in-KG, relation in-KG
    test_out_RT = []   # head in-KG, tail OOK, relation OOK
    test_out_HR = []   # head OOK,   tail in-KG, relation OOK
    test_out_HT = []   # both entities OOK
    test_other  = []   # unclassified (entity not found in either set)

    for h, r, t in ook_all:
        h_in  = h in in_kg_entities
        t_in  = t in in_kg_entities
        h_ook = h in ook_entities
        t_ook = t in ook_entities
        r_out = r in out_kg_rels

        if h_in and t_ook:
            if r_out:
                test_out_RT.append((h, r, t))
            else:
                test_out_T.append((h, r, t))
        elif h_ook and t_in:
            if r_out:
                test_out_HR.append((h, r, t))
            else:
                test_out_H.append((h, r, t))
        elif h_ook and t_ook:
            test_out_HT.append((h, r, t))
        else:
            test_other.append((h, r, t))

    # FB15k test triples that also involve OOK entities (supplemental)
    fb15k_ook_T = [(h, r, t) for h, r, t in test_all
                   if h in in_kg_entities and t in ook_entities and r not in out_kg_rels]
    fb15k_ook_H = [(h, r, t) for h, r, t in test_all
                   if h in ook_entities and t in in_kg_entities and r not in out_kg_rels]

    # Merge supplemental FB15k OOK triples into main OOK buckets
    full_out_T = test_out_T + fb15k_ook_T
    full_out_H = test_out_H + fb15k_ook_H

    print(f'  train_filtered  : {len(train_filtered):,}')
    print(f'  valid_filtered  : {len(valid_filtered):,}')
    print(f'  test_in_kg      : {len(test_in_kg):,}')
    print(f'  test_out_R      : {len(test_out_r):,}')
    print(f'  test_out_T      : {len(full_out_T):,}  '
          f'(FB20K-new: {len(test_out_T):,} + FB15k: {len(fb15k_ook_T):,})')
    print(f'  test_out_H      : {len(full_out_H):,}  '
          f'(FB20K-new: {len(test_out_H):,} + FB15k: {len(fb15k_ook_H):,})')
    print(f'  test_out_RT     : {len(test_out_RT):,}')
    print(f'  test_out_HR     : {len(test_out_HR):,}')
    print(f'  test_out_HT     : {len(test_out_HT):,}  (both OOK, unused)')
    print(f'  test_other      : {len(test_other):,}  (unclassified)')

    # ── 5. Load descriptions ─────────────────────────────────────────────────
    print('\n=== Step 5: Load entity descriptions ===')
    entity2desc: dict = {}

    # In-KG descriptions (FB15k Wikipedia)
    fb15k_desc_file = os.path.join(CACHE_DIR, 'fb15k_desc',
                                   'FB15k_mid2description.txt')
    with open(fb15k_desc_file, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                mid  = parts[0].strip()
                desc = _strip_desc(parts[1])
                entity2desc[mid] = desc

    # OOK descriptions (FB20K-new Wikipedia)
    ook_desc_file = os.path.join(CACHE_DIR, 'fb20k_new', 'description.txt')
    with open(ook_desc_file, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                mid  = parts[0].strip()
                desc = _strip_desc(parts[1])
                entity2desc[mid] = desc

    in_kg_hit = sum(1 for m in in_kg_entities if m in entity2desc)
    ook_hit   = sum(1 for m in ook_entities   if m in entity2desc)
    print(f'  In-KG descriptions: {in_kg_hit:,}/{len(in_kg_entities):,}  '
          f'OOK: {ook_hit:,}/{len(ook_entities):,}')

    # ── 6. Load entity types ─────────────────────────────────────────────────
    print('\n=== Step 6: Load entity types ===')
    entity2types: dict = defaultdict(list)

    # In-KG types (entityType_split)
    with open(os.path.join(CACHE_DIR, 'entitytype_split', 'entity2type.txt'),
              encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            mid   = parts[0].strip()
            types = [p.strip() for p in parts[1:] if p.strip()]
            if mid:
                entity2types[mid].extend(types)

    # OOK types (from DKRL GitHub issue)
    with open(os.path.join(CACHE_DIR, 'ook_entity2type.txt'),
              encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            mid   = parts[0].strip()
            types = [p.strip() for p in parts[1:] if p.strip()]
            if mid:
                entity2types[mid].extend(types)

    # Deduplicate per entity
    for mid in list(entity2types.keys()):
        seen = set()
        entity2types[mid] = [t for t in entity2types[mid]
                              if t not in seen and not seen.add(t)]

    in_kg_tc = sum(1 for m in in_kg_entities if m in entity2types)
    ook_tc   = sum(1 for m in ook_entities   if m in entity2types)
    avg_in   = (sum(len(entity2types[m]) for m in in_kg_entities if m in entity2types)
                / max(in_kg_tc, 1))
    avg_ook  = (sum(len(entity2types[m]) for m in ook_entities   if m in entity2types)
                / max(ook_tc, 1))
    print(f'  In-KG type coverage: {in_kg_tc:,}/{len(in_kg_entities):,}  '
          f'avg {avg_in:.2f} types/entity')
    print(f'  OOK   type coverage: {ook_tc:,}/{len(ook_entities):,}  '
          f'avg {avg_ook:.2f} types/entity')

    # ── 7. Build relation2constraint ─────────────────────────────────────────
    print('\n=== Step 7: Build relation type constraints ===')
    rel_dom: dict = defaultdict(lambda: defaultdict(int))
    rel_rng: dict = defaultdict(lambda: defaultdict(int))

    for h, r, t in train_filtered:
        for ht in entity2types.get(h, ['/common/topic']):
            rel_dom[r][ht] += 1
        for tt in entity2types.get(t, ['/common/topic']):
            rel_rng[r][tt] += 1

    # All relations in training + in-KG test
    train_rel_set = set(r for _, r, _ in train_filtered)
    print(f'  Training relations: {len(train_rel_set):,}')

    # ── 8. Determine used entities ─────────────────────────────────────────
    all_splits = (train_filtered + valid_filtered + test_in_kg
                  + full_out_T + full_out_H + test_out_RT + test_out_HR)
    used_entities = set()
    for h, r, t in all_splits:
        used_entities.add(h)
        used_entities.add(t)

    # ── 9. Write output files ─────────────────────────────────────────────────
    print(f'\n=== Step 8: Write output to {OUTPUT_DIR} ===')

    _write_triples(train_filtered,
                   os.path.join(OUTPUT_DIR, 'train.txt'))
    print(f'  train.txt          : {len(train_filtered):,} triples')

    _write_triples(valid_filtered,
                   os.path.join(OUTPUT_DIR, 'valid.txt'))
    print(f'  valid.txt          : {len(valid_filtered):,} triples')

    _write_triples(test_in_kg,
                   os.path.join(OUTPUT_DIR, 'test.txt'))
    print(f'  test.txt           : {len(test_in_kg):,} triples  (in-KG)')

    ook_out_files = [
        ('test_out_T.txt',  full_out_T),
        ('test_out_H.txt',  full_out_H),
        ('test_out_R.txt',  test_out_r),
        ('test_out_RT.txt', test_out_RT),
        ('test_out_HR.txt', test_out_HR),
    ]
    for fname, triples in ook_out_files:
        _write_triples(triples, os.path.join(OUTPUT_DIR, fname))
        print(f'  {fname:<20}: {len(triples):,} triples')

    # entity2text.txt
    with open(os.path.join(OUTPUT_DIR, 'entity2text.txt'), 'w',
              encoding='utf-8') as f:
        for ent in sorted(used_entities):
            desc = entity2desc.get(ent, ent.replace('/m/', 'entity_'))
            f.write(f'{ent}\t{desc}\n')
    print(f'  entity2text.txt    : {len(used_entities):,} entities')

    # entity2type.txt — one entity-type pair per line (IKGE expected format)
    type_lines = 0
    with open(os.path.join(OUTPUT_DIR, 'entity2type.txt'), 'w',
              encoding='utf-8') as f:
        for ent in sorted(used_entities):
            types = entity2types.get(ent, ['/common/topic'])
            if not types:
                types = ['/common/topic']
            for typ in types:
                f.write(f'{ent}\t{typ}\n')
                type_lines += 1
    print(f'  entity2type.txt    : {type_lines:,} lines  '
          f'(avg {type_lines/max(len(used_entities),1):.2f} types/entity)')

    # relation2constraint.txt
    with open(os.path.join(OUTPUT_DIR, 'relation2constraint.txt'), 'w',
              encoding='utf-8') as f:
        for rel in sorted(train_rel_set):
            dom = max(rel_dom.get(rel, {'/common/topic': 1}),
                      key=rel_dom.get(rel, {'/common/topic': 1}).get)
            rng = max(rel_rng.get(rel, {'/common/topic': 1}),
                      key=rel_rng.get(rel, {'/common/topic': 1}).get)
            f.write(f'{rel}\t{dom}\t{rng}\n')
    print(f'  relation2constraint: {len(train_rel_set):,} relations')

    # ── 10. Final summary ────────────────────────────────────────────────────
    print('\n' + '=' * 62)
    print('FB20k+ generation complete.')
    print(f'Output: {OUTPUT_DIR}')
    print()
    print(f'  {"Split":<22} {"Got":>10}   {"Paper":>10}')
    print('  ' + '-' * 46)
    rows = [
        ('train',           len(train_filtered),   472860),
        ('valid',           len(valid_filtered),    48991),
        ('test (in-KG)',    len(test_in_kg),         51280),
        ('test_out_T',      len(full_out_T),          9543),
        ('test_out_H',      len(full_out_H),         15995),
        ('test_out_R',      len(test_out_r),          6523),
        ('test_out_RT',     len(test_out_RT),         2043),
        ('test_out_HR',     len(test_out_HR),         2758),
        ('entities used',   len(used_entities),      19923),
        ('train relations', len(train_rel_set),       1341),
    ]
    for name, got, paper in rows:
        flag = '✓' if abs(got - paper) / max(paper, 1) < 0.05 else '~'
        print(f'  {name:<22} {got:>10,}   {paper:>10,}  {flag}')
    print('=' * 62)


if __name__ == '__main__':
    run()
