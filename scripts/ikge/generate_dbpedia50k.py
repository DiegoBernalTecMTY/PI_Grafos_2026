"""
DBPedia50k+ Dataset Generator
=============================
This script downloads DBPedia core files, parses the N-Triples streams,
and samples the graph to match the EXACT statistics of the DBPedia50k+ dataset
from the IKGE paper (Byungkook Oh et al. 2021).

Expected Statistics:
- In-KG Entities: 49,900
- Out-of-KG Entities: 5,699
- Total Entities Expected: 55,599
- In-KG Relations: 654
- Out-of-KG Relations: 96
- Total Relations Expected: 750
"""

import os
import bz2
import io
import pickle
import urllib.request
import urllib.parse
import random
from tqdm import tqdm
from collections import defaultdict
from dbo_hierarchy import get_all_types

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "/workspace/ikge/dbpedia_raw"
OUTPUT_DIR = "/workspace/data/DBPedia50k+"

# Using older 2016-10 DBPedia dumps as they were standard for 2018-2021 papers
URLS = {
    "abstracts": "http://downloads.dbpedia.org/2016-10/core-i18n/en/short_abstracts_en.ttl.bz2",
    "types": "http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_en.ttl.bz2",
    "objects": "http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_objects_en.ttl.bz2"
}

FILES = {k: os.path.join(DATA_DIR, os.path.basename(v)) for k, v in URLS.items()}


# Targets exactly defined in IKGE Paper Table 1.
TARGET_IN_ENT = 49900
TARGET_OUT_ENT = 5699
TARGET_TOTAL_ENT = TARGET_IN_ENT + TARGET_OUT_ENT

TARGET_IN_REL = 654
TARGET_OUT_REL = 96
TARGET_TOTAL_REL = TARGET_IN_REL + TARGET_OUT_REL

# Exact split counts from IKGE Paper Table 1.
# --- in-KG only (h, r, t all in in-KG sets) ---
TARGET_TRAIN         = 32388  # train
TARGET_VAL           = 399    # val  (held-out in-KG triples)
TARGET_TEST_IN_KG    = 2001   # test: all in KG
# --- test categories where at least one component is out-KG ---
TARGET_TEST_OUT_T    = 4238   # test: second entity out of KG  (h∈in, r∈in, t∈out)
TARGET_TEST_OUT_H    = 2862   # test: first  entity out of KG  (h∈out, r∈in, t∈in)
TARGET_TEST_OUT_R    = 321    # test: relation out of KG        (h∈in, r∈out, t∈in)
TARGET_TEST_OUT_RT   = 814    # test: relation + second entity  (h∈in, r∈out, t∈out)
TARGET_TEST_OUT_HR   = 473    # test: first entity + relation   (h∈out, r∈out, t∈in)

# -----------------------------------------------------------------------------
# 1. Download Files
# -----------------------------------------------------------------------------
def download_progress(filename):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
            
    def hook(t):
        last_b = [0]
        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b
        return update_to

    return hook

def download_if_missing():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name, url in URLS.items():
        filepath = FILES[name]
        if not os.path.exists(filepath):
            print(f"Downloading {name} from {url} ...")
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=name) as t:
                urllib.request.urlretrieve(url, filename=filepath, reporthook=download_progress(name)(t))
        else:
            print(f"Found {filepath}. Skipping download.")

# -----------------------------------------------------------------------------
# 2. Parsing N-Triples (TTL format) helpers
# -----------------------------------------------------------------------------
def clean_uri(uri):
    """Strips < and > and the dbpedia base URL to leave just the resource name."""
    uri = uri.strip('<>')
    if uri.startswith("http://dbpedia.org/resource/"):
        return uri.replace("http://dbpedia.org/resource/", "dbr:")
    elif uri.startswith("http://dbpedia.org/ontology/"):
        return uri.replace("http://dbpedia.org/ontology/", "dbo:")
    elif uri.startswith("http://dbpedia.org/property/"):
        return uri.replace("http://dbpedia.org/property/", "dbp:")
    return uri


# Relations to reject: meta-predicates that are not meaningful KG facts.
# owl:differentFrom, rdfs:seeAlso, rdf:type etc. pollute the relation sample.
_BLOCKED_PREFIXES = (
    "http://www.w3.org/",        # owl:, rdfs:, rdf:, xsd:, etc.
    "http://purl.org/",          # Dublin Core and similar
    "http://schema.org/",
    "http://xmlns.com/",
)

def is_semantic_relation(rel: str) -> bool:
    """Return True only for dbo: / dbp: predicates (meaningful KG relations)."""
    return rel.startswith("dbo:") or rel.startswith("dbp:")

def extract_literal(literal):
    """Extracts string inside quotes."""
    if '"' in literal:
        start = literal.find('"')
        end = literal.rfind('"')
        if start != -1 and end != -1 and start != end:
            return literal[start+1:end].replace('\\"', '"')
    return literal

# -----------------------------------------------------------------------------
# 3. Graph Extraction Pipeline  (in-memory, 32 GB RAM variant)
# -----------------------------------------------------------------------------

# bz2 decompresses slowly with the default 8 KB chunk; 64 MB keeps the CPU fed.
_BZ2_BUFSIZE = 64 * 1024 * 1024


def _open_bz2(path):
    """Open a bz2-compressed text file with a large read buffer."""
    return io.TextIOWrapper(
        io.BufferedReader(bz2.BZ2File(path), buffer_size=_BZ2_BUFSIZE),
        encoding="utf-8",
    )


def run_pipeline():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    abstracts_cache = os.path.join(DATA_DIR, "entity_descriptions.pkl")
    types_cache     = os.path.join(DATA_DIR, "entity_types.pkl")
    triples_cache   = os.path.join(DATA_DIR, "triples.pkl")

    # ------------------------------------------------------------------
    # Pass 1 + 2  (cached after first run)
    # ------------------------------------------------------------------
    if os.path.exists(abstracts_cache) and os.path.exists(types_cache):
        print("\nLoading cached entity data (Pass 1 & 2)...")
        with open(abstracts_cache, "rb") as f: entity_descriptions = pickle.load(f)
        with open(types_cache,     "rb") as f: entity_types        = pickle.load(f)
        # Detect old format (str per entity) → upgrade to list format in-place
        # and flush the updated cache so subsequent runs skip this upgrade.
        _sample_val = next(iter(entity_types.values())) if entity_types else []
        if isinstance(_sample_val, str):
            print("  Detected old single-type cache format — upgrading to list format...")
            entity_types = defaultdict(list, {k: [v] for k, v in entity_types.items()})
            with open(types_cache, "wb") as _f: pickle.dump(dict(entity_types), _f, protocol=4)
            print("  Cache updated.")
        print(f"  {len(entity_descriptions):,} descriptions  |  {len(entity_types):,} typed entities")
    else:
        print("\n--- Pass 1: Scanning Abstracts ---")
        entity_descriptions = {}
        with _open_bz2(FILES["abstracts"]) as f:
            for line in tqdm(f, desc="Abstracts", unit=" lines", mininterval=1.0):
                if line[0] != '<':
                    continue
                parts = line.split(' ', 2)
                if len(parts) >= 3:
                    entity_descriptions[clean_uri(parts[0])] = extract_literal(parts[2])
        print(f"  {len(entity_descriptions):,} entities with descriptions.")

        print("\n--- Pass 2: Scanning Types ---")
        entity_types: dict[str, list[str]] = defaultdict(list) # Changed to store list of types
        with _open_bz2(FILES["types"]) as f:
            for line in tqdm(f, desc="Types", unit=" lines", mininterval=1.0):
                if line[0] != '<':
                    continue
                parts = line.split(' ', 3)
                if len(parts) >= 3:
                    subj = clean_uri(parts[0])
                    obj_type = clean_uri(parts[2].rsplit(' ', 1)[0])  # strip trailing " ."
                    if subj in entity_descriptions:
                        entity_types[subj].append(obj_type)
        print(f"  {len(entity_types):,} entities with descriptions and types.")

        with open(abstracts_cache, "wb") as f: pickle.dump(entity_descriptions, f, protocol=4)
        with open(types_cache,     "wb") as f: pickle.dump(entity_types,        f, protocol=4)

    # ------------------------------------------------------------------
    # Expand each entity's leaf type to the full ancestor chain using the
    # hardcoded DBPedia ontology hierarchy (compensates for
    # instance_types_en.ttl.bz2 storing only the most-specific leaf type).
    # ------------------------------------------------------------------
    _expanded_count = 0
    for _ent in entity_types:
        _orig = list(entity_types[_ent])
        _full = set()
        for _t in _orig:
            _full.update(get_all_types(_t))   # leaf + all dbo: ancestors
        if len(_full) > len(_orig):
            _expanded_count += 1
        entity_types[_ent] = [t for t in _full if not t.startswith("owl:") and t != "dbo:Thing"]

    print(f"  Type hierarchy expansion: {_expanded_count:,} / {len(entity_types):,} entities gained ancestor types.")
    # Persist expanded types so subsequent runs don't redo the work
    with open(types_cache, "wb") as _f: pickle.dump(dict(entity_types), _f, protocol=4)

    valid_entities = set(entity_types.keys())

    # ------------------------------------------------------------------
    # Pass 3 – load all semantic triples into RAM  (cached after first run)
    # ------------------------------------------------------------------
    if os.path.exists(triples_cache):
        print("\nLoading cached triples (Pass 3)...")
        with open(triples_cache, "rb") as f: all_triples = pickle.load(f)
        print(f"  {len(all_triples):,} triples loaded from cache.")
    else:
        print("\n--- Pass 3: Extracting MappingBased Objects ---")
        all_triples = []       # list of (subj_str, rel_str, obj_str)
        with _open_bz2(FILES["objects"]) as f:
            for line in tqdm(f, desc="Edges", unit=" lines", mininterval=1.0):
                if line[0] != '<':
                    continue
                parts = line.split(' ', 3)
                if len(parts) >= 3:
                    rel = clean_uri(parts[1])
                    if not is_semantic_relation(rel):
                        continue
                    subj = clean_uri(parts[0])
                    obj  = clean_uri(parts[2])
                    # Keep only triples where both endpoints have metadata
                    if subj in valid_entities and obj in valid_entities:
                        all_triples.append((subj, rel, obj))
        print(f"  {len(all_triples):,} semantic triples retained.")
        with open(triples_cache, "wb") as f: pickle.dump(all_triples, f, protocol=4)

    # ------------------------------------------------------------------
    # Pass 4 – sampling
    # ------------------------------------------------------------------
    print("\n--- Pass 4: Sampling to IKGE paper statistics ---")

    global TARGET_TOTAL_REL, TARGET_IN_REL, TARGET_OUT_REL

    relations = sorted({r for _, r, _ in all_triples})
    random.seed(42)
    random.shuffle(relations)

    if len(relations) < TARGET_TOTAL_REL:
        print(f"WARNING: only {len(relations)} relations found; scaling targets.")
        TARGET_TOTAL_REL = len(relations)
        TARGET_IN_REL    = int(TARGET_TOTAL_REL * (654 / 750))
        TARGET_OUT_REL   = TARGET_TOTAL_REL - TARGET_IN_REL

    sampled_rel_list = relations[:TARGET_TOTAL_REL]
    sampled_relations = set(sampled_rel_list)

    # In-KG / out-KG relation split (same seed, next shuffle)
    in_kg_rel_list = sampled_rel_list[:]
    random.shuffle(in_kg_rel_list)
    in_kg_relations  = set(in_kg_rel_list[:TARGET_IN_REL])
    out_kg_relations = sampled_relations - in_kg_relations

    # Entity degree (only for sampled relations)
    entity_degree: dict[str, int] = defaultdict(int)
    for subj, rel, obj in all_triples:
        if rel in sampled_relations:
            entity_degree[subj] += 1
            entity_degree[obj]  += 1

    connected_entities = set(entity_degree)
    print(f"  Connected entities: {len(connected_entities):,}")

    if len(connected_entities) < TARGET_TOTAL_ENT:
        print("  Padding with isolated entities...")
        pool   = sorted(valid_entities - connected_entities)
        needed = TARGET_TOTAL_ENT - len(connected_entities)
        connected_entities.update(pool[:needed])

    # Prioritise high-degree nodes for in-KG (maximises training triple count)
    by_degree  = sorted(connected_entities, key=lambda e: -entity_degree.get(e, 0))
    in_kg_pool  = by_degree[:TARGET_IN_ENT]
    out_kg_pool = by_degree[TARGET_IN_ENT:TARGET_TOTAL_ENT]
    random.shuffle(in_kg_pool)
    random.shuffle(out_kg_pool)

    in_kg_entities  = set(in_kg_pool)
    out_kg_entities = set(out_kg_pool)
    sampled_entities = in_kg_entities | out_kg_entities

    # ------------------------------------------------------------------
    # Pass 5 – categorise triples into 6 paper-defined splits
    # ------------------------------------------------------------------
    print("\n--- Pass 5: Categorising triples into paper splits ---")
    rel2domain: dict[str, set] = defaultdict(set)
    rel2range:  dict[str, set] = defaultdict(set)

    # Buckets keyed by (h_in, r_in, t_in)
    cat: dict[tuple, list] = {
        (True,  True,  True ): [],  # → train / val / test_in_kg
        (True,  True,  False): [],  # → test: second entity out
        (False, True,  True ): [],  # → test: first entity out
        (True,  False, True ): [],  # → test: relation out
        (True,  False, False): [],  # → test: relation + second entity out
        (False, False, True ): [],  # → test: first entity + relation out
        # (False, True, False) and (False, False, False) not used in paper splits
    }

    for subj, rel, obj in all_triples:
        if rel not in sampled_relations:
            continue
        if subj not in sampled_entities or obj not in sampled_entities:
            continue
        h_in = subj in in_kg_entities
        t_in = obj  in in_kg_entities
        r_in = rel  in in_kg_relations
        key = (h_in, r_in, t_in)
        if key in cat:
            cat[key].append((subj, rel, obj))
        for _t in entity_types.get(subj, ["dbo:Thing"]):
            rel2domain[rel].add(_t)
        for _t in entity_types.get(obj, ["dbo:Thing"]):
            rel2range[rel].add(_t)

    for key, triples in cat.items():
        print(f"  {key}: {len(triples):,} triples")

    def _cap(triples, target):
        """Subsample to *target* preserving per-relation balance."""
        if len(triples) <= target:
            return list(triples)
        from collections import defaultdict as _dd
        by_rel = _dd(list)
        for t in triples:
            by_rel[t[1]].append(t)
        ratio = target / len(triples)
        result = []
        for rel_trips in by_rel.values():
            k = max(1, round(len(rel_trips) * ratio))
            result.extend(random.sample(rel_trips, min(k, len(rel_trips))))
        random.shuffle(result)
        return result[:target]

    # In-KG pool → train + val + test_in_kg
    in_kg_pool = _cap(cat[(True, True, True)],
                      TARGET_TRAIN + TARGET_VAL + TARGET_TEST_IN_KG)
    random.shuffle(in_kg_pool)
    train_triples   = in_kg_pool[:TARGET_TRAIN]
    val_triples     = in_kg_pool[TARGET_TRAIN : TARGET_TRAIN + TARGET_VAL]
    test_in_triples = in_kg_pool[TARGET_TRAIN + TARGET_VAL :]

    # Out-KG pools → capped to paper targets
    test_out_t  = _cap(cat[(True,  True,  False)], TARGET_TEST_OUT_T)
    test_out_h  = _cap(cat[(False, True,  True )], TARGET_TEST_OUT_H)
    test_out_r  = _cap(cat[(True,  False, True )], TARGET_TEST_OUT_R)
    test_out_rt = _cap(cat[(True,  False, False)], TARGET_TEST_OUT_RT)
    test_out_hr = _cap(cat[(False, False, True )], TARGET_TEST_OUT_HR)

    print(f"\n  After capping:")
    print(f"    train        : {len(train_triples):,}")
    print(f"    val (in-KG)  : {len(val_triples):,}")
    print(f"    test in-KG   : {len(test_in_triples):,}")
    print(f"    test out-T   : {len(test_out_t):,}")
    print(f"    test out-H   : {len(test_out_h):,}")
    print(f"    test out-R   : {len(test_out_r):,}")
    print(f"    test out-RT  : {len(test_out_rt):,}")
    print(f"    test out-HR  : {len(test_out_hr):,}")

    def select_type(type_set):
        return next(iter(type_set)) if type_set else "dbo:Thing"

    # ------------------------------------------------------------------
    # Pass 6 – write output files
    # ------------------------------------------------------------------
    print("\n--- Pass 6: Writing output files ---")

    with open(os.path.join(OUTPUT_DIR, "entity2text.txt"), "w", encoding="utf-8") as f:
        for ent in sampled_entities:
            f.write(f"{ent}\t{entity_descriptions[ent]}\n")

    with open(os.path.join(OUTPUT_DIR, "entity2type.txt"), "w", encoding="utf-8") as f:
        for ent in sampled_entities:
            # Write each type on a separate line for multi-type entities
            for typ in entity_types.get(ent, []):
                f.write(f"{ent}\t{typ}\n")

    with open(os.path.join(OUTPUT_DIR, "relation2constraint.txt"), "w", encoding="utf-8") as f:
        for rel in sampled_relations:
            f.write(f"{rel}\t{select_type(rel2domain[rel])}\t{select_type(rel2range[rel])}\n")

    train_count = valid_count = test_count = 0
    with open(os.path.join(OUTPUT_DIR, "train.txt"), "w", encoding="utf-8") as ftrain, \
         open(os.path.join(OUTPUT_DIR, "valid.txt"), "w", encoding="utf-8") as fvalid, \
         open(os.path.join(OUTPUT_DIR, "test.txt"),  "w", encoding="utf-8") as ftest:

        for h, r, t in train_triples:
            ftrain.write(f"{h}\t{r}\t{t}\n"); train_count += 1

        for h, r, t in val_triples:
            fvalid.write(f"{h}\t{r}\t{t}\n"); valid_count += 1

        for bucket in (test_in_triples, test_out_t, test_out_h,
                       test_out_r, test_out_rt, test_out_hr):
            for h, r, t in bucket:
                ftest.write(f"{h}\t{r}\t{t}\n"); test_count += 1

    print(f"\nDone! Output: {OUTPUT_DIR}")
    print(f"  Train: {train_count:,}  |  Valid: {valid_count:,}  |  Test: {test_count:,}")
    print(f"\nCaches kept in {DATA_DIR}/ for fast re-runs.")
    print("Delete *.pkl files there to force a full rebuild from the bz2 dumps.")


if __name__ == "__main__":
    download_if_missing()
    run_pipeline()
