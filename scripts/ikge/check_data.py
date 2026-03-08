"""Quick data integrity check for DBPedia50k+."""
from collections import Counter, defaultdict

DATA = '/workspace/data/DBPedia50k+'

e2t_lines     = open(f'{DATA}/entity2text.txt').readlines()
e2type_lines  = open(f'{DATA}/entity2type.txt').readlines()
rel_lines     = open(f'{DATA}/relation2constraint.txt').readlines()
train_lines   = open(f'{DATA}/train.txt').readlines()
val_lines     = open(f'{DATA}/valid.txt').readlines()
test_lines    = open(f'{DATA}/test.txt').readlines()

# ── entity2text ───────────────────────────────────────────────────────────────
e2t_entities = [l.split('\t')[0] for l in e2t_lines if '\t' in l]
e2t_counts   = Counter(e2t_entities)
dups         = {k: v for k, v in e2t_counts.items() if v > 1}
print(f"entity2text : {len(e2t_entities)} lines | {len(e2t_counts)} unique entities")
print(f"  Duplicate entity entries : {len(dups)}")
for ent, cnt in sorted(dups.items(), key=lambda x: -x[1])[:5]:
    print(f"    {ent}: {cnt} entries")
    for l in e2t_lines:
        if l.startswith(ent + '\t'):
            print(f"      -> {l.split(chr(9),1)[1][:90].strip()}")

# ── entity2type ───────────────────────────────────────────────────────────────
type_by_entity = defaultdict(list)
for l in e2type_lines:
    parts = l.strip().split('\t')
    if len(parts) >= 2:
        type_by_entity[parts[0]].append(parts[1])
multi_type = {k: v for k, v in type_by_entity.items() if len(v) > 1}
print(f"\nentity2type : {len(e2type_lines)} lines | {len(type_by_entity)} unique entities")
print(f"  Entities with MULTIPLE types : {len(multi_type)}")
for ent, types in sorted(multi_type.items(), key=lambda x: -len(x[1]))[:5]:
    print(f"    {ent}: {types}")

# type frequency
all_types = [t for types in type_by_entity.values() for t in types]
type_freq  = Counter(all_types)
print(f"\n  Top 15 types:")
for t, c in type_freq.most_common(15):
    print(f"    {t}: {c}")

# ── relation2constraint ───────────────────────────────────────────────────────
rel_entries = defaultdict(list)
for l in rel_lines:
    parts = l.strip().split('\t')
    if len(parts) == 3:
        rel_entries[parts[0]].append((parts[1], parts[2]))
multi_rel = {k: v for k, v in rel_entries.items() if len(v) > 1}
thing_only = {r for r, cv in rel_entries.items()
              if all(d == 'dbo:Thing' and rg == 'dbo:Thing' for d, rg in cv)}
print(f"\nrelation2constraint : {len(rel_lines)} lines | {len(rel_entries)} unique relations")
print(f"  Relations with multiple constraint rows : {len(multi_rel)}")
for rel, cv in list(multi_rel.items())[:5]:
    print(f"    {rel}: {cv}")
print(f"  Relations with ONLY Thing/Thing (unconstrained): {len(thing_only)}/{len(rel_entries)}")

# ── KG entity coverage ────────────────────────────────────────────────────────
kg_ents = set()
for l in train_lines + val_lines + test_lines:
    parts = l.strip().split('\t')
    if len(parts) == 3:
        kg_ents.add(parts[0]); kg_ents.add(parts[2])
missing_desc = kg_ents - set(e2t_counts)
missing_type = kg_ents - set(type_by_entity)
print(f"\nKG entities (all splits) : {len(kg_ents)}")
print(f"  Missing descriptions   : {len(missing_desc)}")
if missing_desc: print(f"  Examples: {list(missing_desc)[:5]}")
print(f"  Missing types          : {len(missing_type)}")
if missing_type: print(f"  Examples: {list(missing_type)[:5]}")

# ── training triples: constraint coverage ────────────────────────────────────
rel_train_cnt = Counter()
for l in train_lines:
    parts = l.strip().split('\t')
    if len(parts) == 3:
        rel_train_cnt[parts[1]] += 1
train_rels = set(rel_train_cnt)
typed_train   = sum(rel_train_cnt[r] for r in train_rels
                    if r in rel_entries and r not in thing_only)
untyped_train = sum(rel_train_cnt[r] for r in train_rels
                    if r not in rel_entries or r in thing_only)
total_train   = sum(rel_train_cnt.values())
print(f"\nTraining triples with type-specific constraints : "
      f"{typed_train}/{total_train} ({100*typed_train/total_train:.1f}%)")
print(f"Training triples with no/Thing constraint       : "
      f"{untyped_train}/{total_train} ({100*untyped_train/total_train:.1f}%)")

# ── type hierarchy check (flat vs hierarchical) ───────────────────────────────
# Sample: pick 10 training triples with typed constraints and check flat match
print("\nFlat type match check on 20 typed training triples:")
checked = 0
mismatches = 0
for l in train_lines:
    parts = l.strip().split('\t')
    if len(parts) != 3:
        continue
    h, r, t = parts
    if r not in rel_entries or r in thing_only:
        continue
    for domain, rng in rel_entries[r]:
        h_types = set(type_by_entity.get(h, []))
        t_types = set(type_by_entity.get(t, []))
        domain_match = (domain == 'dbo:Thing') or (domain in h_types)
        range_match  = (rng   == 'dbo:Thing') or (rng   in t_types)
        if not (domain_match and range_match):
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH: ({h}, {r}, {t})")
                print(f"    h_types={h_types}  domain={domain}  match={domain_match}")
                print(f"    t_types={t_types}  range ={rng}   match={range_match}")
    checked += 1
    if checked >= 20:
        break

# broader mismatch rate
total_typed_checked = 0
flat_mismatch_count = 0
for l in train_lines:
    parts = l.strip().split('\t')
    if len(parts) != 3: continue
    h, r, t = parts
    if r not in rel_entries or r in thing_only: continue
    for domain, rng in rel_entries[r]:
        h_types = set(type_by_entity.get(h, []))
        t_types = set(type_by_entity.get(t, []))
        if not ((domain == 'dbo:Thing' or domain in h_types) and
                (rng   == 'dbo:Thing' or rng   in t_types)):
            flat_mismatch_count += 1
        total_typed_checked += 1
print(f"\nFlat mismatch over all typed train triples: "
      f"{flat_mismatch_count}/{total_typed_checked} "
      f"({100*flat_mismatch_count/max(total_typed_checked,1):.1f}%)")
print("(These fail because entity2type stores LEAF types; "
      "relation constraints use ANCESTOR types not in the file)")
