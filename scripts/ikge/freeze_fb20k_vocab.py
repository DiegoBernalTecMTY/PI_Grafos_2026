"""
One-shot script: rebuild the EXACT word2idx that train_ikge_w2v.py produced
for FB20k+ and save it next to the checkpoint as fb20k_word2idx.pkl.

Run once:
    python3 freeze_fb20k_vocab.py

This is needed because setup_w2v_for_ikge rebuilds vocab from scratch each
time it is called, and any difference in NLTK lemmatizer state or tokenizer
behaviour between the original training session and now will cause the
embedding matrix shape to differ from the checkpoint.
"""

import os
import sys
import pickle
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Import exact tokenize / vocab builder from the W2V module ───────────────
from download_w2v import build_vocabulary_from_descriptions, tokenize_for_w2v

DATA_DIR  = '/workspace/data/FB20k+'
OUT_PATH  = Path(__file__).resolve().parent / 'fb20k_word2idx.pkl'

# ── Copy-paste of training's load_txt (no encoding kwarg, exactly as in train) ──
def load_txt(path):
    with open(path, 'r') as f:
        return [line.strip().split('\t') for line in f if line.strip()]

def _entity_name(e: str, desc: str = '') -> str:
    seg = e.split('/')[-1].split('#')[-1].replace('_', ' ')
    non_alpha = sum(1 for c in seg if not c.isalpha() and c != ' ')
    if seg and non_alpha > len(seg) * 0.4:
        words = desc.split()[:4]
        return ' '.join(words) if words else seg
    return seg

def _norm_type(t):
    return 'dbo:Thing' if 'owl#Thing' in t else t

print("Loading triples...")
train_triples = load_txt(os.path.join(DATA_DIR, 'train.txt'))
val_triples   = load_txt(os.path.join(DATA_DIR, 'valid.txt'))
test_triples  = load_txt(os.path.join(DATA_DIR, 'test.txt'))   # test.txt only — same as training

print(f"  train={len(train_triples):,}  val={len(val_triples):,}  test={len(test_triples):,}")

entity2desc = {x[0]: x[1] for x in load_txt(os.path.join(DATA_DIR, 'entity2text.txt')) if len(x) == 2}
print(f"  entity2desc entries: {len(entity2desc):,}")

entity2types = defaultdict(list)
for x in load_txt(os.path.join(DATA_DIR, 'entity2type.txt')):
    if len(x) == 2:
        entity2types[x[0]].append(_norm_type(x[1]))

rel2domain, rel2range = defaultdict(list), defaultdict(list)
for x in load_txt(os.path.join(DATA_DIR, 'relation2constraint.txt')):
    if len(x) == 3:
        rel2domain[x[0]].append(_norm_type(x[1]))
        rel2range[x[0]].append(_norm_type(x[2]))

all_triples          = train_triples + val_triples + test_triples
all_entities_sorted  = sorted(set(t[0] for t in all_triples) | set(t[2] for t in all_triples))
all_relations_sorted = sorted(set(t[1] for t in all_triples))
print(f"  entities={len(all_entities_sorted):,}  relations={len(all_relations_sorted):,}")

relation2name = {r: r.split('/')[-1].split('#')[-1].replace('_', ' ') for r in all_relations_sorted}

all_types = sorted(
    set(typ for ts in entity2types.values() for typ in ts)
    | set(t for ts in rel2domain.values() for t in ts)
    | set(t for ts in rel2range.values() for t in ts)
)
print(f"  types={len(all_types):,}")

entity_name_strings = [_entity_name(e, entity2desc.get(e, '')) for e in all_entities_sorted]
type_name_strings   = [t.split('/')[-1].split('#')[-1].replace('_', ' ') for t in all_types]
descriptions = (list(entity2desc.values())
                + list(relation2name.values())
                + entity_name_strings
                + type_name_strings)
print(f"  descriptions={len(descriptions):,}  (training had 40,396)")

print("\nBuilding vocabulary (this is the EXACT call training used)...")
word2idx, idx2word, _ = build_vocabulary_from_descriptions(descriptions)
print(f"\nVocab size: {len(word2idx):,}  (checkpoint has 105,129 — these must match)")

if len(word2idx) != 105_129:
    print("\n⚠️  WARNING: vocab size does not match checkpoint (105,129).")
    print("   This means the tokenizer / lemmatizer state differs from training.")
    print("   Using this vocab will still cause load_state_dict to fail.")
    print("   Run this ONLY in the environment where training was done, or")
    print("   patch the checkpoint to use the current vocab size.")
    ans = input("   Save anyway? [y/N] ").strip().lower()
    if ans != 'y':
        print("Aborted.")
        sys.exit(1)

with open(OUT_PATH, 'wb') as f:
    pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)

print(f"\n✅ Saved to {OUT_PATH}")
print("   Eval scripts can now load this instead of calling setup_w2v_for_ikge.")
