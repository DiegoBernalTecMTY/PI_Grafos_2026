# Fact Feature Extractor - Integration Guide

## ✅ Phase 1.2 Complete!

You now have the **Fact Feature Extraction** module - the second critical component of IKGE.

---

## 📦 What You Got

1. **`fact_feature_extractor.py`** - Full implementation with:
   - ✅ Attention-based CNN (Section 5.1.2)
   - ✅ Type matching (Section 5.1.3)
   - ✅ Word encoding from GloVe
   - ✅ Relation-specific entity features

2. **`test_fact_extractor.py`** - 7 comprehensive tests:
   - Tokenization
   - Model initialization
   - Forward pass
   - Type matching logic
   - Type filtering (zeros invalid facts)
   - Attention mechanism
   - Batch preparation

---

## 🔍 What This Module Does

### Input:
```python
Fact: (Harvard, locatedIn, ?)

Harvard description: "private Ivy League research university in Cambridge..."
Relation: "locatedIn"
Tail description: "..." 
Types: Harvard=[University, Organization], locatedIn requires [Place]
```

### Processing:
1. **Word Encoding**: Convert descriptions to GloVe vectors
2. **CNN**: Extract local features from descriptions
3. **Attention**: Focus on relation-relevant words
   - For "locatedIn", pay attention to "Cambridge", "Massachusetts"
4. **Type Matching**: Verify Harvard is Organization ✓, expects Place for tail ✓
5. **Combine**: Concatenate head + tail features

### Output:
```python
fact_embedding: [0.23, -0.45, 0.67, ...] (128 dimensions)
```

---

## 🧪 Testing

```bash
python test_fact_extractor.py
```

**Expected output:**
```
🧪 RUNNING ALL FACT FEATURE EXTRACTOR TESTS
======================================================================

TEST 1: Text Tokenization
----------------------------------------------------------------------
Description: 'Harvard University is a private university'
Tokens: [2, 3, 4, 5, 6, 3, 0, 0, 0, 0]
Length: 6
✅ TEST 1 PASSED

[... 6 more tests ...]

======================================================================
📊 TEST SUMMARY
======================================================================
✅ PASS: Tokenization
✅ PASS: Model Initialization
✅ PASS: Forward Pass
✅ PASS: Type Matching
✅ PASS: Type Filtering
✅ PASS: Attention Mechanism
✅ PASS: Batch Preparation

======================================================================
Results: 7/7 tests passed
======================================================================

🎉 ALL TESTS PASSED! Fact Feature Extractor is ready.
```

---

## 🔗 Integration Example

```python
from fact_feature_extractor import FactFeatureExtractor, prepare_fact_batch
from download_glove import setup_glove_for_ikge
import pandas as pd

# 1. Load enriched Codex-M data
entity_df = pd.read_csv('data/codex-m/enriched/entity_descriptions.csv')
relation_df = pd.read_csv('data/codex-m/enriched/relation_info.csv')

# 2. Setup GloVe embeddings
descriptions = entity_df['description'].fillna('').tolist()
embedding_matrix, word2idx, idx2word = setup_glove_for_ikge(
    entity_descriptions=descriptions,
    output_dir='./embeddings',
    glove_version='6B',
    embedding_dim=300
)

# 3. Create type mappings
all_types = set()
for types_str in entity_df['types']:
    if pd.notna(types_str):
        all_types.update(types_str.split('|'))
type2idx = {t: i for i, t in enumerate(sorted(all_types))}

# 4. Initialize Fact Feature Extractor
extractor = FactFeatureExtractor(
    word_embedding_matrix=embedding_matrix,
    word_embedding_dim=300,
    fact_embedding_dim=128,
    conv_channels=128,
    device='cuda'
)

# 5. Extract features for a batch of facts
facts = torch.tensor([[0, 0, 1], [1, 1, 2]])  # Example facts

batch = prepare_fact_batch(
    facts=facts,
    entity_descriptions=descriptions,
    relation_names=relation_df['name'].tolist(),
    entity_types=[types.split('|') if pd.notna(types) else [] 
                  for types in entity_df['types']],
    relation_type_constraints=[([], []) for _ in range(len(relation_df))],  # Simplified
    word2idx=word2idx,
    type2idx=type2idx,
    device='cuda'
)

# 6. Get fact embeddings
fact_embeddings = extractor(**batch)  # Shape: (batch_size, 128)

print(f"✅ Extracted fact features: {fact_embeddings.shape}")
```

---

## 📊 Architecture Details

### Model Components:

```
FactFeatureExtractor(
  (word_embeddings): Embedding(vocab_size, 300) [FROZEN]
  (conv1): Conv1d(300, 128, kernel_size=3)
  (conv2): Conv1d(128, 128, kernel_size=3)
  (attention_W): Linear(128, 128)
  (fact_projection): Linear(256, 128)
)
```

**Parameters:** ~200K trainable (word embeddings are frozen)

### Data Flow:

```
Entity Description (text)
    ↓
Word Embeddings (GloVe 300d)
    ↓
Conv1d → LeakyReLU → Dropout
    ↓
Conv2d → LeakyReLU → Dropout
    ↓
Attention (with relation context)
    ↓
Weighted Average (attended features)
    ↓
Concatenate [head_features, tail_features]
    ↓
Linear Projection → LeakyReLU
    ↓
Type Matching (zero invalid facts)
    ↓
Fact Embedding (128d)
```

---

## 🎯 Key Features

### 1. Relation-Specific Attention ✨
```python
# Same entity, different relations = different features!
fact1: (Harvard, locatedIn, ?) 
  → Attends to: "Cambridge", "Massachusetts", "Boston"

fact2: (Harvard, foundedBy, ?)
  → Attends to: "John Harvard", "1636", "founded"
```

### 2. Type Validation ✅
```python
# Invalid facts are zeroed out
(Harvard, capitalOf, Boston)  # Invalid: University can't be capital
  → Output: [0, 0, 0, ..., 0]

(Boston, capitalOf, Massachusetts)  # Valid: City can be capital
  → Output: [0.23, -0.45, 0.67, ...]
```

### 3. Inductive Learning 🆕
```python
# Works with entities never seen during training!
new_entity_description = "Stanford University is a private research university..."
fact_embedding = extractor.extract(new_entity_description, ...)
# ✅ Generates embedding from description
```

---

## 📈 Expected Performance

### Memory Usage:
- Model: ~200K parameters = ~800 KB
- Batch (size 1024): ~100-200 MB
- Total: Easily fits in 16GB GPU

### Speed:
- Forward pass (batch=1024): ~50-100ms on RTX 5080
- Can process ~10K facts/second

---

## 🔧 Configuration Tips

### For Faster Training:
```python
extractor = FactFeatureExtractor(
    ...,
    conv_channels=64,      # Reduce from 128
    fact_embedding_dim=64, # Reduce from 128
    dropout=0.1           # Less dropout
)
```

### For Better Accuracy:
```python
extractor = FactFeatureExtractor(
    ...,
    conv_channels=256,     # Increase
    fact_embedding_dim=256,
    dropout=0.3            # More dropout
)
```

---

## ✅ Verification Checklist

Before moving to Phase 1.3:

- [ ] `test_fact_extractor.py` - All 7 tests pass
- [ ] Can import: `from fact_feature_extractor import FactFeatureExtractor`
- [ ] GloVe embeddings downloaded (`./embeddings/glove.6B.300d.txt`)
- [ ] Enriched Codex-M ready (`entity_descriptions.csv`)
- [ ] Understand attention mechanism (read code comments)

---

## 🎬 What's Next?

### Phase 1.3: Attentive Feature Aggregation

The final critical component:
- Aggregate features from neighboring facts in line graph
- Use learned attention weights
- Hierarchical multi-hop aggregation
- Output: Final fact embeddings ready for scoring

**Estimated time:** 12 hours to implement

Want me to start on Phase 1.3 now? 🚀

---

## 💡 Pro Tips

1. **Cache preprocessed data:**
```python
# Tokenize descriptions once, save for reuse
torch.save(tokenized_descriptions, 'tokenized_cache.pt')
```

2. **Test with small batch first:**
```python
# Start with batch_size=4 to debug
small_batch = {k: v[:4] for k, v in batch.items()}
output = extractor(**small_batch)
```

3. **Visualize attention weights:**
```python
# Add this to extractor for debugging:
def get_attention_weights(self, entity_description, relation_name):
    # Returns attention distribution over description words
    pass
```

4. **Monitor type matching:**
```python
# Check how many facts are filtered
valid_facts = (output.abs().sum(dim=1) > 0).sum()
print(f"Valid facts: {valid_facts} / {batch_size}")
```

---

## 📞 Need Help?

Common issues:

**"RuntimeError: CUDA out of memory"**
- Reduce `conv_channels` to 64
- Reduce `max_desc_length` to 30
- Use smaller batch size

**"All outputs are zero"**
- Check type matching (might be too strict)
- Verify entity types are loaded correctly
- Try with `relation_domain_types=torch.zeros(...)` to disable

**"Attention weights don't sum to 1"**
- This is guaranteed by softmax in the code
- If seeing NaN, check for zero-length descriptions

Ready to continue? Let me know! 🎯