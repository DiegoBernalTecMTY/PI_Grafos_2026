# IKGE Implementation Fix - Product Requirements Document (PRD)

**Target Hardware:** RTX 5080 (16GB VRAM)  
**Training Time Budget:** 2 hours maximum  
**Goal:** Achieve paper-level accuracy on Codex-M dataset

---

## 📋 Executive Summary

Transform the current broken IKGE implementation into a paper-faithful version that can train within 2 hours on RTX 5080 and produce valid benchmark results.

---

## 🎯 Critical Components to Fix (Priority Order)

### PHASE 1: Core Architecture Fixes (CRITICAL - Week 1)

#### 1.1 Line Graph Construction ⚠️ HIGHEST PRIORITY
**Current State:** Uses entity-to-entity graph  
**Required State:** Fact-to-fact line graph

**Tasks:**
- [ ] Implement `create_line_graph()` function
  - Input: Knowledge graph triples `[(h, r, t), ...]`
  - Output: Line graph where nodes = facts, edges = adjacent facts
  - Two facts are adjacent if they share an entity
- [ ] Store line graph as `fact_edge_index` tensor
- [ ] Create `fact_id_to_triple` mapping for lookup
- [ ] Verify line graph connectivity (should have way more edges than entity graph)

**Acceptance Criteria:**
```python
# Original: 206,205 triples in Codex-M
# Entity graph: ~17,050 nodes, ~412,410 edges (bidirectional)
# Line graph should have:
#   - 206,205 nodes (one per fact)
#   - ~2-5M edges (facts sharing entities)
assert line_graph.num_nodes == len(train_triples)
assert line_graph.num_edges > 2_000_000
```

**Estimated Time:** 8 hours

---

#### 1.2 Fact Feature Extraction Module ⚠️ CRITICAL
**Current State:** Random features + simple linear projection  
**Required State:** Attention-based CNN on textual descriptions

**Tasks:**
- [ ] **Word Encoding Layer**
  - Load pre-trained word embeddings (GloVe 300d or Word2Vec)
  - Create word vocabulary from entity descriptions
  - Implement word lookup and embedding matrix
  
- [ ] **Attention-Based Convolution (Paper Section 5.1.2)**
  - Implement 2 × 1D CNN layers for entity descriptions
  - Filter width: 3, output channels: embedding_dim
  - Attention mechanism: 
    - Attend entity description to: relation name, type constraints, other entity name
    - Implement equation (1): `A = tanh((D'_h)^T * W_a * cat(w_r, U_r, U_t))`
    - Column-wise max pooling (equation 2)
    - Weighted average (equation 3)
  
- [ ] **Type Matching (Paper Section 5.1.3)**
  - Load entity types and relation type constraints from enriched CSVs
  - Implement equation (5): element-wise multiplication + sum
  - Filter out invalid facts (h-r and t-r type mismatches)
  
- [ ] **Fact Feature Combiner**
  - Extract separate features for head and tail: `e_h, e_t`
  - Concatenate and project: `f = W_p * [e_h; e_t] + b_p`
  - Output: initial fact embedding `f ∈ R^d`

**Architecture:**
```python
class FactFeatureExtractor(nn.Module):
    def __init__(self, vocab_size, word_emb_dim=300, fact_emb_dim=128):
        # Word embedding layer (frozen pre-trained)
        self.word_embeddings = nn.Embedding.from_pretrained(glove_vectors)
        
        # Two 1D convolutions for description
        self.conv1 = nn.Conv1d(word_emb_dim, fact_emb_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(fact_emb_dim, fact_emb_dim, kernel_size=3, padding=1)
        
        # Attention mechanism
        self.attention_W = nn.Linear(fact_emb_dim, fact_emb_dim)
        
        # Final projection
        self.fact_projection = nn.Linear(2 * fact_emb_dim, fact_emb_dim)
    
    def forward(self, h_desc, r_name, t_desc, r_type_constraints, h_types, t_types):
        # Paper Figure 3 implementation
        pass
```

**Acceptance Criteria:**
- [ ] Uses actual entity descriptions from enriched CSV
- [ ] Attention weights sum to 1.0 for each entity
- [ ] Type matching filters ~5-10% of invalid facts
- [ ] Output fact embeddings capture relation-specific entity features

**Estimated Time:** 16 hours

---

#### 1.3 Attentive Feature Aggregation ⚠️ CRITICAL
**Current State:** Aggregates entity neighbors with mean pooling  
**Required State:** Hierarchical fact aggregation with learned attention

**Tasks:**
- [ ] **Fix Aggregation Target**
  - Change from aggregating entity representations to fact representations
  - Use line graph `fact_edge_index` instead of entity graph
  
- [ ] **Implement Attention Mechanism (Paper Section 5.2.1)**
  - Attention score: `att_score_v = f_v^T * W_a^{k+1} * f_u` (equation 8)
  - Attention weights: softmax over neighbors (equation 7)
  - Weighted sum: `h_{N(f_u)} = tanh(Σ a_v * f_v)` (equation 9)
  
- [ ] **Hierarchical Aggregation**
  - K layers of aggregation (K=2 for Codex-M, K=3 for larger datasets)
  - Each layer: aggregate neighbors → combine with current fact → update
  - Update rule: `f̃_u = h_{N(f_u)} + f_u` (equation 10)
  
- [ ] **Learnable Parameters per Depth**
  - Separate `W_a^k` for each aggregation layer k
  - Total params: K × (embedding_dim × embedding_dim)

**Architecture:**
```python
class AttentiveAggregator(nn.Module):
    def __init__(self, embedding_dim, num_layers=2):
        self.num_layers = num_layers
        self.attention_weights = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(num_layers)
        ])
    
    def forward(self, fact_embeddings, fact_edge_index, target_fact_ids):
        # Paper Section 5.2 implementation
        z = fact_embeddings.clone()
        
        for k in range(self.num_layers):
            # Aggregate K-hop neighbors
            z = self.aggregate_layer(z, fact_edge_index, k)
        
        return z[target_fact_ids]
```

**Acceptance Criteria:**
- [ ] Aggregates from line graph (fact-to-fact)
- [ ] Uses learned attention weights (not uniform averaging)
- [ ] Hierarchical: depth K aggregation layers
- [ ] Gradients flow through all K layers

**Estimated Time:** 12 hours

---

### PHASE 2: Data Pipeline (HIGH PRIORITY - Week 1)

#### 2.1 Enhanced Data Loader
**Tasks:**
- [ ] Load enriched entity descriptions from CSV
- [ ] Load relation metadata from CSV
- [ ] Parse entity types and relation type constraints
- [ ] Build word vocabulary from all descriptions
- [ ] Create entity description tensors (padded sequences)
- [ ] Map pre-trained word embeddings (GloVe/Word2Vec)

**Data Structure:**
```python
class EnrichedKGDataLoader:
    def __init__(self, dataset_path, enriched_path):
        # Load triples
        self.train_triples = self.load_triples('train.txt')
        
        # Load enriched metadata
        self.entity_df = pd.read_csv(f'{enriched_path}/entity_descriptions.csv')
        self.relation_df = pd.read_csv(f'{enriched_path}/relation_info.csv')
        
        # Build vocabulary
        self.vocab = self.build_vocab(self.entity_df['description'])
        
        # Load pre-trained embeddings
        self.word_embeddings = self.load_glove('glove.6B.300d.txt')
        
        # Prepare description tensors
        self.entity_desc_tensors = self.tokenize_descriptions(
            self.entity_df['description']
        )
```

**Acceptance Criteria:**
- [ ] Returns entity descriptions as token sequences
- [ ] Returns relation names and type constraints
- [ ] Handles missing/empty descriptions gracefully
- [ ] Word embeddings loaded correctly (check with known words)

**Estimated Time:** 6 hours

---

#### 2.2 Dataset Partitioning for 2-Hour Training
**Current:** Full Codex-M (206k triples)  
**Required:** Subset that trains in 2 hours

**Strategy:**
```python
# Option A: Sample uniformly
train_subset = random.sample(train_triples, k=50_000)  # ~25% of data

# Option B: Sample by relation (better for evaluation)
train_subset = sample_per_relation(train_triples, samples_per_rel=1000)

# Option C: Use validation set size (paper uses small valid sets)
# Codex-M probably has ~10-20k validation triples
train_subset = train_triples[:20_000]
```

**Tasks:**
- [ ] Implement sampling function
- [ ] Ensure all relations represented in subset
- [ ] Verify subset still creates valid line graph
- [ ] Document subset statistics

**Acceptance Criteria:**
- [ ] Subset size: 20-50k triples (adjust based on speed tests)
- [ ] All 51 relations appear at least 100 times
- [ ] Training completes in < 2 hours on RTX 5080

**Estimated Time:** 3 hours

---

### PHASE 3: Training Infrastructure (HIGH PRIORITY - Week 1-2)

#### 3.1 Optimized Training Loop
**Tasks:**
- [ ] **Batch Size Optimization**
  - Start with batch_size = 2048 (RTX 5080 can handle it)
  - Monitor GPU memory usage
  - Adjust based on OOM errors
  
- [ ] **Mixed Precision Training (FP16)**
  ```python
  from torch.cuda.amp import autocast, GradScaler
  
  scaler = GradScaler()
  
  for batch in train_loader:
      with autocast():
          scores = model(batch)
          loss = criterion(scores, labels)
      
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```
  - Reduces memory usage by ~40%
  - Speeds up training by ~2x
  
- [ ] **Gradient Accumulation** (if still OOM)
  ```python
  accumulation_steps = 4
  for i, batch in enumerate(train_loader):
      loss = loss / accumulation_steps
      loss.backward()
      
      if (i + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
  ```

- [ ] **DataLoader Optimizations**
  - `num_workers=4` for parallel data loading
  - `pin_memory=True` for faster GPU transfer
  - `prefetch_factor=2` to preload batches

**Acceptance Criteria:**
- [ ] GPU utilization > 85%
- [ ] Training speed: > 1000 triples/second
- [ ] No OOM errors with chosen batch size
- [ ] Full epoch completes in < 15 minutes

**Estimated Time:** 4 hours

---

#### 3.2 Model Checkpointing & Resume
**Tasks:**
- [ ] Save checkpoint every N epochs
- [ ] Save on keyboard interrupt (Ctrl+C)
- [ ] Save best model based on validation metric
- [ ] Resume from checkpoint with all state

**Implementation:**
```python
def save_checkpoint(model, optimizer, epoch, metrics, path):
    checkpoint = {
        # Model state
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        
        # Training state
        'epoch': epoch,
        'train_loss': metrics['train_loss'],
        'val_metrics': metrics,
        
        # Hyperparameters
        'hyperparameters': {
            'embedding_dim': model.embedding_dim,
            'num_layers': model.num_agg_layers,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'batch_size': BATCH_SIZE,
        },
        
        # Data info
        'num_entities': model.num_entities,
        'num_relations': model.num_relations,
        'vocab_size': len(model.vocab),
        
        # Random states for reproducibility
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }
    
    torch.save(checkpoint, path)
    print(f"💾 Checkpoint saved: {path}")

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore random states
    torch.set_rng_state(checkpoint['torch_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['python_rng_state'])
    
    return checkpoint['epoch'], checkpoint['val_metrics']
```

**Checkpoint Structure:**
```
checkpoints/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── checkpoint_best.pt          # Best validation MRR
└── checkpoint_latest.pt         # Most recent (for resume)
```

**Acceptance Criteria:**
- [ ] Can resume training from any checkpoint
- [ ] Metrics match exactly after resume
- [ ] Checkpoints include all necessary state
- [ ] Automatic save on Ctrl+C

**Estimated Time:** 3 hours

---

#### 3.3 Hyperparameter Configuration
**Optimized for RTX 5080 + 2-hour budget:**

```python
CONFIG = {
    # Model architecture
    'embedding_dim': 128,           # Paper uses 128-256
    'word_emb_dim': 300,           # GloVe dimension
    'num_agg_layers': 2,           # K=2 for speed (paper uses 2-3)
    'dropout_rate': 0.2,
    
    # Training
    'batch_size': 2048,            # Large batch for RTX 5080
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'max_epochs': 20,              # ~6 min per epoch = 2 hours
    'gradient_clip': 1.0,
    
    # Data
    'train_subset_size': 50_000,   # 25% of Codex-M
    'negative_samples': 1,         # 1 negative per positive
    'max_desc_length': 50,         # Truncate long descriptions
    
    # Optimization
    'use_mixed_precision': True,   # FP16 for speed
    'num_workers': 4,
    'pin_memory': True,
    
    # Checkpointing
    'save_every': 5,               # Save every 5 epochs
    'checkpoint_dir': './checkpoints',
}
```

**Estimated Time:** 2 hours (testing different configs)

---

### PHASE 4: Evaluation & Reporting (MEDIUM PRIORITY - Week 2)

#### 4.1 Evaluation Metrics (Already exists in code)
**Current State:** `UnifiedKGScorer` class exists ✅

**Required Verification:**
- [ ] Verify MRR calculation matches paper
- [ ] Verify Hits@K calculation
- [ ] Add Mean Rank (MR)
- [ ] Add classification metrics (accuracy, F1, AUC)

**Note:** The existing `UnifiedKGScorer` looks good, just need to verify it works with new architecture.

**Estimated Time:** 2 hours

---

#### 4.2 Enhanced PDF Report Generation
**Current State:** Basic report exists  
**Required State:** Comprehensive evaluation report

**Enhancements:**
- [ ] **Training Curves**
  - Loss over epochs
  - Validation MRR over epochs
  - Learning rate schedule
  
- [ ] **Performance Breakdown**
  - Results by relation type
  - Results by entity frequency (head/tail)
  - Results by fact type (in-KG vs out-of-KG)
  
- [ ] **Model Information**
  - Architecture summary
  - Hyperparameters used
  - Training time
  - GPU memory usage
  
- [ ] **Comparison Table**
  - Paper's reported results
  - This implementation's results
  - Gap analysis

**Report Structure:**
```
Page 1: Executive Summary
  - Model: IKGE (Codex-M)
  - Training Time: 1.5 hours
  - Best MRR: 0.XX
  - vs Paper: 0.XX (gap: XX%)

Page 2: Training Curves
  - Loss curve
  - Validation metrics over time
  
Page 3: Classification Metrics
  - ROC curve
  - Precision-Recall curve
  - Confusion matrix
  
Page 4: Ranking Metrics
  - Hits@K bar chart
  - MRR by relation type
  
Page 5: Architecture & Hyperparameters
  - Model diagram
  - Config table
  - Resource usage
```

**Estimated Time:** 4 hours

---

### PHASE 5: Code Quality & Documentation (LOW PRIORITY - Week 2)

#### 5.1 Code Organization
**Tasks:**
- [ ] Split into modules:
  ```
  ikge/
  ├── data/
  │   ├── loader.py          # EnrichedKGDataLoader
  │   └── line_graph.py      # Line graph construction
  ├── models/
  │   ├── fact_extractor.py  # FactFeatureExtractor
  │   ├── aggregator.py      # AttentiveAggregator
  │   └── ikge.py           # Main IKGE model
  ├── training/
  │   ├── trainer.py         # Training loop
  │   └── checkpoint.py      # Checkpointing utils
  ├── evaluation/
  │   └── scorer.py          # UnifiedKGScorer
  └── utils/
      ├── config.py          # Configuration
      └── visualization.py   # Plotting utils
  ```

- [ ] Add docstrings to all functions
- [ ] Add type hints
- [ ] Add inline comments for complex logic

**Estimated Time:** 6 hours

---

#### 5.2 Testing & Validation
**Tasks:**
- [ ] Unit tests for line graph construction
- [ ] Unit tests for fact feature extraction
- [ ] Integration test: small dataset end-to-end
- [ ] Validate attention weights sum to 1
- [ ] Validate type matching logic

**Estimated Time:** 4 hours

---

## 📊 Time Estimation Summary

| Phase | Component | Hours | Priority |
|-------|-----------|-------|----------|
| 1.1 | Line Graph Construction | 8 | 🔴 CRITICAL |
| 1.2 | Fact Feature Extraction | 16 | 🔴 CRITICAL |
| 1.3 | Attentive Aggregation | 12 | 🔴 CRITICAL |
| 2.1 | Enhanced Data Loader | 6 | 🟠 HIGH |
| 2.2 | Dataset Partitioning | 3 | 🟠 HIGH |
| 3.1 | Training Optimization | 4 | 🟠 HIGH |
| 3.2 | Checkpointing | 3 | 🟠 HIGH |
| 3.3 | Hyperparameter Tuning | 2 | 🟠 HIGH |
| 4.1 | Evaluation Metrics | 2 | 🟡 MEDIUM |
| 4.2 | PDF Report | 4 | 🟡 MEDIUM |
| 5.1 | Code Organization | 6 | 🟢 LOW |
| 5.2 | Testing | 4 | 🟢 LOW |
| **TOTAL** | | **70 hours** | |

**Critical Path (Minimum Viable):** ~50 hours (Phases 1-3 only)  
**Full Implementation:** ~70 hours

---

## 🎯 Success Criteria

### Minimum Viable (MVP):
- [ ] Model trains without errors
- [ ] Completes training in < 2 hours
- [ ] Produces evaluation metrics (MRR, Hits@K)
- [ ] Can save and load checkpoints

### Good:
- [ ] MRR within 50% of paper results
- [ ] All three core components correctly implemented
- [ ] Comprehensive evaluation report

### Excellent:
- [ ] MRR within 80% of paper results
- [ ] Training curves look reasonable
- [ ] Code is modular and well-documented

---

## 📈 Expected Results Timeline

**Week 1:** Core architecture working, basic training  
**Week 2:** Optimized training, full evaluation, documentation

**Paper Results (DBpedia50k+):**
- Head Entity Prediction: MR=104, Hits@10=54%, MRR=0.52
- Tail Entity Prediction: MR=31, Hits@10=78%, MRR=0.61

**Expected Results (Our Implementation, Codex-M subset):**
- First attempt: MRR ~0.15-0.25 (debugging phase)
- After fixes: MRR ~0.30-0.40 (80% of paper on subset)
- Optimized: MRR ~0.40-0.50 (close to paper)

---

## 🔧 Development Workflow

### Phase 1: Week 1 (Critical Path)
**Day 1-2:** Line graph construction + testing  
**Day 3-5:** Fact feature extraction module  
**Day 6-7:** Attentive aggregation module

### Phase 2: Week 1 (Data & Training)
**Day 6-7:** Enhanced data loader + partitioning  
**Day 7:** Training loop optimization + checkpointing

### Phase 3: Week 2 (Polish & Evaluate)
**Day 8-9:** Full training runs + hyperparameter tuning  
**Day 10:** Evaluation + report generation  
**Day 11-12:** Code cleanup + documentation (optional)

---

## 🚨 Risk Mitigation

### Risk 1: OOM Errors
**Mitigation:**
- Start with batch_size=1024, increase gradually
- Use mixed precision (FP16)
- Implement gradient accumulation
- Reduce num_agg_layers to 1 if desperate

### Risk 2: Training Too Slow
**Mitigation:**
- Reduce train_subset_size to 20k
- Reduce max_desc_length to 30
- Use only 1 aggregation layer (K=1)
- Skip type matching initially

### Risk 3: Poor Results
**Mitigation:**
- Start with paper's hyperparameters exactly
- Verify each component with unit tests
- Compare intermediate outputs with paper's examples
- Train on DBpedia50k+ (paper's dataset) for validation

---

## 📝 Deliverables Checklist

### Code:
- [ ] `ikge_v2.py` - Fixed implementation
- [ ] `enrich_codex_m.py` - Data enrichment script
- [ ] `train.py` - Training script
- [ ] `evaluate.py` - Evaluation script
- [ ] `config.yaml` - Hyperparameters

### Checkpoints:
- [ ] `checkpoint_best.pt` - Best model
- [ ] `checkpoint_latest.pt` - Latest model
- [ ] `training_history.json` - Metrics log

### Reports:
- [ ] `evaluation_report.pdf` - Full results
- [ ] `training_log.txt` - Training logs
- [ ] `README.md` - Usage instructions

---

## 💡 Quick Wins (Do First)

1. **Line Graph Construction** (8h) - Biggest architectural fix
2. **Load Enriched Data** (3h) - Actually use entity descriptions
3. **Training Optimization** (4h) - Make it fast enough
4. **Checkpointing** (3h) - Don't lose progress

These 4 tasks (18 hours) will give you a trainable model that actually resembles IKGE.

---

## ❓ Open Questions

1. **Do you have pre-trained word embeddings (GloVe)?**
   - If not, I'll add download script

2. **Preferred report format?**
   - Current code has UnifiedKGScorer - is this the class you mentioned?

3. **Dataset size preference?**
   - Full Codex-M (slower, better results)
   - Or 25% subset (faster, good enough results)

4. **Priority order?**
   - Should we focus on correctness (all paper details) or speed (get it working fast)?

---

## 🎬 Ready to Start?

**Recommended Starting Point:**
Phase 1.1 - Line Graph Construction

This is the biggest architectural change and everything else depends on it.

**Let me know:**
1. Do you have the enriched Codex-M data ready? (entity_descriptions.csv)
2. Do you want me to start with line graph construction code?
3. Any questions about the PRD?