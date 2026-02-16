# INGRAM Implementation Audit Report

## Executive Summary

**Overall Match Percentage: 55-60%**

The provided code implements the core concepts of the INGRAM paper but contains several critical deviations from the original methodology that significantly impact performance. The implementation achieves only ~8-10% of the expected MRR performance compared to the paper's reported results.

---

## 1. Methodology Comparison

### 1.1 Relation Graph Construction

| Component | Paper Specification | Code Implementation | Match |
|-----------|---------------------|---------------------|-------|
| Head matrix E^h | E^h[i,j] = frequency of v_i as head of r_j | ✓ Correctly implemented | ✓ |
| Tail matrix E^t | E^t[i,j] = frequency of v_i as tail of r_j | ✓ Correctly implemented | ✓ |
| Degree normalization | D_h[i,i] = Σ_j E^h[i,j], D_h^-2 | ✓ Correctly implemented | ✓ |
| Affinity matrix | A = E^h^T D_h^-2 E^h + E^t^T D_t^-2 E^t | ✓ Correctly implemented | ✓ |
| Self-loops | A contains self-loops (N_i includes r_i) | ✓ Added via identity matrix | ✓ |

**Verdict: 100% Match** - The relation graph construction is correctly implemented using vectorized operations.

---

### 1.2 Relation-Level Aggregation

| Component | Paper Specification | Code Implementation | Match |
|-----------|---------------------|---------------------|-------|
| Initial projection | z_i^(0) = H x_i | ✓ Uses `relation_feature_proj` | ✓ |
| Attention formula | α_ij = softmax(y σ(P[z_i ∥ z_j]) + c_s(i,j)) | ⚠️ Modified attention | Partial |
| Binning strategy | s(i,j) = ⌈rank(a_ij) × B / nnz(A)⌉ | ⚠️ Percentile-based, different | ❌ |
| Weight matrix W^(l) | Applied inside aggregation | ✓ Present | ✓ |
| LeakyReLU activation | σ = LeakyReLU | ⚡ Uses LayerNorm + residual | ❌ |
| Multi-head attention | K ∈ {8, 16} heads | ✓ K=16 heads | ✓ |

**Critical Issues:**

1. **Binning Strategy Mismatch:**
   - **Paper:** Uses rank-based binning where each relation pair is assigned to bin `s(i,j) = ⌈rank(a_ij) × B / nnz(A)⌉`
   - **Code:** Uses percentile-based binning which divides elements by percentiles
   - **Impact:** The paper's binning ensures that bins have roughly equal number of elements, while the code's implementation may have uneven distributions

2. **LayerNorm Addition:**
   - **Paper:** Uses LeakyReLU activation only: `z_i^(l+1) = σ(Σ α_ij W^(l) z_j)`
   - **Code:** Adds LayerNorm and residual connection: `return self.layer_norm(z_aggregated + z)`
   - **Impact:** Changes the learning dynamics significantly

---

### 1.3 Entity-Level Aggregation

| Component | Paper Specification | Code Implementation | Match |
|-----------|---------------------|---------------------|-------|
| Initial projection | h_i^(0) = H_c x̄_i | ✓ Uses `entity_feature_proj` | ✓ |
| Neighbor definition | N̄_i = {v_j | (v_j, r_k, v_i) ∈ F} | ✓ Correct | ✓ |
| Relation mean vector | z̄_i = Σ_{v_j ∈ N̄_i} Σ_{r_k ∈ R_ji} z_k^(L) / |R_ji| | ⚠️ Simplified | Partial |
| Attention coefficient | β_ijk with P̄, ȳ_b | ⚠️ Simplified for speed | ❌ |
| Self-loop attention | β_ii computed separately | ⚠️ Merged with neighbors | ❌ |
| Multi-head attention | K̄ ∈ {8, 16} heads | ✓ K̄=16 heads | ✓ |

**Critical Issues:**

1. **Attention Computation Simplification:**
   - **Paper:** Computes separate attention for self-loop (β_ii) and neighbors (β_ijk)
   - **Code:** Uses simplified mean pooling for speed, not implementing full attention mechanism
   - **Impact:** Major deviation from the paper's attention mechanism

2. **LayerNorm Addition:**
   - Same issue as relation-level aggregation

---

### 1.4 Scoring Function

| Component | Paper Specification | Code Implementation | Match |
|-----------|---------------------|---------------------|-------|
| Formula | f(v_i, r_k, v_j) = h_i^T diag(W z_k) h_j | `(h_i * Wz_k * h_j).sum()` | ✓ |
| Weight matrix | W ∈ R^(d_b × d) converts relation dim | ✓ `scoring_weight` | ✓ |

**Verdict: 100% Match** - The scoring function is correctly implemented and equivalent to the paper's DistMult variant.

---

### 1.5 Training Procedure

| Component | Paper Specification | Code Implementation | Match |
|-----------|---------------------|---------------------|-------|
| Loss function | Margin-based ranking loss | ✓ `F.relu(margin - pos + neg).mean()` | ✓ |
| Negative sampling | Corrupt head or tail | ✓ Implemented | ✓ |
| Optimizer | Adam with mini-batch | ✓ Adam optimizer | ✓ |
| Dynamic split | Re-split F_tr and T_tr every epoch (3:1 ratio) | ❌ **NOT IMPLEMENTED** | ❌ |
| Random re-initialization | Re-initialize features every epoch | ❌ **NOT IMPLEMENTED** | ❌ |
| Epochs | 10,000 epochs | 1,000 epochs | ⚠️ Reduced |
| Validation | Every 200 epochs | Every 5 epochs | ⚠️ Different |

**Critical Missing Components:**

1. **Dynamic Split Strategy:**
   - **Paper:** "For every epoch, we randomly re-split F_tr and T_tr with the minimal constraint that F_tr includes the minimum spanning tree and covers all relations"
   - **Code:** Uses fixed split with 75%/25% division once
   - **Impact:** This is a MAJOR deviation. The paper explicitly states: "This dynamic split and re-initialization strategy allow INGRAM to robustly learn the model parameters, which makes the model more easily generalizable to an inference graph"

2. **Random Re-initialization:**
   - **Paper:** "At the beginning of each epoch, we initialize all feature vectors using Glorot initialization"
   - **Code:** Only initializes once at the start
   - **Impact:** The paper learns to aggregate from random features, making the model truly inductive

---

## 2. Hyperparameter Comparison

| Hyperparameter | Paper Value | Code Value | Match |
|----------------|-------------|------------|-------|
| Embedding dim (d) | 32 | 32 | ✓ |
| Relation hidden dim (d') | {32, 64, 128, 256} | 64 | ✓ |
| Entity hidden dim (d̄') | {128, 256} | 128 | ✓ |
| Relation layers (L) | {1, 2, 3} | 2 | ✓ |
| Entity layers (L̄) | {2, 3, 4} | 3 | ✓ |
| Relation heads (K) | {8, 16} | 16 | ✓ |
| Entity heads (K̄) | {8, 16} | 16 | ✓ |
| Number of bins (B) | {1, 5, 10}, best=10 | 10 | ✓ |
| Margin (γ) | {1.0, 1.5, 2.0, 2.5} | 1.5 | ✓ |
| Learning rate | {0.0005, 0.001} | 0.001 | ✓ |
| Negative samples | 10 | 10 | ✓ |
| Epochs | 10,000 | 1,000 | ⚠️ |
| Dropout | Not specified | 0.1 | - |

---

## 3. Results Comparison

### 3.1 Your Implementation Results (CoDEx-M, NL-25)

| Metric | Your Result | Expected (Paper NL-25) | Gap |
|--------|-------------|------------------------|-----|
| **MRR** | 0.0274 | 0.334 | **-91.8%** |
| **Hits@1** | 0.0036 | 0.241 | **-98.5%** |
| **Hits@3** | 0.0123 | ~0.28 | **-95.6%** |
| **Hits@10** | 0.0621 | 0.501 | **-87.6%** |
| **MR** | 560.40 | 90.1 | **6x worse** |
| **AUC** | 0.7996 | N/A | - |

### 3.2 Paper's Expected Results by Dataset

| Dataset | MRR | Hits@10 | Hits@1 |
|---------|-----|---------|--------|
| NL-100 | 0.309 | 0.506 | 0.212 |
| NL-75 | 0.261 | 0.464 | 0.167 |
| NL-50 | 0.281 | 0.453 | 0.193 |
| NL-25 | 0.334 | 0.501 | 0.241 |
| WK-100 | 0.107 | 0.169 | 0.072 |
| FB-100 | 0.223 | 0.371 | 0.146 |

---

## 4. Root Cause Analysis

### 4.1 Critical Issues (Must Fix)

1. **Missing Dynamic Split Strategy (CRITICAL)**
   - The paper explicitly states this is essential for generalization
   - Without it, the model memorizes specific embeddings rather than learning aggregation patterns
   - **Fix:** Re-split F_tr/T_tr every epoch with 3:1 ratio, ensuring minimum spanning tree coverage

2. **Missing Feature Re-initialization (CRITICAL)**
   - Paper: "INGRAM learns how to compute embedding vectors using random feature vectors"
   - This enables true inductive capability
   - **Fix:** Call `init_features()` at the start of each epoch

3. **Binning Strategy Incorrect**
   - Paper uses rank-based binning with ceiling function
   - Code uses percentile-based approach
   - **Fix:** Implement: `s(i,j) = ceil(rank(a_ij) * B / nnz(A))`

4. **Entity-level Attention Oversimplified**
   - Code uses mean pooling for speed
   - Paper uses full attention with separate self-loop handling
   - **Fix:** Implement proper attention with β_ii and β_ijk formulas

### 4.2 Moderate Issues

1. **LayerNorm Added**
   - Not in original paper, changes learning dynamics
   - May or may not help performance

2. **Reduced Training Epochs**
   - Paper: 10,000 epochs with validation every 200
   - Code: 1,000 epochs with validation every 5
   - Could be insufficient for convergence

### 4.3 Minor Issues

1. **Graph Caching**
   - Code caches relation graph
   - Paper recomputes every time (necessary for dynamic setting)

---

## 5. Trust Assessment

### Can Results Be Trusted to Exemplify Core Ideas?

**Answer: NO - With significant caveats**

The implementation correctly captures:
- ✓ The concept of relation graphs for defining relation neighborhoods
- ✓ The overall two-level aggregation architecture (relation → entity)
- ✓ The DistMult-variant scoring function
- ✓ Multi-head attention mechanism concept
- ✓ The binning concept for incorporating affinity in attention

However, it fails to capture:
- ❌ The critical training regime (dynamic split + re-initialization)
- ❌ The specific attention formulas for both levels
- ❌ The true inductive learning capability

### What Paper Results Should Look Like

For a **properly implemented INGRAM** on a similar inductive dataset:

| Metric | Expected Range |
|--------|----------------|
| MRR | 0.20 - 0.35 |
| Hits@10 | 0.40 - 0.55 |
| Hits@1 | 0.15 - 0.25 |
| MR | 50 - 150 |

Your current results (MRR=0.027) are **an order of magnitude worse** than expected, indicating fundamental implementation issues.

---

## 6. Recommendations

### Priority 1 - Critical Fixes

```python
# Fix 1: Dynamic split in training loop
for epoch in range(epochs):
    # Re-split every epoch
    perm = torch.randperm(len(train_data))
    split_point = int(0.75 * len(train_data))
    F_tr = train_data[perm[:split_point]]
    T_tr = train_data[perm[split_point:]]
    
    # Re-initialize features every epoch
    entity_features, relation_features = model.init_features(device)
    
    # Forward pass with new features
    entity_emb, relation_emb = model(F_tr, entity_features, relation_features)
```

### Priority 2 - Architecture Fixes

```python
# Fix 2: Proper binning strategy
def compute_bins_paper(A, num_bins):
    """Paper's binning: s(i,j) = ceil(rank(a_ij) * B / nnz(A))"""
    flat_A = A.flatten()
    nonzero_mask = flat_A > 0
    nonzero_vals = flat_A[nonzero_mask]
    
    # Get ranks (1-indexed, descending order)
    sorted_indices = torch.argsort(nonzero_vals, descending=True)
    ranks = torch.zeros_like(nonzero_vals, dtype=torch.long)
    ranks[sorted_indices] = torch.arange(1, len(nonzero_vals) + 1, device=A.device)
    
    # Compute bin indices
    bins = torch.ceil(ranks.float() * num_bins / len(nonzero_vals)).long()
    bins = torch.clamp(bins, 1, num_bins)
    
    # Map back to matrix
    result = torch.zeros_like(A, dtype=torch.long)
    result[nonzero_mask] = bins
    return result
```

### Priority 3 - Training Configuration

- Increase epochs to 10,000
- Validate every 200 epochs
- Remove graph caching during training
- Consider removing LayerNorm additions

---

## 7. Component-by-Component Match Summary

| Component | Match % | Notes |
|-----------|---------|-------|
| Relation Graph Construction | 100% | Fully correct |
| Relation-level Aggregation | 65% | Binning wrong, LayerNorm added |
| Entity-level Aggregation | 40% | Attention oversimplified |
| Scoring Function | 100% | Correctly implemented |
| Training Procedure | 30% | Missing critical dynamic split and re-initialization |
| Hyperparameters | 85% | Mostly correct, epochs reduced |
| **Overall** | **55-60%** | Critical training regime missing |

---

## 8. Conclusion

This implementation captures the **architectural concepts** of INGRAM but fails to implement the **training methodology** that makes INGRAM truly inductive. The missing dynamic split and feature re-initialization strategies are not minor optimizations - they are **fundamental** to INGRAM's ability to generalize to new entities and relations.

**Key Takeaway:** The paper's main innovation is not just the architecture, but the training regime that forces the model to learn aggregation patterns from random features. Without this, the model degenerates to a standard (poorly performing) knowledge graph embedding model.

---

*Audit Date: Generated by AI Analysis*
*Paper: INGRAM: Inductive Knowledge Graph Embedding via Relation Graphs (Lee et al., ICML 2023)*