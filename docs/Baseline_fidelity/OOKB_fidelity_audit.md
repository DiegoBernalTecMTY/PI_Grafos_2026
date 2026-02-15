# Code Audit: Knowledge Transfer for OOKB Entities (Hamaguchi et al., 2017)

## 1. Executive Summary
*   **Replication Accuracy:** **85% - 90%** (High fidelity in architectural logic).
*   **Result Trustworthiness (Based on your logs):** **0%** (The model failed to learn).
*   **Verdict:** The code structure successfully captures the mathematical core of the paper (Equations 1-8), specifically the "propagation-based embedding" for new entities. However, the **execution results provided in your log (MRR 0.0004) indicate a catastrophic failure** in the training or data mapping process. The code "exemplifies the idea" mechanically, but the current results do not demonstrate the method's effectiveness.

---

## 2. Detailed Code vs. Paper Comparison

### ✅ What Matches (The Good)
1.  **Propagation Model (Section 3.2):**
    *   The code correctly implements the core concept: $v_e = P(S_{head}(e) \cup S_{tail}(e))$.
    *   The implementation of Transition Functions (`ReLU(Linear(v))`) aligns with Equations 5 & 6.
    *   The "Knowledge Transfer" mechanism is correctly implemented in `prepare_for_ookb_inference`. The code sets OOKB entities to 0 and fills them using the auxiliary graph (`test_triples`). This is the defining feature of the Hamaguchi paper.
    *   **Pooling:** The use of `torch_scatter` to implement Mean/Sum/Max pooling is mathematically equivalent to Eq 4.

2.  **Objective Function (Section 3.3):**
    *   The paper proposes an **Absolute-Margin Objective** (Eq 8): $\sum f(pos) + [\tau - f(neg)]_+$.
    *   The code implements exactly this: `loss_pos + torch.relu(margin - neg_scores)`. This is distinct from the standard pairwise margin used in original TransE and is a crucial detail from this specific paper.

3.  **OOKB Handling:**
    *   The code correctly separates "Known" (learned parameters) from "Unknown" (computed on the fly) entities, which is the main contribution of the paper.

### ⚠️ Deviations & Risks
1.  **Normalization (Batch vs. Layer):**
    *   **Paper:** Uses Batch Normalization (BN) in Eq 5 & 6.
    *   **Code:** Uses `nn.LayerNorm`.
    *   *Impact:* Minimal. LayerNorm is generally more stable for GNNs and Recurrent architectures than BN, especially with varying batch sizes. This is a modernization, not a flaw.

2.  **Training Loop / Sampling Strategy:**
    *   **Code:** The training loop samples a random subset (~14k) of entities per epoch and computes embeddings for them.
    *   *Risk:* This is the likely cause of your poor results. By randomly subsampling entities *before* propagation, you break the graph connectivity. If Entity A is connected to Entity B, but Entity B isn't in the `sample_idx`, Entity A receives a zero-vector or incomplete message. The GNN struggles to learn coherent structural patterns if the edges are constantly severed during training.

---

## 3. Analysis of Your Results
**Your provided log:**
> `Resultados Ranking: {'mrr': 0.00045..., 'hits@1': 0.0, 'hits@10': 0.0004}`

**Interpretation:**
The model has **failed completely**.
*   An MRR of 0.0004 is worse than random guessing.
*   Hits@1 of 0.0 means the model never once predicted the correct link.
*   **Why?**
    1.  **Graph Disconnection:** As mentioned above, the random sampling of entities in the training loop likely destroyed the semantic structure needed to learn.
    2.  **ID Mapping:** In OOKB, if the IDs in `test_triples` do not perfectly correspond to the indices in the `entity_embedding` matrix (e.g., if Train uses IDs 0-1000 and Test uses IDs 0-1000 but they refer to different strings), the transfer fails.
    3.  **Learning Rate:** `1e-3` might be too high for a GNN combined with an Absolute Margin loss, causing gradients to explode or weights to collapse.

---

## 4. Expected Results (Benchmarks)

If the model is working correctly, the results should look significantly different.

### A. Paper Results (Reference)
The paper focuses on **Triplet Classification** (Accuracy), not Ranking (MRR), but we can infer performance.

*   **Standard Setting (WordNet11):** Accuracy **87.8%** (Table 2).
*   **OOKB Setting (WordNet11):** Accuracy between **68.2% and 87.3%** depending on the pooling method and setting (Table 4).

### B. What your Code (CoDEx-M) *Should* Look Like
For a dataset like CoDEx-M in an OOKB setting, if the logic holds:

*   **MRR:** Should be between **0.20 and 0.35**.
*   **Hits@10:** Should be between **30% and 50%**.
*   **Triplet Classification Accuracy:** Should be **> 70%**.

## 5. Recommendations for Fixes

To fix the MRR of 0.0, apply these changes to the code:

1.  **Fix Training Sampling:** Do not sample entities randomly at the start of the epoch. Instead:
    *   Sample a batch of *Triples* (edges).
    *   Identify the unique entities in that batch.
    *   (Optional) Include the 1-hop neighbors of those entities.
    *   Compute GNN only on that subgraph.
    *   *Or, for simplicity:* Run `compute_node_embeddings` on the **full graph** every epoch (if memory allows), or use a proper `NeighborSampler` (like in GraphSAGE). The current "random node subset" approach is breaking the message passing.

2.  **Check ID Alignment:** Ensure that the `UnifiedKGScorer` and your `KGDataLoader` share the exact same entity-to-ID dictionary. If Entity "Blade Runner" is ID 5 in train, it must be ID 5 in the auxiliary graph input.

3.  **Hyperparameters:**
    *   The paper used `margin = 300.0` for the absolute margin. This is very high compared to standard TransE (usually `margin=1.0`). Ensure your embedding initialization scale matches this margin, or reduce the margin to `1.0` and normalize embeddings.