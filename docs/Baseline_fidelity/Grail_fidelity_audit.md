# GraIL Code Adaptation Audit Report

**Date:** February 15, 2026  
**Auditor:** Grok (xAI)  
**Paper:** "Inductive Relation Prediction by Subgraph Reasoning" (Teru et al., ICML 2020)  
**Code:** Practical Fast Version of GraIL (provided)  
**Dataset in run:** FB15k-237 (inductive "NL-25" split)  

---

## 1. Executive Summary

**Architectural Similarity: ~45%**  
The code follows the **high-level pipeline** of GraIL (enclosing subgraph extraction → GNN scoring on subgraph → pooling + head/tail/relation features for scoring). It preserves the core *spirit* of subgraph-based inductive reasoning.

However, it makes **substantial simplifications** to the two most critical technical components that define GraIL's contributions:
- Double-radius distance-based node labeling (replaced by degree + crude indicators).
- Relation-aware GNN with target-relation-conditioned attention (replaced by standard homogeneous GAT).

**Can the results be trusted to exemplify the paper's core ideas?**  
**Partially (60-65% confidence).**  
The code demonstrates a *fast, practical subgraph GNN for KG relation prediction* and can illustrate the general benefit of local subgraph reasoning in an inductive-like setting.

It **does not** faithfully replicate or validate the paper's key claims:
- The theoretical ability to encode path-based logical rules (Theorem 1 relies on distance labeling + target-aware attention).
- The strong inductive bias from structural node roles.
- Proper ranking metrics (AUC-PR, Hits@10).

The reported "accuracy" (>0.5 threshold on positives only) is **not comparable** to the paper's metrics. With 5% subsampling, 5 epochs, and a very large batch size, the run is more of a "smoke test" than a reproduction.

**Recommendation:** This is a useful *pragmatic approximation* for real-world speed (as advertised), but **not** a faithful replication. For research or "exemplifying the paper," major upgrades are needed.

---

## 2. Detailed Comparison

| Component                  | Original GraIL (Paper)                                      | Practical Code                                      | Fidelity | Impact on Core Ideas |
|---------------------------|-------------------------------------------------------------|-----------------------------------------------------|----------|----------------------|
| **Subgraph Extraction**   | k-hop (usually 2-3) enclosing = intersection of neighborhoods + pruning + add target edge | k-hop=1 or 2, intersection + h/t, no explicit pruning, excludes direct edge in pos | High    | Good (core idea preserved) |
| **Node Features / Labeling** | Double-radius: one-hot(dist_to_head, dist_to_tail). Targets uniquely (0,1) & (1,0). Purely structural. | 4-dim: [norm_degree, is_head, is_tail, target_rel_scalar]. Same scalar rel for all nodes in subgraph. | **Low** | **Critical deviation**. Removes structural role encoding that enables rule learning. |
| **GNN Architecture**      | Custom multi-relational (R-GCN style) with basis sharing + **target-relation-conditioned edge attention** + JK-connections (all layers) | Standard **GATConv** (homogeneous, no edge types used despite being extracted) + simple cat of layers + linear proj | **Low** | **Critical**. Loses relational message passing and attention tied to target relation. |
| **Scoring**               | Mean-pool subgraph + h_repr + t_repr + rel_emb → linear | Same high-level concat (graph + h + t + rel) → MLP scorer | High    | Good |
| **Loss & Training**       | Hinge loss (margin=10), full negatives, proper ranking eval | Same hinge, but eval only positives (>0.5 acc) | Medium  | Eval is insufficient |
| **Inductive Setup**       | Train-graph & test-graph have **disjoint entities** | Builds extractor on train+valid+test (full graph). For true disjoint entities this is approximately OK, but degrees/neighbors use full data. | Medium-High | Minor leakage risk |
| **Hyperparameters**       | 3 layers, dim=32, 3-hop, 50 epochs, small batches | 2-3 layers, dim=32, k=1-2, 5 epochs, 5% subsample, batch=2048 | Low     | Heavily optimized for speed, not fidelity |

**Key Insight from Paper (Section 3 & Appendix):**  
The distance labeling + target-conditioned attention is what allows the model to detect specific paths and count rule firings (Theorem 1 & Corollary 1). The code's degree + is_head/is_tail + GAT approximates "some structure around the pair" but does **not** implement the same inductive bias.

---

## 3. What the Paper Results Should Look Like

**Primary Metrics (Inductive Setting):**
- **AUC-PR** (main reported)
- **Hits@10** (ranking vs 50 random negatives)

**Example from Paper (FB15k-237, mean over 5 runs):**

| Version | GraIL AUC-PR | GraIL Hits@10 | RuleN (strongest baseline) |
|---------|--------------|---------------|----------------------------|
| v1      | 84.69        | 64.15         | 75.24 / 49.76              |
| v2      | 90.57        | 81.80         | 88.70 / 77.82              |
| v3      | **91.68**    | **82.83**     | 91.24 / 87.69              |
| v4      | 94.46        | 89.29         | 91.79 / 85.60              |

GraIL consistently beats NeuralLP/DRUM/RuleN by **large margins** (especially on harder versions).

**Transductive setting** (when ensembled with KGE models like RotatE): small but consistent gains (0.6-1.5% relative).

**Your run reported ~96.7% "accuracy"** → **not comparable**. It only measures how often positives score >0.5. The paper cares about **ranking against hard negatives**.

---

## 4. Strengths of the Adaptation (Why It's Still Useful)

- Excellent engineering for speed (adjacency lists, CPU extraction, AMP, large batches, subsampling) → matches the "Practical Fast Version" goal.
- Correct high-level subgraph + GNN + scoring loop.
- Works on real datasets and produces reasonable positive scores quickly.
- Good starting point for production or ablation studies.

---

## 5. Recommendations to Make It Faithful

**Minimal changes for ~75-80% fidelity (still fast):**
1. Implement proper double-radius labeling (BFS distances to head and tail → one-hot or learned embeddings).
2. Switch to R-GCN or RGAT that conditions attention on target relation (or at least use edge types in GAT).
3. Change eval to proper AUC-PR / Hits@10 / MRR with filtered or random negatives.
4. Train longer (20-50 epochs) on more data (at least 30-50%).

**Ideal faithful fast version:**
- Keep k_hop=2.
- Use distance features + lightweight relational GNN (e.g., CompGCN or simple basis-sharing).
- Add proper ranking evaluation.

---

## 6. Final Verdict

**Closeness: 45/100**  
**Trustworthiness for exemplifying paper's core concepts: 60/100** (good for "subgraph GNN works", poor for "GraIL's specific inductive bias and theoretical power work").

This is a **pragmatic, production-oriented approximation** — exactly as the code comments advertise ("Practical optimizations for real-world training time"). It is **not** a scientific replication or faithful implementation of the 2020 GraIL model.

If you want help upgrading the code toward higher fidelity (distance labeling + relational GNN + proper metrics) while keeping it reasonably fast, just say the word and I'll provide patches.

---

**End of Audit**