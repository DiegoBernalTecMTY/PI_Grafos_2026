"""
Fact Feature Extraction Module for IKGE
========================================

Implements Section 5.1 of the paper:
- Word Encoding (Section 5.1.1)
- Attention-Based Convolution (Section 5.1.2)  
- Type Matching (Section 5.1.3)

This module extracts relation-specific entity features from textual descriptions.

Paper Reference: "Open-world knowledge graph completion for unseen entities 
                  and relations via attentive feature aggregation"
                  Figure 3 - Fact Feature Information Extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class FactFeatureExtractor(nn.Module):
    """
    Extracts fact features from entity descriptions and relation information.
    
    Key Innovation: Instead of fixed entity embeddings, generates relation-specific
    entity features by attending to relevant parts of entity descriptions.
    
    Example:
        For fact (Harvard, locatedIn, Massachusetts):
        - Harvard's features focus on "Cambridge", "Massachusetts" from its description
        - Massachusetts's features focus on location-related words
        
    Architecture:
        1. Word Encoding: Convert descriptions to word embeddings (GloVe)
        2. Attention-Based Convolution: Extract entity features with attention
        3. Type Matching: Validate h-r and t-r type constraints
        4. Fact Feature Combination: Concatenate head and tail features
    """
    
    def __init__(self,
                 word_embedding_matrix: torch.Tensor,
                 word_embedding_dim: int = 300,
                 fact_embedding_dim: int = 300,
                 conv_channels: int = 300,
                 num_types: int = 100,
                 kernel_size: int = 3,
                 dropout: float = 0.25,
                 device: str = 'cuda'):
        """
        Args:
            word_embedding_matrix: Pre-trained word embeddings (vocab_size, word_emb_dim)
            word_embedding_dim: Dimension of word embeddings (300)
            fact_embedding_dim: Output fact embedding dim d (must equal word_embedding_dim)
            conv_channels: CNN output channels (= d, must equal word_embedding_dim)
            num_types: Size of the type vocabulary for type-matching vectors
            kernel_size: CNN kernel width (paper uses 3)
            dropout: Dropout rate (paper: 0.25)
            device: 'cuda' or 'cpu'
        """
        super().__init__()

        # Paper uses a single shared dimension d throughout (Section 5.2.4).  All of
        # word_embedding_dim, conv_channels, and fact_embedding_dim must be equal.
        assert conv_channels == word_embedding_dim, (
            f"conv_channels ({conv_channels}) must equal word_embedding_dim ({word_embedding_dim}) "
            "per paper Section 5.2.4 (single dimension d throughout)"
        )
        assert fact_embedding_dim == word_embedding_dim, (
            f"fact_embedding_dim ({fact_embedding_dim}) must equal word_embedding_dim ({word_embedding_dim}) "
            "per Equation 4: W_p ∈ R^{{d×2d}}, output ∈ R^d"
        )

        self.word_embedding_dim = word_embedding_dim
        self.fact_embedding_dim = fact_embedding_dim
        self.conv_channels = conv_channels
        self.device = device

        # ====================================================================
        # 1. Word Encoding Layer (Section 5.1.1)
        # ====================================================================
        # Word embeddings frozen per paper (Section 6.1.3: "We did not train the
        # word embeddings").  Un-seen words use Kaiming/He uniform initialisation.
        self.word_embeddings = nn.Embedding.from_pretrained(
            word_embedding_matrix,
            freeze=True,
            padding_idx=0  # Index 0 is <PAD>
        )

        # ====================================================================
        # 2. Attention-Based Convolution (Section 5.1.2)
        # ====================================================================
        # Two 1D convolution layers Wc1, Wc2 ∈ R^{d×k×d} shared across
        # both entity feature extractions (note in the paper).
        self.conv1 = nn.Conv1d(
            in_channels=word_embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # Attention weight matrix W_a ∈ R^{d×d} (Equation 1, no bias per paper)
        self.attention_W = nn.Linear(word_embedding_dim, word_embedding_dim, bias=False)

        # Dropout (applied after each convolution; rate = 0.25 per paper Section 6.1.3)
        self.dropout = nn.Dropout(dropout)

        # ====================================================================
        # 3. Fact Feature Projection (Equation 4)
        # ====================================================================
        # W_p ∈ R^{d×2d}, b_p ∈ R^d  →  f = W_p [e_h; e_t] + b_p
        self.fact_projection = nn.Linear(
            2 * conv_channels,  # [e_h; e_t] concatenation
            fact_embedding_dim
        )

        # num_types stored for type-matching (Equation 5)
        self.num_types = num_types

        self.to(device)
    
    def forward(self,
                head_descriptions: torch.Tensor,
                tail_descriptions: torch.Tensor,
                head_names: torch.Tensor,
                tail_names: torch.Tensor,
                relation_names: torch.Tensor,
                relation_domain_types: torch.Tensor,
                relation_range_types: torch.Tensor,
                relation_domain_words: torch.Tensor,
                relation_range_words: torch.Tensor,
                head_types: torch.Tensor,
                tail_types: torch.Tensor,
                head_desc_lengths: torch.Tensor,
                tail_desc_lengths: torch.Tensor) -> torch.Tensor:
        """
        Extract fact features from entity descriptions and relation information.

        Args:
            head_descriptions:    (batch, max_desc_len)  - Head entity description word indices
            tail_descriptions:    (batch, max_desc_len)  - Tail entity description word indices
            head_names:           (batch, max_name_len)  - Head entity name word indices (U_h)
            tail_names:           (batch, max_name_len)  - Tail entity name word indices (U_t)
            relation_names:       (batch, max_rel_len)   - Relation name word indices (U_r)
            relation_domain_types:(batch, num_types)     - Domain constraint multi-hot (Eq 5)
            relation_range_types: (batch, num_types)     - Range constraint multi-hot (Eq 5)
            relation_domain_words:(batch, max_type_len)  - Domain constraint type word indices
            relation_range_words: (batch, max_type_len)  - Range constraint type word indices
            head_types:           (batch, num_types)     - Head entity types multi-hot
            tail_types:           (batch, num_types)     - Tail entity types multi-hot
            head_desc_lengths:    (batch,)               - Actual head description lengths
            tail_desc_lengths:    (batch,)               - Actual tail description lengths

        Returns:
            fact_features: (batch, d) - Extracted fact features f = φ(fact)
        """
        # ====================================================================
        # Step 1: Extract relation-specific entity features (Section 5.1.2)
        # For the h-r pair: D_h attended by [T_{r,r}, U_r, U_t]   (range cstr + rel name + TAIL name)
        # For the t-r pair: D_t attended by [T_{r,d}, U_r, U_h]   (domain cstr + rel name + HEAD name)
        # See paper Section 5.1 and Figure 3.
        # ====================================================================
        head_features = self._extract_entity_features(
            entity_descriptions=head_descriptions,
            relation_names=relation_names,
            other_entity_names=tail_names,           # U_t  (paper: other entity NAME, Section 5.1.2)
            type_constraint_words=relation_range_words,  # T_{r,r} embedded as words
            desc_lengths=head_desc_lengths
        )

        tail_features = self._extract_entity_features(
            entity_descriptions=tail_descriptions,
            relation_names=relation_names,
            other_entity_names=head_names,           # U_h
            type_constraint_words=relation_domain_words,  # T_{r,d} embedded as words
            desc_lengths=tail_desc_lengths
        )

        # ====================================================================
        # Step 2: Fact feature projection  (Equation 4)
        # f = W_p [e_h; e_t] + b_p   (plain linear, no extra activation)
        # ====================================================================
        combined = torch.cat([head_features, tail_features], dim=1)  # (batch, 2d)
        fact_features = self.fact_projection(combined)               # (batch, d)

        # ====================================================================
        # Step 3: Type matching gate  (Equation 5)
        # f ← f × gate   where gate = 0.1 + 0.9 × (head_match × tail_match)
        # Soft floor at 0.1 instead of paper's hard zero: the paper's flat
        # multi-hot intersection fails for DBPedia's hierarchical type system
        # (entity has dbo:President, constraint requires parent dbo:Person) —
        # 87% of training triples receive gate=0, killing all gradients.
        # Floor 0.1 preserves gradient flow while still down-weighting
        # type-mismatched triples by 10× relative to matched ones.
        # ====================================================================
        head_type_match = self._type_matching(head_types, relation_domain_types)  # (batch,)
        tail_type_match = self._type_matching(tail_types, relation_range_types)   # (batch,)
        type_validity = (head_type_match * tail_type_match).unsqueeze(1)          # (batch, 1)
        type_gate = 0.1 + 0.9 * type_validity   # 0.1 (mismatch) .. 1.0 (match)
        fact_features = fact_features * type_gate

        return fact_features
    
    def _extract_entity_features(self,
                                 entity_descriptions: torch.Tensor,
                                 relation_names: torch.Tensor,
                                 other_entity_names: torch.Tensor,
                                 type_constraint_words: torch.Tensor,
                                 desc_lengths: torch.Tensor) -> torch.Tensor:
        """
        Extract entity features using attention-based convolution (Section 5.1.2).

        Implements Figure 3:
          1. Embed description words → D_e
          2. Two 1D convolutions → D'_e
          3. Attention: A = tanh(D'^T W_a cat(w_r, U_r, U_{e'}))  (Equation 1)
          4. Column-wise max-pool A → A'                           (Equation 2)
          5. Weighted average: e = D' softmax(A')                  (Equation 3)

        Args:
            entity_descriptions:   (batch, max_desc_len) word indices for this entity
            relation_names:        (batch, max_rel_len)  word indices for relation name U_r
            other_entity_names:    (batch, max_name_len) word indices for other entity NAME U_{e'}
                                   (per paper Section 5.1.2 and Figure 3: "the other entity name")
            type_constraint_words: (batch, max_type_len) word indices for type constraint T_{r,*}
                                   (embedded as words in the shared vocabulary)
            desc_lengths:          (batch,) actual non-padded description lengths

        Returns:
            entity_features: (batch, d)
        """
        batch_size = entity_descriptions.size(0)
        max_desc_len = entity_descriptions.size(1)

        # ====================================================================
        # Step 1: Word Encoding  (Section 5.1.1)
        # ====================================================================
        desc_emb = self.word_embeddings(entity_descriptions)  # (batch, n, d)
        desc_emb = desc_emb.transpose(1, 2)                   # (batch, d, n) for Conv1d

        # ====================================================================
        # Step 2: Two 1D convolutions → D'_h  (Section 5.1.2)
        # Paper: "two 1D convolutions" with filter width k=3.
        # The paper does not specify an activation between them; however,
        # without one the two stacked linears collapse to a single linear
        # operation — no extra expressiveness is added by the second layer.
        # We add ReLU between conv1 and conv2 for practical non-linearity.
        # (Reproducibility gap #7 / architectural assumption — see paper_code_correspondence.md)
        # ====================================================================
        conv1_out = self.conv1(desc_emb)      # (batch, d, n)
        conv1_out = self.dropout(conv1_out)
        conv1_out = F.relu(conv1_out)          # nonlinearity between convs: makes stack non-linear
        conv2_out = self.conv2(conv1_out)      # (batch, d, n)  ← D'_h
        conv2_out = self.dropout(conv2_out)
        desc_features = conv2_out             # (batch, d, n)

        # ====================================================================
        # Step 3: Build attention context  cat(w_r, U_r, U_t)
        # All three components are embedded through the SHARED word embedding
        # matrix (Section 5.1.1: "shared through a same vocabulary w_i ∈ W").
        # w_r  = word embedding of type constraint word(s) → mean over NON-PAD
        #        tokens to get (batch,1,d).  Paper: w_r ∈ R^{d×1} (single vec).
        #        padding_idx=0 makes zero-pad embeddings exactly 0, so a plain
        #        .mean() over 5 slots (1 real + 4 zeros) would divide by 5 and
        #        silently scale down w_r.  Average only over non-zero positions.
        # U_r  = word embeddings of relation name tokens   → (batch, k, d)
        # U_t  = word embeddings of other entity name tokens → (batch, p, d)
        # ====================================================================
        type_emb  = self.word_embeddings(type_constraint_words)          # (batch, T, d)
        type_mask = (type_constraint_words != 0).float().unsqueeze(-1)   # (batch, T, 1)
        type_sum  = (type_emb * type_mask).sum(dim=1, keepdim=True)      # (batch, 1, d)
        type_cnt  = type_mask.sum(dim=1, keepdim=True).clamp(min=1.0)    # (batch, 1, 1)
        type_emb  = type_sum / type_cnt                                  # (batch, 1, d)  w_r
        rel_emb   = self.word_embeddings(relation_names)                 # (batch, k, d)  U_r
        name_emb  = self.word_embeddings(other_entity_names)             # (batch, p, d)  U_t
        context_embedded = torch.cat([type_emb, rel_emb, name_emb], dim=1)  # (batch, 1+k+p, d)

        # ====================================================================
        # Step 4: Attention score matrix  (Equation 1)
        # A = tanh( D'^T  W_a  cat(...) )
        # For each (i,j): A[i,j] = d'_i · (W_a c_j)
        #
        # Derivation: (context_embedded @ W_a.T)[b,j,:] = W_a c_j  for each j
        # Then bmm(desc_for_att, (context @ W.T).T) gives exactly D'^T W_a cat(...)
        # ====================================================================
        desc_for_att = desc_features.transpose(1, 2)        # (batch, n, d)
        context_Wa   = context_embedded @ self.attention_W.weight.T  # (batch, C, d): W_a c_j
        attention_matrix = torch.bmm(desc_for_att, context_Wa.transpose(1, 2))  # (batch, n, C)
        attention_matrix = torch.tanh(attention_matrix)     # Equation 1

        # ====================================================================
        # Step 5: Column-wise max pooling  (Equation 2)
        # A'_i = max_{1 ≤ j ≤ 1+k+p} A_{i,j}
        # ====================================================================
        attention_scores, _ = torch.max(attention_matrix, dim=2)  # (batch, n)

        # Mask padding positions before softmax
        mask = (torch.arange(max_desc_len, device=entity_descriptions.device)
                .unsqueeze(0).expand(batch_size, -1))        # (batch, n)
        mask = mask < desc_lengths.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)

        # ====================================================================
        # Step 6: Weighted average  (Equation 3)
        # e_h = D'_h softmax(A')   (matrix × column-vector = weighted average)
        # ====================================================================
        attn_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)  # (batch, 1, n)
        entity_features = torch.bmm(attn_weights, desc_features.transpose(1, 2))  # (batch, 1, d)
        entity_features = entity_features.squeeze(1)                               # (batch, d)

        return entity_features
    
    def _type_matching(self,
                      entity_types: torch.Tensor,
                      constraint_types: torch.Tensor) -> torch.Tensor:
        """
        Hard binary type matching gate (Section 5.1.3, Equation 5).

        Paper: f ← f × (Σ_i(t_h ⊙ t_{r,d})_i × Σ_i(t_t ⊙ t_{r,r})_i)
        "If the type constraint is satisfied, the result values ... will be 1,
         otherwise 0.  Thus, the fact feature information f becomes a zero vector
         and disappears if the type constraint is not satisfied."

        tr,d and tr,r are ONE-HOT type vectors (single constraint per relation);
        th and tt are MULTI-HOT entity type vectors.
        The element-wise product is non-zero iff the entity has the required type.

        If no constraint exists (all zeros), validity = 1.0 (no restriction).

        Args:
            entity_types:     (batch, num_types) multi-hot
            constraint_types: (batch, num_types) one-hot (or multi-hot for multi-constraint)

        Returns:
            validity: (batch,) — 1.0 if valid or no constraint, 0.0 if invalid
        """
        # Σ_i (t_entity ⊙ t_constraint)_i
        match_sum      = (entity_types * constraint_types).sum(dim=1)   # (batch,)
        constraint_sum = constraint_types.sum(dim=1)                     # (batch,)

        # If no constraint (constraint_sum == 0) → no restriction → validity = 1
        # If constraint exists → validity = 1 iff entity satisfies it, else 0
        validity = torch.where(
            constraint_sum > 0,
            (match_sum > 0).float(),        # hard binary: 1 or 0
            torch.ones_like(match_sum)      # no constraint: always valid
        )
        # Additionally: if entity has NO type annotations at all (entity_sum == 0),
        # we cannot evaluate the constraint — default to valid rather than invalid.
        # Without this, entities missing from entity2type.txt always score 0,
        # killing gradients for ~30-60% of training triples.
        entity_sum = entity_types.sum(dim=1)   # (batch,)
        validity = torch.where(entity_sum > 0, validity, torch.ones_like(validity))
        return validity
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Helper Functions for Data Preparation
# ============================================================================

def tokenize_description(description: str, 
                        word2idx: Dict[str, int],
                        max_length: int = 50) -> Tuple[List[int], int]:
    """
    Convert description text to token indices.
    
    Args:
        description: Entity description text
        word2idx: Word to index mapping
        max_length: Maximum sequence length
        
    Returns:
        tokens: List of token indices (padded/truncated to max_length)
        actual_length: Actual length before padding
    """
    # Simple tokenization (split on whitespace, lowercase)
    words = description.lower().split()
    
    # Convert to indices
    tokens = []
    for word in words[:max_length]:
        if word in word2idx:
            tokens.append(word2idx[word])
        else:
            tokens.append(word2idx['<UNK>'])  # Unknown token
    
    actual_length = len(tokens)
    
    # Pad to max_length
    while len(tokens) < max_length:
        tokens.append(word2idx['<PAD>'])  # Padding token
    
    return tokens, actual_length


def prepare_fact_batch(facts: torch.Tensor,
                      entity_descriptions: List[str],
                      entity_names: List[str],
                      relation_names: List[str],
                      entity_types: List[List[str]],
                      relation_type_constraints: List[Tuple[List[str], List[str]]],
                      word2idx: Dict[str, int],
                      type2idx: Dict[str, int],
                      max_desc_length: int = 50,
                      max_rel_length: int = 10,
                      max_name_length: int = 5,
                      max_type_length: int = 5,
                      device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of facts for the FactFeatureExtractor.

    Args:
        facts: (batch, 3) - [head_id, relation_id, tail_id]
        entity_descriptions: List of entity description strings (D_e)
        entity_names: List of entity name strings (U_e)
        relation_names: List of relation name strings (U_r)
        entity_types: List of type lists for each entity (multi-hot)
        relation_type_constraints: List of (domain_type_str, range_type_str) for each
            relation.  Each element should be a SINGLE type string (or empty string)
            because the paper treats T_{r,d} and T_{r,r} as one-hot vectors
            (Section 5.1.3: "mask the domain/range type constraint as one-hot type vectors").
        word2idx: Word to index mapping
        type2idx: Type to index mapping
        max_desc_length: Max description length
        max_rel_length: Max relation name length
        max_name_length: Max entity/type name length
        max_type_length: Max type constraint word length
        device: 'cuda' or 'cpu'

    Returns:
        batch_dict: Dictionary of tensors ready for FactFeatureExtractor.forward()
    """
    batch_size = facts.size(0)
    num_types = len(type2idx)

    head_desc_tokens = []
    tail_desc_tokens = []
    head_name_tokens = []
    tail_name_tokens = []
    rel_name_tokens  = []
    rel_domain_word_tokens = []
    rel_range_word_tokens  = []
    head_desc_lengths = []
    tail_desc_lengths = []

    head_type_tensors  = torch.zeros(batch_size, num_types)
    tail_type_tensors  = torch.zeros(batch_size, num_types)
    rel_domain_tensors = torch.zeros(batch_size, num_types)
    rel_range_tensors  = torch.zeros(batch_size, num_types)

    for i, (h, r, t) in enumerate(facts):
        h, r, t = h.item(), r.item(), t.item()

        # Descriptions
        h_tokens, h_len = tokenize_description(entity_descriptions[h], word2idx, max_desc_length)
        t_tokens, t_len = tokenize_description(entity_descriptions[t], word2idx, max_desc_length)
        head_desc_tokens.append(h_tokens)
        tail_desc_tokens.append(t_tokens)
        head_desc_lengths.append(h_len)
        tail_desc_lengths.append(t_len)

        # Entity names (U_h, U_t)
        hn_tokens, _ = tokenize_description(entity_names[h], word2idx, max_name_length)
        tn_tokens, _ = tokenize_description(entity_names[t], word2idx, max_name_length)
        head_name_tokens.append(hn_tokens)
        tail_name_tokens.append(tn_tokens)

        # Relation name (U_r)
        r_tokens, _ = tokenize_description(relation_names[r], word2idx, max_rel_length)
        rel_name_tokens.append(r_tokens)

        # Type constraint tokens (T_{r,d} and T_{r,r} as words)
        domain_type_str, range_type_str = relation_type_constraints[r]
        dw_tokens, _ = tokenize_description(domain_type_str, word2idx, max_type_length)
        rw_tokens, _ = tokenize_description(range_type_str,  word2idx, max_type_length)
        rel_domain_word_tokens.append(dw_tokens)
        rel_range_word_tokens.append(rw_tokens)

        # Entity types — multi-hot
        for type_name in entity_types[h]:
            if type_name in type2idx:
                head_type_tensors[i, type2idx[type_name]] = 1.0
        for type_name in entity_types[t]:
            if type_name in type2idx:
                tail_type_tensors[i, type2idx[type_name]] = 1.0

        # Type constraints — one-hot per paper Section 5.1.3
        # domain_type_str / range_type_str should be a single type name or empty
        if domain_type_str and domain_type_str in type2idx:
            rel_domain_tensors[i, type2idx[domain_type_str]] = 1.0
        if range_type_str and range_type_str in type2idx:
            rel_range_tensors[i, type2idx[range_type_str]] = 1.0

    batch_dict = {
        'head_descriptions':     torch.tensor(head_desc_tokens,       dtype=torch.long, device=device),
        'tail_descriptions':     torch.tensor(tail_desc_tokens,       dtype=torch.long, device=device),
        'head_names':            torch.tensor(head_name_tokens,       dtype=torch.long, device=device),
        'tail_names':            torch.tensor(tail_name_tokens,       dtype=torch.long, device=device),
        'relation_names':        torch.tensor(rel_name_tokens,        dtype=torch.long, device=device),
        'relation_domain_words': torch.tensor(rel_domain_word_tokens, dtype=torch.long, device=device),
        'relation_range_words':  torch.tensor(rel_range_word_tokens,  dtype=torch.long, device=device),
        'relation_domain_types': rel_domain_tensors.to(device),
        'relation_range_types':  rel_range_tensors.to(device),
        'head_types':            head_type_tensors.to(device),
        'tail_types':            tail_type_tensors.to(device),
        'head_desc_lengths':     torch.tensor(head_desc_lengths, dtype=torch.long, device=device),
        'tail_desc_lengths':     torch.tensor(tail_desc_lengths, dtype=torch.long, device=device),
    }

    return batch_dict


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 TESTING FACT FEATURE EXTRACTOR")
    print("=" * 80)
    
    # Create dummy word embeddings
    vocab_size = 1000
    word_emb_dim = 300
    word_embedding_matrix = torch.randn(vocab_size, word_emb_dim)
    
    # Initialize model
    # NOTE: fact_embedding_dim and conv_channels MUST equal word_embedding_dim (d=300)
    # per the single-dimension constraint in __init__. Using 128 here would crash the
    # assertion (fact_embedding_dim == word_embedding_dim).
    model = FactFeatureExtractor(
        word_embedding_matrix=word_embedding_matrix,
        word_embedding_dim=word_emb_dim,  # 300
        fact_embedding_dim=word_emb_dim,  # must equal word_emb_dim
        conv_channels=word_emb_dim,       # must equal word_emb_dim
        device='cpu'
    )
    
    print(f"\n✅ Model initialized")
    print(f"   Parameters: {model.get_num_parameters():,}")
    
    # Create dummy batch
    batch_size = 4
    max_desc_len = 20
    max_rel_len = 5
    num_types = 50
    
    dummy_batch = {
        'head_descriptions': torch.randint(0, vocab_size, (batch_size, max_desc_len)),
        'tail_descriptions': torch.randint(0, vocab_size, (batch_size, max_desc_len)),
        'relation_names': torch.randint(0, vocab_size, (batch_size, max_rel_len)),
        'relation_domain_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'relation_range_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'head_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'tail_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'head_desc_lengths': torch.tensor([15, 18, 12, 20]),
        'tail_desc_lengths': torch.tensor([10, 20, 15, 18]),
    }
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    fact_features = model(**dummy_batch)
    
    print(f"✅ Output shape: {fact_features.shape}")
    print(f"   Expected: ({batch_size}, {word_emb_dim})")

    assert fact_features.shape == (batch_size, word_emb_dim), "Output shape mismatch!"
    
    print("\n🎉 All tests passed! Fact Feature Extractor is working.")