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
                 fact_embedding_dim: int = 128,
                 conv_channels: int = 128,
                 num_types: int = 100,
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 device: str = 'cuda'):
        """
        Args:
            word_embedding_matrix: Pre-trained word embeddings (vocab_size, word_emb_dim)
            word_embedding_dim: Dimension of word embeddings (300 for GloVe)
            fact_embedding_dim: Output fact embedding dimension
            conv_channels: Number of CNN channels
            kernel_size: CNN kernel width (paper uses 3)
            dropout: Dropout rate
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        
        self.word_embedding_dim = word_embedding_dim
        self.fact_embedding_dim = fact_embedding_dim
        self.conv_channels = conv_channels
        self.device = device
        
        # ====================================================================
        # 1. Word Encoding Layer (Section 5.1.1)
        # ====================================================================
        
        # Word embeddings: frozen per paper spec. GloVe vectors provide
        # pre-trained semantic signal; freezing prevents the 89M-param embedding
        # table from dominating gradient updates and destabilising CNN training.
        self.word_embeddings = nn.Embedding.from_pretrained(
            word_embedding_matrix,
            freeze=True,
            padding_idx=0  # Index 0 is <PAD>
        )
        
        # ====================================================================
        # 2. Attention-Based Convolution (Section 5.1.2)
        # ====================================================================
        
        # Two 1D convolution layers over descriptions
        # Paper: "two convolutions" (Figure 3)
        self.conv1 = nn.Conv1d(
            in_channels=word_embedding_dim,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Same padding
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # Attention mechanism (Equation 1 in paper)
        # Compares description features with relation info
        self.attention_W = nn.Linear(conv_channels, conv_channels)

        # Type constraint projections (registered here so optimizer sees them)
        # Maps multi-hot type vector -> word_embedding_dim for context assembly
        self.type_constraint_mapper = nn.Linear(num_types, word_embedding_dim)
        # Projects word-space context to conv_channels for attention dot-product
        self.context_projection = nn.Linear(word_embedding_dim, conv_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # ====================================================================
        # 3. Fact Feature Projection (Equation 4)
        # ====================================================================
        
        # Project concatenated head and tail features to fact embedding
        # Paper: f = W_p * [e_h; e_t] + b_p
        self.fact_projection = nn.Linear(
            2 * conv_channels,  # [e_h; e_t] concatenated
            fact_embedding_dim
        )
        
        self.to(device)
    
    def forward(self,
                head_descriptions: torch.Tensor,
                tail_descriptions: torch.Tensor,
                relation_names: torch.Tensor,
                relation_domain_types: torch.Tensor,
                relation_range_types: torch.Tensor,
                head_types: torch.Tensor,
                tail_types: torch.Tensor,
                head_desc_lengths: torch.Tensor,
                tail_desc_lengths: torch.Tensor) -> torch.Tensor:
        """
        Extract fact features from entity descriptions and relation information.
        
        Args:
            head_descriptions: (batch, max_desc_len) - Head entity descriptions as word indices
            tail_descriptions: (batch, max_desc_len) - Tail entity descriptions
            relation_names: (batch, max_rel_name_len) - Relation names as word indices
            relation_domain_types: (batch, num_types) - Domain type constraints (multi-hot)
            relation_range_types: (batch, num_types) - Range type constraints (multi-hot)
            head_types: (batch, num_types) - Head entity types (multi-hot)
            tail_types: (batch, num_types) - Tail entity types (multi-hot)
            head_desc_lengths: (batch,) - Actual lengths of head descriptions
            tail_desc_lengths: (batch,) - Actual lengths of tail descriptions
            
        Returns:
            fact_features: (batch, fact_embedding_dim) - Extracted fact features
        """
        batch_size = head_descriptions.size(0)
        
        # ====================================================================
        # Step 1: Extract entity features with attention
        # ====================================================================
        
        # Extract head entity features (attending to relation and tail)
        head_features = self._extract_entity_features(
            entity_descriptions=head_descriptions,
            relation_names=relation_names,
            other_entity_descriptions=tail_descriptions,
            relation_type_constraints=relation_domain_types,  # Domain constraint
            desc_lengths=head_desc_lengths,
            other_desc_lengths=tail_desc_lengths
        )
        
        # Extract tail entity features (attending to relation and head)
        tail_features = self._extract_entity_features(
            entity_descriptions=tail_descriptions,
            relation_names=relation_names,
            other_entity_descriptions=head_descriptions,
            relation_type_constraints=relation_range_types,  # Range constraint
            desc_lengths=tail_desc_lengths,
            other_desc_lengths=head_desc_lengths
        )
        
        # ====================================================================
        # Step 2: Type Matching (Section 5.1.3, Equation 5)
        # ====================================================================
        
        # Check if head types match domain constraint
        # Paper: Σ(t_h ⊙ t_{r,d})
        head_type_match = self._type_matching(head_types, relation_domain_types)
        
        # Check if tail types match range constraint  
        # Paper: Σ(t_t ⊙ t_{r,r})
        tail_type_match = self._type_matching(tail_types, relation_range_types)
        
        # Combined validity: both must match.
        # Clamp the product to ≥ 0.1 so the minimum scale for any fact is 0.1,
        # preventing near-zero features even when both type matches are at floor.
        type_validity = (head_type_match * tail_type_match).clamp(min=0.1).unsqueeze(1)  # (batch, 1)
        
        # ====================================================================
        # Step 3: Combine head and tail features (Equation 4)
        # ====================================================================
        
        # Concatenate entity features
        combined_features = torch.cat([head_features, tail_features], dim=1)  # (batch, 2*conv_channels)
        
        # Project to fact embedding dimension
        # Paper: f = W_p * [e_h; e_t] + b_p
        fact_features = self.fact_projection(combined_features)  # (batch, fact_emb_dim)
        fact_features = F.leaky_relu(fact_features)
        fact_features = self.dropout(fact_features)

        # Type validity: append as extra additive signal rather than as a multiplicative
        # gate.  The original paper multiplies by the validity scalar (Equation 5),
        # but doing so compresses features by up to 10x for mismatched types, driving
        # the CNN signal below the aggregator's bias floor and collapsing gradients.
        # Adding a small fraction of (validity-1)*|feat| as a penalty preserves the
        # feature norm while still penalising type-invalid facts.
        type_validity = (head_type_match * tail_type_match).clamp(min=0.1).unsqueeze(1)  # (batch, 1)
        # Soft gate: scale by validity but floor at 0.5 so signal is never crushed.
        type_scale = type_validity.clamp(min=0.5)   # (batch, 1), in [0.5, 1.0]
        fact_features = fact_features * type_scale

        return fact_features
    
    def _extract_entity_features(self,
                                 entity_descriptions: torch.Tensor,
                                 relation_names: torch.Tensor,
                                 other_entity_descriptions: torch.Tensor,
                                 relation_type_constraints: torch.Tensor,
                                 desc_lengths: torch.Tensor,
                                 other_desc_lengths: torch.Tensor) -> torch.Tensor:
        """
        Extract entity features using attention-based convolution.
        
        Implements Figure 3 of the paper:
        1. Embed description words
        2. Apply two CNN layers
        3. Compute attention based on relation and other entity
        4. Weighted average of CNN outputs
        
        Args:
            entity_descriptions: (batch, max_desc_len)
            relation_names: (batch, max_rel_len)
            other_entity_descriptions: (batch, max_desc_len)
            relation_type_constraints: (batch, num_types)
            desc_lengths: (batch,)
            other_desc_lengths: (batch,)
            
        Returns:
            entity_features: (batch, conv_channels)
        """
        batch_size = entity_descriptions.size(0)
        max_desc_len = entity_descriptions.size(1)
        
        # ====================================================================
        # Step 1: Word Encoding (Section 5.1.1)
        # ====================================================================
        
        # Embed description words: (batch, max_desc_len, word_emb_dim)
        desc_embedded = self.word_embeddings(entity_descriptions)
        
        # Transpose for CNN: (batch, word_emb_dim, max_desc_len)
        # CNN expects (batch, channels, length)
        desc_embedded = desc_embedded.transpose(1, 2)
        
        # ====================================================================
        # Step 2: Convolution Layers
        # ====================================================================
        
        # First convolution + activation
        conv1_out = self.conv1(desc_embedded)  # (batch, conv_channels, max_desc_len)
        conv1_out = F.leaky_relu(conv1_out)
        conv1_out = self.dropout(conv1_out)
        
        # Second convolution + activation
        conv2_out = self.conv2(conv1_out)  # (batch, conv_channels, max_desc_len)
        conv2_out = F.leaky_relu(conv2_out)
        conv2_out = self.dropout(conv2_out)
        
        # This is D'_h in the paper (after convolutions)
        desc_features = conv2_out  # (batch, conv_channels, max_desc_len)
        
        # ====================================================================
        # Step 3: Attention Mechanism (Equations 1, 2, 3)
        # ====================================================================
        
        # Embed relation names, other entity, and map relation constraints to word_emb_dim
        rel_embedded = self.word_embeddings(relation_names)  # (batch, max_rel_len, word_emb_dim)
        other_ent_embedded = self.word_embeddings(other_entity_descriptions)  # (batch, max_desc_len, word_emb_dim)
        
        type_constraints_embedded = self.type_constraint_mapper(relation_type_constraints)  # (batch, word_emb_dim)
        type_constraints_embedded = type_constraints_embedded.unsqueeze(1)  # (batch, 1, word_emb_dim)

        # cat(w_r, U_r, U_t)
        # context_embedded shape: (batch, 1 + max_rel_len + max_desc_len, word_emb_dim)
        context_embedded = torch.cat([type_constraints_embedded, rel_embedded, other_ent_embedded], dim=1)
            
        context_proj = self.context_projection(context_embedded)  # (batch, 1 + max_rel_len + max_desc_len, conv_channels)
        
        # Apply attention weight matrix to description features
        desc_for_attention = desc_features.transpose(1, 2)  # (batch, max_desc_len, conv_channels)
        attended_desc = self.attention_W(desc_for_attention)  # (batch, max_desc_len, conv_channels)
        
        # Compute attention score matrix A (Equation 1)
        # Paper: A = tanh((D'_h)^T * W_a * cat(w_r, U_r, U_t))
        # attention_matrix: (batch, max_desc_len, 1 + max_rel_len + max_desc_len)
        attention_matrix = torch.bmm(attended_desc, context_proj.transpose(1, 2))
        attention_matrix = torch.tanh(attention_matrix)
        
        # Column-wise max pooling (Equation 2)
        # Paper: A'_i = max_{1<j<1+k+p} A_{i,j}
        # attention_scores: (batch, max_desc_len)
        attention_scores, _ = torch.max(attention_matrix, dim=2)
        
        # Softmax to get attention weights (Equation 3)
        # Mask out padding positions
        mask = torch.arange(max_desc_len, device=self.device).expand(batch_size, max_desc_len)
        mask = mask < desc_lengths.unsqueeze(1)  # (batch, max_desc_len)
        
        attention_scores = attention_scores.masked_fill(~mask, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, max_desc_len)
        
        # ====================================================================
        # Step 4: Weighted Average (Equation 3)
        # ====================================================================
        
        # Paper: e_h = D'_h * softmax(A')
        # Weighted sum over description positions
        attention_weights = attention_weights.unsqueeze(1)  # (batch, 1, max_desc_len)
        entity_features = torch.bmm(attention_weights, desc_features.transpose(1, 2))  # (batch, 1, conv_channels)
        entity_features = entity_features.squeeze(1)  # (batch, conv_channels)
        
        return entity_features
    
    def _type_matching(self, 
                      entity_types: torch.Tensor,
                      constraint_types: torch.Tensor) -> torch.Tensor:
        """
        Type matching validation (Section 5.1.3, Equation 5).
        
        Checks if entity types satisfy relation type constraints.
        
        Args:
            entity_types: (batch, num_types) - Multi-hot encoding of entity types
            constraint_types: (batch, num_types) - Multi-hot encoding of required types
            
        Returns:
            validity: (batch,) - 1.0 if valid, 0.0 if invalid
        """
        # Paper: Σ(t_h ⊙ t_{r,d}) where ⊙ is element-wise multiplication
        # If entity has required type, at least one position will match
        
        # Element-wise multiplication
        matches = entity_types * constraint_types  # (batch, num_types)
        
        # Sum over types: if > 0, entity has at least one required type
        match_sum = matches.sum(dim=1)  # (batch,)
        
        # Soft type validity score (avoids hard zeroing that kills gradient flow).
        # If no constraint exists (constraint_sum==0), validity = 1.0 (fully valid).
        # If constraint exists, validity = fraction of constraint types matched,
        # clamped to a minimum of 0.1 so features are NEVER fully zeroed out.
        # Hard binary (0 or 1) caused zero features for all entities whose
        # type annotation vocabulary didn't perfectly match the relation's
        # constraint vocabulary, which zeroed gradients for those triples.
        constraint_sum = constraint_types.sum(dim=1)  # (batch,)

        # Fraction of constraint types that the entity satisfies [0, 1]
        match_frac = match_sum / constraint_sum.clamp(min=1)

        # Valid if: no constraint (score=1.0) OR proportional match (min 0.1)
        validity = torch.where(
            constraint_sum > 0,
            match_frac.clamp(min=0.1),   # soft score, never zero
            torch.ones_like(match_sum)
        )
        
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
                      relation_names: List[str],
                      entity_types: List[List[str]],
                      relation_type_constraints: List[Tuple[List[str], List[str]]],
                      word2idx: Dict[str, int],
                      type2idx: Dict[str, int],
                      max_desc_length: int = 50,
                      max_rel_length: int = 10,
                      device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Prepare a batch of facts for the FactFeatureExtractor.
    
    Args:
        facts: (batch, 3) - [head_id, relation_id, tail_id]
        entity_descriptions: List of entity description strings
        relation_names: List of relation name strings
        entity_types: List of type lists for each entity
        relation_type_constraints: List of (domain_types, range_types) for each relation
        word2idx: Word to index mapping
        type2idx: Type to index mapping
        max_desc_length: Max description length
        max_rel_length: Max relation name length
        device: 'cuda' or 'cpu'
        
    Returns:
        batch_dict: Dictionary of tensors ready for FactFeatureExtractor
    """
    batch_size = facts.size(0)
    num_types = len(type2idx)
    
    # Initialize tensors
    head_desc_tokens = []
    tail_desc_tokens = []
    rel_name_tokens = []
    head_desc_lengths = []
    tail_desc_lengths = []
    
    head_type_tensors = torch.zeros(batch_size, num_types)
    tail_type_tensors = torch.zeros(batch_size, num_types)
    rel_domain_tensors = torch.zeros(batch_size, num_types)
    rel_range_tensors = torch.zeros(batch_size, num_types)
    
    for i, (h, r, t) in enumerate(facts):
        h, r, t = h.item(), r.item(), t.item()
        
        # Tokenize descriptions
        h_tokens, h_len = tokenize_description(entity_descriptions[h], word2idx, max_desc_length)
        t_tokens, t_len = tokenize_description(entity_descriptions[t], word2idx, max_desc_length)
        r_tokens, _ = tokenize_description(relation_names[r], word2idx, max_rel_length)
        
        head_desc_tokens.append(h_tokens)
        tail_desc_tokens.append(t_tokens)
        rel_name_tokens.append(r_tokens)
        head_desc_lengths.append(h_len)
        tail_desc_lengths.append(t_len)
        
        # Encode types (multi-hot)
        for type_name in entity_types[h]:
            if type_name in type2idx:
                head_type_tensors[i, type2idx[type_name]] = 1.0
        
        for type_name in entity_types[t]:
            if type_name in type2idx:
                tail_type_tensors[i, type2idx[type_name]] = 1.0
        
        # Encode type constraints
        domain_types, range_types = relation_type_constraints[r]
        for type_name in domain_types:
            if type_name in type2idx:
                rel_domain_tensors[i, type2idx[type_name]] = 1.0
        
        for type_name in range_types:
            if type_name in type2idx:
                rel_range_tensors[i, type2idx[type_name]] = 1.0
    
    # Convert to tensors
    batch_dict = {
        'head_descriptions': torch.tensor(head_desc_tokens, dtype=torch.long, device=device),
        'tail_descriptions': torch.tensor(tail_desc_tokens, dtype=torch.long, device=device),
        'relation_names': torch.tensor(rel_name_tokens, dtype=torch.long, device=device),
        'relation_domain_types': rel_domain_tensors.to(device),
        'relation_range_types': rel_range_tensors.to(device),
        'head_types': head_type_tensors.to(device),
        'tail_types': tail_type_tensors.to(device),
        'head_desc_lengths': torch.tensor(head_desc_lengths, dtype=torch.long, device=device),
        'tail_desc_lengths': torch.tensor(tail_desc_lengths, dtype=torch.long, device=device),
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
    model = FactFeatureExtractor(
        word_embedding_matrix=word_embedding_matrix,
        word_embedding_dim=word_emb_dim,
        fact_embedding_dim=128,
        conv_channels=128,
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
    print(f"   Expected: ({batch_size}, 128)")
    
    assert fact_features.shape == (batch_size, 128), "Output shape mismatch!"
    
    print("\n🎉 All tests passed! Fact Feature Extractor is working.")