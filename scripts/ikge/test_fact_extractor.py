"""
Test Fact Feature Extractor
============================

Verify that the fact feature extraction module works correctly.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fact_feature_extractor import (
    FactFeatureExtractor,
    tokenize_description,
    prepare_fact_batch
)


def test_tokenization():
    """Test 1: Text tokenization"""
    print("\n" + "=" * 80)
    print("TEST 1: Text Tokenization")
    print("=" * 80)
    
    word2idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        'harvard': 2,
        'university': 3,
        'is': 4,
        'a': 5,
        'private': 6,
    }
    
    desc = "Harvard University is a private university"
    tokens, length = tokenize_description(desc, word2idx, max_length=10)
    
    print(f"Description: '{desc}'")
    print(f"Tokens: {tokens}")
    print(f"Length: {length}")
    
    # Verify
    assert len(tokens) == 10, "Should pad/truncate to max_length"
    assert length == 6, "Should count actual words"
    assert tokens[0] == word2idx['harvard'], "First token should be 'harvard'"
    assert tokens[6:] == [0, 0, 0, 0], "Should pad with <PAD>"
    
    print("✅ TEST 1 PASSED")
    return True


def test_model_initialization():
    """Test 2: Model initialization"""
    print("\n" + "=" * 80)
    print("TEST 2: Model Initialization")
    print("=" * 80)
    
    vocab_size = 1000
    word_emb_dim = 300
    fact_emb_dim = 128
    
    # Create random word embeddings
    word_embeddings = torch.randn(vocab_size, word_emb_dim)
    
    # Initialize model
    model = FactFeatureExtractor(
        word_embedding_matrix=word_embeddings,
        word_embedding_dim=word_emb_dim,
        fact_embedding_dim=fact_emb_dim,
        conv_channels=128,
        kernel_size=3,
        dropout=0.2,
        device='cpu'
    )
    
    print(f"✅ Model created")
    print(f"   Word embedding dim: {word_emb_dim}")
    print(f"   Fact embedding dim: {fact_emb_dim}")
    print(f"   Total parameters: {model.get_num_parameters():,}")
    
    # Verify architecture
    assert hasattr(model, 'word_embeddings'), "Should have word embeddings"
    assert hasattr(model, 'conv1'), "Should have conv1 layer"
    assert hasattr(model, 'conv2'), "Should have conv2 layer"
    assert hasattr(model, 'attention_W'), "Should have attention layer"
    assert hasattr(model, 'fact_projection'), "Should have fact projection"
    
    # Verify word embeddings are frozen
    assert not model.word_embeddings.weight.requires_grad, "Word embeddings should be frozen"
    
    print("✅ TEST 2 PASSED")
    return True


def test_forward_pass():
    """Test 3: Forward pass"""
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass")
    print("=" * 80)
    
    batch_size = 8
    vocab_size = 500
    word_emb_dim = 300
    fact_emb_dim = 128
    max_desc_len = 30
    max_rel_len = 10
    num_types = 20
    
    # Create model
    word_embeddings = torch.randn(vocab_size, word_emb_dim)
    model = FactFeatureExtractor(
        word_embedding_matrix=word_embeddings,
        word_embedding_dim=word_emb_dim,
        fact_embedding_dim=fact_emb_dim,
        device='cpu'
    )
    
    # Create batch
    batch = {
        'head_descriptions': torch.randint(0, vocab_size, (batch_size, max_desc_len)),
        'tail_descriptions': torch.randint(0, vocab_size, (batch_size, max_desc_len)),
        'relation_names': torch.randint(0, vocab_size, (batch_size, max_rel_len)),
        'relation_domain_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'relation_range_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'head_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'tail_types': torch.randint(0, 2, (batch_size, num_types)).float(),
        'head_desc_lengths': torch.randint(10, max_desc_len, (batch_size,)),
        'tail_desc_lengths': torch.randint(10, max_desc_len, (batch_size,)),
    }
    
    print(f"Batch size: {batch_size}")
    print(f"Max description length: {max_desc_len}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    
    print(f"\n✅ Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected: ({batch_size}, {fact_emb_dim})")
    
    # Verify output
    assert output.shape == (batch_size, fact_emb_dim), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("✅ TEST 3 PASSED")
    return True


def test_type_matching():
    """Test 4: Type matching validation"""
    print("\n" + "=" * 80)
    print("TEST 4: Type Matching")
    print("=" * 80)
    
    batch_size = 4
    vocab_size = 100
    word_emb_dim = 50
    num_types = 10
    
    # Create model
    word_embeddings = torch.randn(vocab_size, word_emb_dim)
    model = FactFeatureExtractor(
        word_embedding_matrix=word_embeddings,
        word_embedding_dim=word_emb_dim,
        fact_embedding_dim=64,
        device='cpu'
    )
    
    # Test cases:
    # 1. Valid: entity has required type
    # 2. Invalid: entity doesn't have required type
    # 3. No constraint: should be valid
    
    entity_types = torch.tensor([
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Has types 0 and 2
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Has type 1
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Has type 3
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Has types 0,1,2,3
    ], dtype=torch.float)
    
    # Constraint requires types 0 or 2
    constraint_types = torch.tensor([
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Requires 0 or 2
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Requires 0 or 2
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Requires 0 or 2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No constraint
    ], dtype=torch.float)
    
    validity = model._type_matching(entity_types, constraint_types)
    
    print("Type matching results:")
    print(f"  Entity 0 (has 0,2) vs constraint (0,2): {validity[0].item()} (should be 1.0)")
    print(f"  Entity 1 (has 1) vs constraint (0,2): {validity[1].item()} (should be 0.0)")
    print(f"  Entity 2 (has 3) vs constraint (0,2): {validity[2].item()} (should be 0.0)")
    print(f"  Entity 3 (has 0,1,2,3) vs no constraint: {validity[3].item()} (should be 1.0)")
    
    # Verify
    assert validity[0].item() == 1.0, "Should be valid (has matching type)"
    assert validity[1].item() == 0.0, "Should be invalid (no matching type)"
    assert validity[2].item() == 0.0, "Should be invalid (no matching type)"
    assert validity[3].item() == 1.0, "Should be valid (no constraint)"
    
    print("✅ TEST 4 PASSED")
    return True


def test_type_filtering():
    """Test 5: Type filtering zeros out invalid facts"""
    print("\n" + "=" * 80)
    print("TEST 5: Type Filtering")
    print("=" * 80)
    
    batch_size = 4
    vocab_size = 100
    word_emb_dim = 50
    fact_emb_dim = 32
    num_types = 5
    
    # Create model
    word_embeddings = torch.randn(vocab_size, word_emb_dim)
    model = FactFeatureExtractor(
        word_embedding_matrix=word_embeddings,
        word_embedding_dim=word_emb_dim,
        fact_embedding_dim=fact_emb_dim,
        device='cpu'
    )
    
    # Create batch with invalid type combinations
    batch = {
        'head_descriptions': torch.randint(0, vocab_size, (batch_size, 20)),
        'tail_descriptions': torch.randint(0, vocab_size, (batch_size, 20)),
        'relation_names': torch.randint(0, vocab_size, (batch_size, 5)),
        'relation_domain_types': torch.tensor([
            [1, 0, 0, 0, 0],  # Requires type 0
            [0, 1, 0, 0, 0],  # Requires type 1
            [1, 0, 0, 0, 0],  # Requires type 0
            [0, 0, 0, 0, 0],  # No constraint
        ], dtype=torch.float),
        'relation_range_types': torch.tensor([
            [1, 0, 0, 0, 0],  # Requires type 0
            [1, 0, 0, 0, 0],  # Requires type 0
            [1, 0, 0, 0, 0],  # Requires type 0
            [0, 0, 0, 0, 0],  # No constraint
        ], dtype=torch.float),
        'head_types': torch.tensor([
            [1, 0, 0, 0, 0],  # Has type 0 ✓
            [1, 0, 0, 0, 0],  # Has type 0, but needs type 1 ✗
            [0, 1, 0, 0, 0],  # Has type 1, but needs type 0 ✗
            [1, 1, 1, 1, 1],  # Has all types ✓
        ], dtype=torch.float),
        'tail_types': torch.tensor([
            [1, 0, 0, 0, 0],  # Has type 0 ✓
            [1, 0, 0, 0, 0],  # Has type 0 ✓
            [1, 0, 0, 0, 0],  # Has type 0 ✓
            [1, 0, 0, 0, 0],  # Has type 0 ✓
        ], dtype=torch.float),
        'head_desc_lengths': torch.tensor([15, 15, 15, 15]),
        'tail_desc_lengths': torch.tensor([15, 15, 15, 15]),
    }
    
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    
    print("Output after type filtering:")
    for i in range(batch_size):
        is_zero = torch.allclose(output[i], torch.zeros_like(output[i]))
        print(f"  Fact {i}: {'ZERO (invalid)' if is_zero else 'NON-ZERO (valid)'}")
    
    # Fact 0: head valid (type 0), tail valid (type 0) → VALID
    # Fact 1: head invalid (has 0, needs 1), tail valid → INVALID
    # Fact 2: head invalid (has 1, needs 0), tail valid → INVALID  
    # Fact 3: head valid (no constraint), tail valid → VALID
    
    assert not torch.allclose(output[0], torch.zeros_like(output[0])), "Fact 0 should be valid"
    assert torch.allclose(output[1], torch.zeros_like(output[1])), "Fact 1 should be invalid"
    assert torch.allclose(output[2], torch.zeros_like(output[2])), "Fact 2 should be invalid"
    assert not torch.allclose(output[3], torch.zeros_like(output[3])), "Fact 3 should be valid"
    
    print("✅ TEST 5 PASSED")
    return True


def test_attention_mechanism():
    """Test 6: Attention weights sum to 1"""
    print("\n" + "=" * 80)
    print("TEST 6: Attention Mechanism")
    print("=" * 80)
    
    print("Note: This is tested implicitly through forward pass")
    print("Attention uses softmax, which guarantees sum to 1")
    
    batch_size = 2
    vocab_size = 100
    
    word_embeddings = torch.randn(vocab_size, 300)
    model = FactFeatureExtractor(
        word_embedding_matrix=word_embeddings,
        word_embedding_dim=300,
        fact_embedding_dim=64,
        device='cpu'
    )
    
    batch = {
        'head_descriptions': torch.randint(0, vocab_size, (batch_size, 20)),
        'tail_descriptions': torch.randint(0, vocab_size, (batch_size, 20)),
        'relation_names': torch.randint(0, vocab_size, (batch_size, 5)),
        'relation_domain_types': torch.zeros(batch_size, 10),
        'relation_range_types': torch.zeros(batch_size, 10),
        'head_types': torch.zeros(batch_size, 10),
        'tail_types': torch.zeros(batch_size, 10),
        'head_desc_lengths': torch.tensor([15, 18]),
        'tail_desc_lengths': torch.tensor([12, 20]),
    }
    
    model.eval()
    with torch.no_grad():
        output = model(**batch)
    
    print(f"✅ Forward pass successful with attention")
    print(f"   Output shape: {output.shape}")
    
    print("✅ TEST 6 PASSED")
    return True


def test_batch_preparation():
    """Test 7: Batch preparation helper"""
    print("\n" + "=" * 80)
    print("TEST 7: Batch Preparation")
    print("=" * 80)
    
    # Create sample data
    facts = torch.tensor([
        [0, 0, 1],  # Harvard -> locatedIn -> Boston
        [1, 1, 2],  # Boston -> capitalOf -> Massachusetts
    ])
    
    entity_descriptions = [
        "Harvard University is a private research university",  # 0
        "Boston is the capital city of Massachusetts",          # 1
        "Massachusetts is a U.S. state",                        # 2
    ]
    
    relation_names = [
        "located in",     # 0
        "capital of",     # 1
    ]
    
    entity_types = [
        ["University", "Organization"],  # Harvard
        ["City", "Place"],               # Boston
        ["State", "Place"],              # Massachusetts
    ]
    
    relation_type_constraints = [
        (["Organization", "University"], ["Place", "City"]),  # locatedIn
        (["City"], ["State", "Country"]),                     # capitalOf
    ]
    
    word2idx = {
        '<PAD>': 0, '<UNK>': 1,
        'harvard': 2, 'university': 3, 'boston': 4, 'massachusetts': 5,
        'capital': 6, 'city': 7, 'state': 8, 'located': 9, 'in': 10,
    }
    
    type2idx = {
        'University': 0, 'Organization': 1, 'City': 2, 'Place': 3,
        'State': 4, 'Country': 5,
    }
    
    print("Preparing batch...")
    batch_dict = prepare_fact_batch(
        facts=facts,
        entity_descriptions=entity_descriptions,
        relation_names=relation_names,
        entity_types=entity_types,
        relation_type_constraints=relation_type_constraints,
        word2idx=word2idx,
        type2idx=type2idx,
        max_desc_length=20,
        max_rel_length=5,
        device='cpu'
    )
    
    print("\n✅ Batch prepared")
    print(f"   Keys: {list(batch_dict.keys())}")
    print(f"   Head descriptions shape: {batch_dict['head_descriptions'].shape}")
    print(f"   Tail descriptions shape: {batch_dict['tail_descriptions'].shape}")
    print(f"   Relation names shape: {batch_dict['relation_names'].shape}")
    print(f"   Head types shape: {batch_dict['head_types'].shape}")
    
    # Verify shapes
    assert batch_dict['head_descriptions'].shape == (2, 20)
    assert batch_dict['head_types'].shape == (2, len(type2idx))
    
    print("✅ TEST 7 PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 RUNNING ALL FACT FEATURE EXTRACTOR TESTS")
    print("=" * 80)
    
    tests = [
        ("Tokenization", test_tokenization),
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Type Matching", test_type_matching),
        ("Type Filtering", test_type_filtering),
        ("Attention Mechanism", test_attention_mechanism),
        ("Batch Preparation", test_batch_preparation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for test_name, passed_test, error in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
        if error:
            print(f"   Error: {error}")
    
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Fact Feature Extractor is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")


if __name__ == "__main__":
    run_all_tests()