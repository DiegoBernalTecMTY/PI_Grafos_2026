"""
Test Attentive Aggregator
==========================

Verify that the attentive feature aggregation module works correctly.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from attentive_aggregator import (
    AttentiveAggregator,
    SampledAttentiveAggregator,
    aggregate_batch_facts
)


def test_initialization():
    """Test 1: Model initialization"""
    print("\n" + "=" * 80)
    print("TEST 1: Model Initialization")
    print("=" * 80)
    
    # Test different configurations
    configs = [
        {'fact_embedding_dim': 64, 'num_layers': 1},
        {'fact_embedding_dim': 128, 'num_layers': 2},
        {'fact_embedding_dim': 256, 'num_layers': 3},
    ]
    
    for config in configs:
        model = AttentiveAggregator(**config, device='cpu')
        print(f"✅ Config {config}: {model.get_num_parameters():,} parameters")
        
        # Verify structure
        assert len(model.attention_layers) == config['num_layers']
        assert len(model.update_layers) == config['num_layers']
    
    print("✅ TEST 1 PASSED")
    return True


def test_forward_pass_all_facts():
    """Test 2: Forward pass returning all facts"""
    print("\n" + "=" * 80)
    print("TEST 2: Forward Pass (All Facts)")
    print("=" * 80)
    
    num_facts = 50
    fact_emb_dim = 64
    num_edges = 200
    
    # Create data
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    fact_edge_index = torch.stack([
        torch.randint(0, num_facts, (num_edges,)),
        torch.randint(0, num_facts, (num_edges,))
    ], dim=0)
    
    # Create model
    model = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        device='cpu'
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(
            fact_embeddings=fact_embeddings,
            fact_edge_index=fact_edge_index,
            target_fact_ids=None  # Return all
        )
    
    print(f"Input shape: {fact_embeddings.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify
    assert output.shape == (num_facts, fact_emb_dim), "Shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print("✅ TEST 2 PASSED")
    return True


def test_forward_pass_target_facts():
    """Test 3: Forward pass returning target facts only"""
    print("\n" + "=" * 80)
    print("TEST 3: Forward Pass (Target Facts Only)")
    print("=" * 80)
    
    num_facts = 100
    fact_emb_dim = 128
    num_edges = 500
    num_targets = 10
    
    # Create data
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    fact_edge_index = torch.stack([
        torch.randint(0, num_facts, (num_edges,)),
        torch.randint(0, num_facts, (num_edges,))
    ], dim=0)
    target_ids = torch.randint(0, num_facts, (num_targets,))
    
    # Create model
    model = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        device='cpu'
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(
            fact_embeddings=fact_embeddings,
            fact_edge_index=fact_edge_index,
            target_fact_ids=target_ids
        )
    
    print(f"Target IDs: {target_ids.tolist()}")
    print(f"Output shape: {output.shape}")
    
    # Verify
    assert output.shape == (num_targets, fact_emb_dim), "Shape mismatch"
    
    print("✅ TEST 3 PASSED")
    return True


def test_multi_layer_aggregation():
    """Test 4: Multi-layer aggregation"""
    print("\n" + "=" * 80)
    print("TEST 4: Multi-Layer Aggregation")
    print("=" * 80)
    
    num_facts = 50
    fact_emb_dim = 64
    num_edges = 150
    
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    fact_edge_index = torch.stack([
        torch.randint(0, num_facts, (num_edges,)),
        torch.randint(0, num_facts, (num_edges,))
    ], dim=0)
    
    # Test different layer counts
    for num_layers in [1, 2, 3]:
        model = AttentiveAggregator(
            fact_embedding_dim=fact_emb_dim,
            num_layers=num_layers,
            device='cpu'
        )
        
        model.eval()
        with torch.no_grad():
            output = model(
                fact_embeddings=fact_embeddings,
                fact_edge_index=fact_edge_index
            )
        
        print(f"✅ {num_layers}-layer aggregation: output shape {output.shape}")
        assert output.shape == (num_facts, fact_emb_dim)
    
    print("✅ TEST 4 PASSED")
    return True


def test_attention_weights():
    """Test 5: Attention weight normalization"""
    print("\n" + "=" * 80)
    print("TEST 5: Attention Weight Normalization")
    print("=" * 80)
    
    # Create simple line graph
    # Fact 0 has neighbors: 1, 2, 3
    # Fact 1 has neighbors: 0, 2
    fact_edge_index = torch.tensor([
        [0, 0, 0, 1, 1],  # Sources
        [1, 2, 3, 0, 2],  # Targets
    ])
    
    num_facts = 4
    fact_emb_dim = 32
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    
    model = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=1,
        device='cpu'
    )
    
    # Test internal softmax function
    attention_scores = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5])
    source_indices = fact_edge_index[0]
    
    attention_weights = model._softmax_per_source(
        attention_scores=attention_scores,
        source_indices=source_indices,
        num_sources=num_facts
    )
    
    print(f"Attention scores: {attention_scores.tolist()}")
    print(f"Source indices: {source_indices.tolist()}")
    print(f"Attention weights: {attention_weights.tolist()}")
    
    # Check normalization per source
    # Fact 0 has 3 neighbors (indices 0,1,2), weights should sum to 1
    fact_0_weights = attention_weights[[0, 1, 2]]
    print(f"\nFact 0 neighbor weights: {fact_0_weights.tolist()}")
    print(f"Sum: {fact_0_weights.sum().item():.6f}")
    assert abs(fact_0_weights.sum().item() - 1.0) < 1e-5, "Weights don't sum to 1"
    
    # Fact 1 has 2 neighbors (indices 3,4), weights should sum to 1
    fact_1_weights = attention_weights[[3, 4]]
    print(f"\nFact 1 neighbor weights: {fact_1_weights.tolist()}")
    print(f"Sum: {fact_1_weights.sum().item():.6f}")
    assert abs(fact_1_weights.sum().item() - 1.0) < 1e-5, "Weights don't sum to 1"
    
    print("✅ TEST 5 PASSED")
    return True


def test_isolated_facts():
    """Test 6: Facts with no neighbors"""
    print("\n" + "=" * 80)
    print("TEST 6: Isolated Facts (No Neighbors)")
    print("=" * 80)
    
    num_facts = 10
    fact_emb_dim = 64
    
    # Create graph where some facts have no neighbors
    # Facts 0-4 are connected, facts 5-9 are isolated
    fact_edge_index = torch.tensor([
        [0, 1, 2, 3, 4],  # Sources
        [1, 2, 3, 4, 0],  # Targets (circular)
    ])
    
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    
    model = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        device='cpu'
    )
    
    model.eval()
    with torch.no_grad():
        output = model(
            fact_embeddings=fact_embeddings,
            fact_edge_index=fact_edge_index
        )
    
    print(f"Input embeddings shape: {fact_embeddings.shape}")
    print(f"Output embeddings shape: {output.shape}")
    
    # Isolated facts should still have output (from self-connections)
    isolated_outputs = output[5:]
    print(f"\nIsolated facts output (should not be zero):")
    print(f"  Mean magnitude: {isolated_outputs.abs().mean().item():.4f}")
    
    # Verify no NaN or Inf
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    
    print("✅ TEST 6 PASSED")
    return True


def test_gradient_flow():
    """Test 7: Gradient flow through layers"""
    print("\n" + "=" * 80)
    print("TEST 7: Gradient Flow")
    print("=" * 80)
    
    num_facts = 20
    fact_emb_dim = 32
    num_edges = 50
    
    fact_embeddings = torch.randn(num_facts, fact_emb_dim, requires_grad=True)
    fact_edge_index = torch.stack([
        torch.randint(0, num_facts, (num_edges,)),
        torch.randint(0, num_facts, (num_edges,))
    ], dim=0)
    
    model = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        device='cpu'
    )
    
    # Forward pass
    model.train()
    output = model(
        fact_embeddings=fact_embeddings,
        fact_edge_index=fact_edge_index,
        target_fact_ids=torch.tensor([0, 1, 2])
    )
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    print(f"Input embeddings gradient: {fact_embeddings.grad is not None}")
    
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            print(f"  ✅ {name}: grad shape {param.grad.shape}")
    
    print(f"\nTotal parameters with gradients: {grad_count}/{len(list(model.parameters()))}")
    
    assert grad_count > 0, "No gradients computed!"
    assert fact_embeddings.grad is not None, "Input gradients not computed!"
    
    print("✅ TEST 7 PASSED")
    return True


def test_sampled_aggregator():
    """Test 8: Sampled aggregator (memory efficient)"""
    print("\n" + "=" * 80)
    print("TEST 8: Sampled Aggregator")
    print("=" * 80)
    
    num_facts = 100
    fact_emb_dim = 64
    num_edges = 1000  # Dense graph
    
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    fact_edge_index = torch.stack([
        torch.randint(0, num_facts, (num_edges,)),
        torch.randint(0, num_facts, (num_edges,))
    ], dim=0)
    
    # Regular aggregator
    regular = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        device='cpu'
    )
    
    # Sampled aggregator (only use 5 neighbors per fact)
    sampled = SampledAttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        num_samples=5,
        device='cpu'
    )
    
    target_ids = torch.tensor([0, 1, 2, 3, 4])
    
    # Compare outputs
    regular.eval()
    sampled.eval()
    
    with torch.no_grad():
        output_regular = regular(fact_embeddings, fact_edge_index, target_ids)
        output_sampled = sampled(fact_embeddings, fact_edge_index, target_ids)
    
    print(f"Regular output shape: {output_regular.shape}")
    print(f"Sampled output shape: {output_sampled.shape}")
    
    # Both should have same shape
    assert output_regular.shape == output_sampled.shape
    
    # Outputs will be different due to sampling, but both should be valid
    assert not torch.isnan(output_regular).any()
    assert not torch.isnan(output_sampled).any()
    
    print("✅ TEST 8 PASSED")
    return True


def test_batch_processing():
    """Test 9: Batch processing helper function"""
    print("\n" + "=" * 80)
    print("TEST 9: Batch Processing")
    print("=" * 80)
    
    num_facts = 200
    fact_emb_dim = 64
    num_edges = 800
    num_targets = 50
    
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    fact_edge_index = torch.stack([
        torch.randint(0, num_facts, (num_edges,)),
        torch.randint(0, num_facts, (num_edges,))
    ], dim=0)
    target_ids = torch.randint(0, num_facts, (num_targets,))
    
    model = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        device='cpu'
    )
    
    # Use batch processing
    output = aggregate_batch_facts(
        aggregator=model,
        fact_embeddings=fact_embeddings,
        fact_edge_index=fact_edge_index,
        target_fact_ids=target_ids,
        batch_size=10
    )
    
    print(f"Processed {num_targets} targets in batches of 10")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (num_targets, fact_emb_dim)
    
    print("✅ TEST 9 PASSED")
    return True


def test_large_scale():
    """Test 10: Large-scale test (performance)"""
    print("\n" + "=" * 80)
    print("TEST 10: Large-Scale Performance")
    print("=" * 80)
    
    # Simulate Codex-M scale
    num_facts = 20_000
    fact_emb_dim = 128
    num_edges = 100_000
    
    print(f"Creating large graph: {num_facts} facts, {num_edges} edges...")
    
    fact_embeddings = torch.randn(num_facts, fact_emb_dim)
    fact_edge_index = torch.stack([
        torch.randint(0, num_facts, (num_edges,)),
        torch.randint(0, num_facts, (num_edges,))
    ], dim=0)
    
    model = AttentiveAggregator(
        fact_embedding_dim=fact_emb_dim,
        num_layers=2,
        device='cpu'
    )
    
    # Test with subset of targets
    target_ids = torch.randint(0, num_facts, (1000,))
    
    print(f"Running aggregation for {len(target_ids)} targets...")
    
    import time
    start = time.time()
    
    model.eval()
    with torch.no_grad():
        output = model(
            fact_embeddings=fact_embeddings,
            fact_edge_index=fact_edge_index,
            target_fact_ids=target_ids
        )
    
    elapsed = time.time() - start
    
    print(f"\n⏱️  Time: {elapsed:.2f} seconds")
    print(f"⏱️  Speed: {len(target_ids)/elapsed:.0f} facts/second")
    print(f"📊 Output shape: {output.shape}")
    
    assert output.shape == (len(target_ids), fact_emb_dim)
    
    print("✅ TEST 10 PASSED")
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 RUNNING ALL ATTENTIVE AGGREGATOR TESTS")
    print("=" * 80)
    
    tests = [
        ("Initialization", test_initialization),
        ("Forward Pass (All)", test_forward_pass_all_facts),
        ("Forward Pass (Targets)", test_forward_pass_target_facts),
        ("Multi-Layer Aggregation", test_multi_layer_aggregation),
        ("Attention Weights", test_attention_weights),
        ("Isolated Facts", test_isolated_facts),
        ("Gradient Flow", test_gradient_flow),
        ("Sampled Aggregator", test_sampled_aggregator),
        ("Batch Processing", test_batch_processing),
        ("Large Scale", test_large_scale),
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
        print("\n🎉 ALL TESTS PASSED! Attentive Aggregator is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")


if __name__ == "__main__":
    run_all_tests()