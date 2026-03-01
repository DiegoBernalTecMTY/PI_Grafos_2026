"""
Test Line Graph Construction
============================

Run this to verify line graph is working correctly before integrating with IKGE.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from line_graph import create_line_graph, LineGraph


def test_basic_construction():
    """Test 1: Basic line graph construction"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Line Graph Construction")
    print("=" * 80)
    
    # Simple KG: 4 facts, should create clear adjacencies
    triples = torch.tensor([
        [0, 0, 1],  # Fact 0: Harvard -> locatedIn -> Boston
        [1, 1, 2],  # Fact 1: Boston -> capitalOf -> Massachusetts
        [2, 2, 3],  # Fact 2: Massachusetts -> partOf -> USA
        [0, 2, 2],  # Fact 3: Harvard -> partOf -> Massachusetts
    ])
    
    edge_index, line_graph = create_line_graph(triples, verbose=True)
    
    # Verify structure
    assert edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert edge_index.shape[1] > 0, "Should have at least one edge"
    
    # Check specific adjacencies
    neighbors_0 = set(line_graph.get_neighbors(0))
    neighbors_1 = set(line_graph.get_neighbors(1))
    
    print(f"\n✅ Fact 0 neighbors: {neighbors_0}")
    print(f"✅ Fact 1 neighbors: {neighbors_1}")
    
    # Fact 0 (Harvard, locatedIn, Boston) shares entities with:
    #   - Fact 1 (shares Boston)
    #   - Fact 3 (shares Harvard)
    assert 1 in neighbors_0, "Fact 0 should be adjacent to Fact 1 (share Boston)"
    assert 3 in neighbors_0, "Fact 0 should be adjacent to Fact 3 (share Harvard)"
    
    print("✅ TEST 1 PASSED: Basic construction works!")
    return True


def test_isolated_facts():
    """Test 2: Facts with no shared entities"""
    print("\n" + "=" * 80)
    print("TEST 2: Isolated Facts")
    print("=" * 80)
    
    # KG with isolated facts (no shared entities)
    triples = torch.tensor([
        [0, 0, 1],  # Fact 0: Entity 0 -> 1
        [2, 1, 3],  # Fact 1: Entity 2 -> 3 (isolated)
    ])
    
    edge_index, line_graph = create_line_graph(triples, verbose=True)
    
    neighbors_0 = line_graph.get_neighbors(0)
    neighbors_1 = line_graph.get_neighbors(1)
    
    print(f"\n✅ Fact 0 neighbors: {neighbors_0}")
    print(f"✅ Fact 1 neighbors: {neighbors_1}")
    
    assert len(neighbors_0) == 0, "Isolated fact should have no neighbors"
    assert len(neighbors_1) == 0, "Isolated fact should have no neighbors"
    
    stats = line_graph.get_statistics()
    assert stats['isolated_nodes'] == 2, "Should have 2 isolated nodes"
    
    print("✅ TEST 2 PASSED: Isolated facts handled correctly!")
    return True


def test_dense_graph():
    """Test 3: Dense graph (many shared entities)"""
    print("\n" + "=" * 80)
    print("TEST 3: Dense Graph")
    print("=" * 80)
    
    # All facts share entity 0
    triples = torch.tensor([
        [0, 0, 1],
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
    ])
    
    edge_index, line_graph = create_line_graph(triples, verbose=True)
    
    # Each fact should be connected to all others (fully connected)
    for fact_id in range(5):
        neighbors = line_graph.get_neighbors(fact_id)
        print(f"Fact {fact_id} has {len(neighbors)} neighbors")
        assert len(neighbors) == 4, f"Fact {fact_id} should have 4 neighbors (all others)"
    
    stats = line_graph.get_statistics()
    print(f"\n📊 Statistics: {stats}")
    
    # In a fully connected graph of 5 nodes:
    # Each node connects to 4 others = 5 * 4 = 20 directed edges
    assert stats['num_edges'] == 20, "Should have 20 edges (fully connected, directed)"
    assert stats['avg_degree'] == 4.0, "Average degree should be 4"
    
    print("✅ TEST 3 PASSED: Dense graph works correctly!")
    return True


def test_k_hop_neighbors():
    """Test 4: K-hop neighbor retrieval"""
    print("\n" + "=" * 80)
    print("TEST 4: K-hop Neighbor Retrieval")
    print("=" * 80)
    
    # Linear chain: 0 -> 1 -> 2 -> 3
    triples = torch.tensor([
        [0, 0, 1],  # Fact 0
        [1, 0, 2],  # Fact 1 (shares entity 1 with Fact 0)
        [2, 0, 3],  # Fact 2 (shares entity 2 with Fact 1)
        [3, 0, 4],  # Fact 3 (shares entity 3 with Fact 2)
    ])
    
    edge_index, line_graph = create_line_graph(triples, verbose=True)
    
    # Get 3-hop neighbors of Fact 0
    k_hop = line_graph.get_k_hop_neighbors(fact_id=0, k=3)
    
    print("\n🔍 K-hop neighbors of Fact 0:")
    for hop, neighbors in k_hop.items():
        print(f"   Hop {hop}: {neighbors}")
    
    # Verify chain structure
    assert 0 in k_hop[0], "0-hop should contain itself"
    assert 1 in k_hop[1], "1-hop should contain Fact 1"
    assert 2 in k_hop[2], "2-hop should contain Fact 2"
    assert 3 in k_hop[3], "3-hop should contain Fact 3"
    
    print("✅ TEST 4 PASSED: K-hop retrieval works!")
    return True


def test_large_scale():
    """Test 5: Larger graph (performance test)"""
    print("\n" + "=" * 80)
    print("TEST 5: Large-Scale Graph")
    print("=" * 80)
    
    # Simulate 1000 facts
    num_facts = 1000
    num_entities = 200
    
    # Random triples
    torch.manual_seed(42)
    heads = torch.randint(0, num_entities, (num_facts,))
    relations = torch.randint(0, 10, (num_facts,))
    tails = torch.randint(0, num_entities, (num_facts,))
    triples = torch.stack([heads, relations, tails], dim=1)
    
    print(f"Building line graph for {num_facts} facts...")
    
    import time
    start = time.time()
    edge_index, line_graph = create_line_graph(triples, verbose=True)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Construction time: {elapsed:.2f} seconds")
    
    stats = line_graph.get_statistics()
    print(f"\n📊 Final Statistics:")
    for key, val in stats.items():
        print(f"   {key}: {val:,}")
    
    # Performance checks
    assert elapsed < 30, "Construction should complete in < 30 seconds"
    assert stats['num_edges'] > 0, "Should have edges"
    
    print("✅ TEST 5 PASSED: Large-scale construction successful!")
    return True


def test_codex_m_scale():
    """Test 6: Codex-M scale test (20k facts)"""
    print("\n" + "=" * 80)
    print("TEST 6: Codex-M Scale Test (~20k facts)")
    print("=" * 80)
    
    # Simulate Codex-M scale
    num_facts = 20_000
    num_entities = 17_050  # Actual Codex-M size
    num_relations = 51
    
    print(f"Simulating Codex-M: {num_facts} facts, {num_entities} entities...")
    
    # Random triples (uniform distribution)
    torch.manual_seed(42)
    heads = torch.randint(0, num_entities, (num_facts,))
    relations = torch.randint(0, num_relations, (num_facts,))
    tails = torch.randint(0, num_entities, (num_facts,))
    triples = torch.stack([heads, relations, tails], dim=1)
    
    import time
    start = time.time()
    edge_index, line_graph = create_line_graph(triples, verbose=True)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Construction time: {elapsed:.2f} seconds")
    print(f"⏱️  Time per fact: {elapsed/num_facts*1000:.2f} ms")
    
    stats = line_graph.get_statistics()
    print(f"\n📊 Final Statistics:")
    for key, val in stats.items():
        print(f"   {key}: {val:,}")
    
    # Expected: Should complete in reasonable time
    assert elapsed < 120, "Should complete in < 2 minutes"
    
    # Expected: Reasonable connectivity
    assert stats['avg_degree'] > 1, "Should have reasonable connectivity"
    
    print("✅ TEST 6 PASSED: Codex-M scale successful!")
    print(f"\n💡 Estimated time for full Codex-M (206k facts): {elapsed * 206000/20000 / 60:.1f} minutes")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("🧪 RUNNING ALL LINE GRAPH TESTS")
    print("=" * 80)
    
    tests = [
        ("Basic Construction", test_basic_construction),
        ("Isolated Facts", test_isolated_facts),
        ("Dense Graph", test_dense_graph),
        ("K-hop Neighbors", test_k_hop_neighbors),
        ("Large Scale", test_large_scale),
        ("Codex-M Scale", test_codex_m_scale),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed, None))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
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
        print("\n🎉 ALL TESTS PASSED! Line graph is ready for IKGE.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")


if __name__ == "__main__":
    run_all_tests()