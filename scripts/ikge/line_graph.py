"""
Line Graph Construction for IKGE
=================================

In IKGE, facts are nodes in a "line graph" where edges connect facts that share entities.

Example:
  Facts: (Harvard, locatedIn, Boston)
         (Boston, capitalOf, Massachusetts)  
  
  These become adjacent nodes in the line graph because they share "Boston"

Paper Reference: Section 5.2, Figure 2
"""

import torch
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm


class LineGraph:
    """
    Constructs a line graph from knowledge graph triples.
    
    In a line graph:
    - Each fact (h, r, t) becomes a node
    - Two fact-nodes are connected if they share an entity
    
    This allows aggregating information across related facts.
    """
    
    def __init__(self, triples: torch.Tensor, verbose: bool = True):
        """
        Args:
            triples: Tensor of shape (N, 3) containing [head, relation, tail] IDs
            verbose: Print construction progress
        """
        self.triples = triples
        self.num_facts = len(triples)
        self.verbose = verbose
        
        # Fact ID -> (h, r, t)
        self.fact_id_to_triple = {i: tuple(triples[i].tolist()) for i in range(len(triples))}
        
        # (h, r, t) -> Fact ID
        self.triple_to_fact_id = {tuple(triples[i].tolist()): i for i in range(len(triples))}
        
        # Line graph structure
        self.edge_index = None
        self.num_edges = 0
        
        if self.verbose:
            print(f"📊 Line Graph Info:")
            print(f"   Number of facts (nodes): {self.num_facts:,}")
    
    def build(self) -> torch.Tensor:
        """
        Build line graph edge index.
        
        Returns:
            edge_index: Tensor of shape (2, E) where E is number of edges
                       edge_index[0] = source fact IDs
                       edge_index[1] = target fact IDs
        """
        if self.verbose:
            print("\n🔨 Building line graph...")
            print("   Step 1: Indexing facts by entities...")
        
        # Index facts by entities: entity_id -> list of fact_ids containing that entity
        entity_to_facts = self._index_facts_by_entities()
        
        if self.verbose:
            print(f"   Step 2: Finding adjacent facts...")
        
        # Find all adjacent fact pairs
        edges = self._find_adjacent_facts(entity_to_facts)
        
        if self.verbose:
            print(f"   Step 3: Creating edge tensor...")
        
        # Convert to tensor
        if len(edges) == 0:
            print("⚠️  WARNING: No edges found in line graph!")
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            self.edge_index = torch.tensor(edges, dtype=torch.long).T  # Shape: (2, E)
        
        self.num_edges = self.edge_index.shape[1]
        
        if self.verbose:
            print(f"\n✅ Line graph constructed!")
            print(f"   Nodes (facts): {self.num_facts:,}")
            print(f"   Edges (adjacent fact pairs): {self.num_edges:,}")
            print(f"   Avg degree: {self.num_edges / self.num_facts:.1f}")
            print(f"   Density: {self.num_edges / (self.num_facts * self.num_facts):.6f}")
        
        return self.edge_index
    
    def _index_facts_by_entities(self) -> Dict[int, List[int]]:
        """
        Create index: entity_id -> [fact_id1, fact_id2, ...]
        
        For each entity, track which facts contain it (as head or tail)
        """
        entity_to_facts = defaultdict(list)
        
        for fact_id, (h, r, t) in enumerate(self.fact_id_to_triple.values()):
            entity_to_facts[h].append(fact_id)
            entity_to_facts[t].append(fact_id)
        
        if self.verbose:
            avg_facts_per_entity = np.mean([len(facts) for facts in entity_to_facts.values()])
            print(f"      Found {len(entity_to_facts):,} unique entities")
            print(f"      Avg facts per entity: {avg_facts_per_entity:.1f}")
        
        return entity_to_facts
    
    def _find_adjacent_facts(self, entity_to_facts: Dict[int, List[int]]) -> List[Tuple[int, int]]:
        """
        Find all pairs of facts that share at least one entity.
        
        Two facts are adjacent if:
        - fact1 = (h1, r1, t1)
        - fact2 = (h2, r2, t2)
        - They share an entity: h1==h2 OR h1==t2 OR t1==h2 OR t1==t2
        
        Returns:
            List of (fact_id_1, fact_id_2) tuples
        """
        edges = []
        seen_pairs = set()  # Avoid duplicate edges
        
        # For each entity, connect all facts containing it
        for entity_id, fact_ids in tqdm(entity_to_facts.items(), 
                                       desc="Finding edges",
                                       disable=not self.verbose):
            
            # Connect every pair of facts sharing this entity
            num_facts = len(fact_ids)
            
            for i in range(num_facts):
                for j in range(num_facts):
                    if i != j:  # Don't self-loop
                        fact_i = fact_ids[i]
                        fact_j = fact_ids[j]
                        
                        # Avoid duplicates (directed graph, so (i,j) and (j,i) are both added)
                        pair = (fact_i, fact_j)
                        if pair not in seen_pairs:
                            edges.append(pair)
                            seen_pairs.add(pair)
        
        return edges
    
    def get_neighbors(self, fact_id: int) -> List[int]:
        """
        Get neighboring fact IDs for a given fact.
        
        Args:
            fact_id: ID of the fact
            
        Returns:
            List of neighboring fact IDs
        """
        if self.edge_index is None:
            raise ValueError("Line graph not built yet. Call build() first.")
        
        # Find all edges where fact_id is the source
        mask = self.edge_index[0] == fact_id
        neighbors = self.edge_index[1][mask].tolist()
        
        return neighbors
    
    def get_k_hop_neighbors(self, fact_id: int, k: int) -> Dict[int, List[int]]:
        """
        Get k-hop neighbors of a fact.
        
        Args:
            fact_id: ID of the fact
            k: Number of hops
            
        Returns:
            Dictionary mapping hop number -> list of fact IDs at that hop
        """
        if self.edge_index is None:
            raise ValueError("Line graph not built yet. Call build() first.")
        
        neighbors_by_hop = {0: [fact_id]}
        visited = {fact_id}
        
        for hop in range(1, k + 1):
            current_hop_neighbors = []
            
            # Get neighbors of all facts at previous hop
            for prev_fact in neighbors_by_hop[hop - 1]:
                neighbors = self.get_neighbors(prev_fact)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        current_hop_neighbors.append(neighbor)
                        visited.add(neighbor)
            
            neighbors_by_hop[hop] = current_hop_neighbors
        
        return neighbors_by_hop
    
    def visualize_sample(self, num_facts: int = 5):
        """
        Print a small sample of the line graph for debugging.
        
        Args:
            num_facts: Number of facts to show
        """
        print(f"\n🔍 Sample Line Graph Structure (first {num_facts} facts):")
        print("=" * 80)
        
        for fact_id in range(min(num_facts, self.num_facts)):
            h, r, t = self.fact_id_to_triple[fact_id]
            neighbors = self.get_neighbors(fact_id)
            
            print(f"\nFact {fact_id}: (entity_{h}, relation_{r}, entity_{t})")
            print(f"  → {len(neighbors)} neighbors:")
            
            for neighbor_id in neighbors[:5]:  # Show first 5 neighbors
                nh, nr, nt = self.fact_id_to_triple[neighbor_id]
                print(f"     • Fact {neighbor_id}: (entity_{nh}, relation_{nr}, entity_{nt})")
            
            if len(neighbors) > 5:
                print(f"     ... and {len(neighbors) - 5} more")
        
        print("=" * 80)
    
    def get_statistics(self) -> Dict:
        """
        Compute line graph statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.edge_index is None:
            raise ValueError("Line graph not built yet. Call build() first.")
        
        # Degree distribution
        degrees = torch.zeros(self.num_facts, dtype=torch.long)
        unique, counts = torch.unique(self.edge_index[0], return_counts=True)
        degrees[unique] = counts
        
        stats = {
            'num_nodes': self.num_facts,
            'num_edges': self.num_edges,
            'avg_degree': float(degrees.float().mean()),
            'max_degree': int(degrees.max()),
            'min_degree': int(degrees.min()),
            'isolated_nodes': int((degrees == 0).sum()),
        }
        
        return stats


# ============================================================================
# Helper Functions
# ============================================================================

def create_line_graph(triples: torch.Tensor, verbose: bool = True) -> Tuple[torch.Tensor, LineGraph]:
    """
    Convenience function to create line graph from triples.
    
    Args:
        triples: Tensor of shape (N, 3) with [head, relation, tail] IDs
        verbose: Print progress
        
    Returns:
        edge_index: Tensor of shape (2, E)
        line_graph: LineGraph object (for additional operations)
    """
    line_graph = LineGraph(triples, verbose=verbose)
    edge_index = line_graph.build()
    return edge_index, line_graph


def test_line_graph():
    """
    Test line graph construction with a simple example.
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING LINE GRAPH CONSTRUCTION")
    print("=" * 80)
    
    # Create simple test knowledge graph
    # Entities: 0=Harvard, 1=Boston, 2=Massachusetts, 3=USA
    # Relations: 0=locatedIn, 1=capitalOf, 2=partOf
    test_triples = torch.tensor([
        [0, 0, 1],  # Harvard locatedIn Boston
        [1, 1, 2],  # Boston capitalOf Massachusetts  
        [2, 2, 3],  # Massachusetts partOf USA
        [0, 2, 2],  # Harvard partOf Massachusetts (shares entity with fact 1 and 2)
    ])
    
    print("\n📋 Test Knowledge Graph:")
    entity_names = {0: "Harvard", 1: "Boston", 2: "Massachusetts", 3: "USA"}
    relation_names = {0: "locatedIn", 1: "capitalOf", 2: "partOf"}
    
    for i, (h, r, t) in enumerate(test_triples):
        print(f"  Fact {i}: ({entity_names[h.item()]}, {relation_names[r.item()]}, {entity_names[t.item()]})")
    
    # Build line graph
    edge_index, line_graph = create_line_graph(test_triples, verbose=True)
    
    # Verify adjacencies
    print("\n✅ Expected Adjacencies:")
    print("  - Fact 0 and Fact 1 share 'Boston' → should be adjacent")
    print("  - Fact 1 and Fact 2 share 'Massachusetts' → should be adjacent")
    print("  - Fact 0 and Fact 3 share 'Harvard' → should be adjacent")
    print("  - Fact 1 and Fact 3 share 'Massachusetts' → should be adjacent")
    print("  - Fact 2 and Fact 3 share 'Massachusetts' → should be adjacent")
    
    # Show actual structure
    line_graph.visualize_sample(num_facts=4)
    
    # Statistics
    stats = line_graph.get_statistics()
    print("\n📊 Line Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run test
    test_line_graph()
    
    print("\n" + "=" * 80)
    print("💡 Usage Example:")
    print("=" * 80)
    print("""
from line_graph import create_line_graph

# Your training triples
train_triples = torch.tensor([
    [h1, r1, t1],
    [h2, r2, t2],
    ...
])

# Create line graph
fact_edge_index, line_graph = create_line_graph(train_triples, verbose=True)

# Use in model
model = IKGEModel(
    ...,
    fact_edge_index=fact_edge_index  # Pass this to aggregation module
)

# Get neighbors of a fact
neighbors = line_graph.get_neighbors(fact_id=0)

# Get k-hop neighbors
k_hop_neighbors = line_graph.get_k_hop_neighbors(fact_id=0, k=2)
""")