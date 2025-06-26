# connectivity_utils.py
"""
Comprehensive Connectivity Toolbox for Network Analysis

This module provides various connectivity patterns and utilities for constructing
custom network topologies. Researchers can use predefined patterns or create
their own connectivity matrices for network analysis and modeling.

Author: Connectivity Toolbox
Version: 2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Tuple, List, Union
import random


# ============================================================================
# BASIC CONNECTIVITY PATTERNS
# ============================================================================

def tnet9x():
    """
    Load the 9-node "X" topology connectivity matrix.
    Returns a (9 x 9) binary adjacency matrix.
    """
    C = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1]
    ], dtype=int)
    return C


def tnet9d():
    """
    Load the 9-node "D" topology connectivity matrix.
    Returns a (9 x 9) binary adjacency matrix.
    """
    C = np.array([
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1]
    ], dtype=int)
    return C


# ============================================================================
# STANDARD NETWORK TOPOLOGIES
# ============================================================================

def ring_network(n: int, directed: bool = False) -> np.ndarray:
    """
    Create a ring network topology.
    
    Args:
        n: Number of nodes
        directed: If True, creates directed ring; if False, undirected
    
    Returns:
        n x n adjacency matrix
    """
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        C[i, (i + 1) % n] = 1
        if not directed:
            C[(i + 1) % n, i] = 1
    return C


def star_network(n: int, center: int = 0, directed: bool = False) -> np.ndarray:
    """
    Create a star network topology.
    
    Args:
        n: Number of nodes
        center: Index of the center node (default: 0)
        directed: If True, creates directed star; if False, undirected
    
    Returns:
        n x n adjacency matrix
    """
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        if i != center:
            C[center, i] = 1
            if not directed:
                C[i, center] = 1
    return C


def fully_connected(n: int, directed: bool = False, self_loops: bool = False) -> np.ndarray:
    """
    Create a fully connected network.
    
    Args:
        n: Number of nodes
        directed: If True, creates directed network; if False, undirected
        self_loops: If True, includes self-connections
    
    Returns:
        n x n adjacency matrix
    """
    C = np.ones((n, n), dtype=int)
    if not self_loops:
        np.fill_diagonal(C, 0)
    if not directed:
        C = np.triu(C) + np.triu(C).T
    return C


def lattice_network(rows: int, cols: int, periodic: bool = False) -> np.ndarray:
    """
    Create a 2D lattice network.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        periodic: If True, creates periodic boundary conditions
    
    Returns:
        n x n adjacency matrix where n = rows * cols
    """
    n = rows * cols
    C = np.zeros((n, n), dtype=int)
    
    for i in range(rows):
        for j in range(cols):
            node = i * cols + j
            
            # Right neighbor
            if j < cols - 1:
                C[node, node + 1] = 1
            elif periodic:
                C[node, i * cols] = 1
            
            # Left neighbor
            if j > 0:
                C[node, node - 1] = 1
            elif periodic:
                C[node, i * cols + cols - 1] = 1
            
            # Bottom neighbor
            if i < rows - 1:
                C[node, node + cols] = 1
            elif periodic:
                C[node, j] = 1
            
            # Top neighbor
            if i > 0:
                C[node, node - cols] = 1
            elif periodic:
                C[node, (rows - 1) * cols + j] = 1
    
    return C


def small_world_network(n: int, k: int, p: float, directed: bool = False) -> np.ndarray:
    """
    Create a Watts-Strogatz small-world network.
    
    Args:
        n: Number of nodes
        k: Average degree (must be even)
        p: Rewiring probability
        directed: If True, creates directed network
    
    Returns:
        n x n adjacency matrix
    """
    if k % 2 != 0:
        raise ValueError("k must be even for small-world network")
    
    # Start with ring network
    C = ring_network(n, directed)
    
    # Add k-2 additional connections per node
    for i in range(n):
        for j in range(1, k // 2):
            target = (i + j) % n
            C[i, target] = 1
            if not directed:
                C[target, i] = 1
    
    # Rewire with probability p
    for i in range(n):
        for j in range(n):
            if C[i, j] == 1 and random.random() < p:
                # Remove connection
                C[i, j] = 0
                if not directed:
                    C[j, i] = 0
                
                # Add new random connection
                new_target = random.randint(0, n-1)
                while new_target == i or C[i, new_target] == 1:
                    new_target = random.randint(0, n-1)
                
                C[i, new_target] = 1
                if not directed:
                    C[new_target, i] = 1
    
    return C


def scale_free_network(n: int, m: int, directed: bool = False) -> np.ndarray:
    """
    Create a Barabási-Albert scale-free network.
    
    Args:
        n: Number of nodes
        m: Number of edges to attach from a new node to existing nodes
        directed: If True, creates directed network
    
    Returns:
        n x n adjacency matrix
    """
    C = np.zeros((n, n), dtype=int)
    
    # Start with m+1 fully connected nodes
    for i in range(m + 1):
        for j in range(m + 1):
            if i != j:
                C[i, j] = 1
    
    # Add remaining nodes
    for i in range(m + 1, n):
        # Calculate degree distribution
        degrees = np.sum(C, axis=1)
        total_degree = np.sum(degrees)
        
        # Preferential attachment
        for _ in range(m):
            # Choose target based on degree
            probs = degrees / total_degree
            target = np.random.choice(n, p=probs)
            
            # Avoid self-loops and duplicate edges
            while target == i or C[i, target] == 1:
                target = np.random.choice(n, p=probs)
            
            C[i, target] = 1
            if not directed:
                C[target, i] = 1
            
            # Update degrees
            degrees[i] += 1
            degrees[target] += 1
            total_degree += 2
    
    return C


# ============================================================================
# MODULAR NETWORK TOPOLOGIES
# ============================================================================

def modular_network(n_modules: int, nodes_per_module: int, 
                   p_within: float = 0.8, p_between: float = 0.1,
                   directed: bool = False) -> np.ndarray:
    """
    Create a modular network with specified connectivity probabilities.
    
    Args:
        n_modules: Number of modules
        nodes_per_module: Number of nodes per module
        p_within: Probability of connection within modules
        p_between: Probability of connection between modules
        directed: If True, creates directed network
    
    Returns:
        n x n adjacency matrix where n = n_modules * nodes_per_module
    """
    n = n_modules * nodes_per_module
    C = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
                
            # Determine if nodes are in same module
            module_i = i // nodes_per_module
            module_j = j // nodes_per_module
            
            if module_i == module_j:
                # Within module connection
                if random.random() < p_within:
                    C[i, j] = 1
            else:
                # Between module connection
                if random.random() < p_between:
                    C[i, j] = 1
    
    # Make undirected if specified
    if not directed:
        C = np.maximum(C, C.T)
    
    return C


def hierarchical_network(n_levels: int, nodes_per_level: List[int],
                        p_within: float = 0.8, p_between: float = 0.3,
                        directed: bool = False) -> np.ndarray:
    """
    Create a hierarchical network structure.
    
    Args:
        n_levels: Number of hierarchy levels
        nodes_per_level: List of nodes at each level
        p_within: Probability of connection within same level
        p_between: Probability of connection between adjacent levels
        directed: If True, creates directed network
    
    Returns:
        n x n adjacency matrix where n = sum(nodes_per_level)
    """
    n = sum(nodes_per_level)
    C = np.zeros((n, n), dtype=int)
    
    # Calculate starting indices for each level
    start_indices = [0]
    for i in range(n_levels - 1):
        start_indices.append(start_indices[-1] + nodes_per_level[i])
    
    # Within-level connections
    for level in range(n_levels):
        start_idx = start_indices[level]
        end_idx = start_idx + nodes_per_level[level]
        
        for i in range(start_idx, end_idx):
            for j in range(start_idx, end_idx):
                if i != j and random.random() < p_within:
                    C[i, j] = 1
    
    # Between-level connections
    for level in range(n_levels - 1):
        current_start = start_indices[level]
        current_end = current_start + nodes_per_level[level]
        next_start = start_indices[level + 1]
        next_end = next_start + nodes_per_level[level + 1]
        
        for i in range(current_start, current_end):
            for j in range(next_start, next_end):
                if random.random() < p_between:
                    C[i, j] = 1
    
    # Make undirected if specified
    if not directed:
        C = np.maximum(C, C.T)
    
    return C


# ============================================================================
# CUSTOM NETWORK CONSTRUCTORS
# ============================================================================

def custom_connectivity(n: int, connection_pattern: str = "random", 
                       density: float = 0.1, seed: Optional[int] = None,
                       **kwargs) -> np.ndarray:
    """
    Create custom connectivity patterns.
    
    Args:
        n: Number of nodes
        connection_pattern: Type of pattern ("random", "sparse", "dense", "banded")
        density: Connection density (for random patterns)
        seed: Random seed for reproducibility
        **kwargs: Additional parameters for specific patterns
    
    Returns:
        n x n adjacency matrix
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if connection_pattern == "random":
        C = np.random.random((n, n)) < density
        np.fill_diagonal(C, 0)  # No self-loops
        return C.astype(int)
    
    elif connection_pattern == "sparse":
        # Create sparse matrix with few connections
        C = np.zeros((n, n), dtype=int)
        n_connections = int(density * n * (n - 1) / 2)
        connections = random.sample([(i, j) for i in range(n) for j in range(i+1, n)], n_connections)
        for i, j in connections:
            C[i, j] = C[j, i] = 1
        return C
    
    elif connection_pattern == "dense":
        # Create dense matrix with many connections
        C = np.ones((n, n), dtype=int)
        np.fill_diagonal(C, 0)  # No self-loops
        return C
    
    elif connection_pattern == "banded":
        # Create banded matrix
        bandwidth = kwargs.get('bandwidth', 2)
        C = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
                if i != j:
                    C[i, j] = 1
        return C
    
    else:
        raise ValueError(f"Unknown connection pattern: {connection_pattern}")


def from_adjacency_list(adj_list: List[List[int]], n: Optional[int] = None) -> np.ndarray:
    """
    Create connectivity matrix from adjacency list.
    
    Args:
        adj_list: List of lists where adj_list[i] contains neighbors of node i
        n: Number of nodes (if None, inferred from adj_list)
    
    Returns:
        n x n adjacency matrix
    """
    if n is None:
        n = len(adj_list)
    
    C = np.zeros((n, n), dtype=int)
    for i, neighbors in enumerate(adj_list):
        for j in neighbors:
            if 0 <= j < n:
                C[i, j] = 1
    
    return C


def from_edge_list(edges: List[Tuple[int, int]], n: Optional[int] = None, 
                   directed: bool = False) -> np.ndarray:
    """
    Create connectivity matrix from edge list.
    
    Args:
        edges: List of (source, target) tuples
        n: Number of nodes (if None, inferred from edges)
        directed: If True, creates directed network
    
    Returns:
        n x n adjacency matrix
    """
    if n is None:
        n = max(max(edge) for edge in edges) + 1
    
    C = np.zeros((n, n), dtype=int)
    for source, target in edges:
        if 0 <= source < n and 0 <= target < n:
            C[source, target] = 1
            if not directed:
                C[target, source] = 1
    
    return C


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_network(C: np.ndarray, layout: str = "spring", 
                     node_size: int = 300, node_color: str = "lightblue",
                     edge_color: str = "gray", figsize: Tuple[int, int] = (8, 6),
                     title: str = "Network Visualization") -> None:
    """
    Visualize a network using NetworkX and Matplotlib.
    
    Args:
        C: Adjacency matrix
        layout: Layout algorithm ("spring", "circular", "random", "shell")
        node_size: Size of nodes
        node_color: Color of nodes
        edge_color: Color of edges
        figsize: Figure size
        title: Plot title
    """
    G = nx.from_numpy_array(C)
    
    plt.figure(figsize=figsize)
    
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    nx.draw(G, pos, with_labels=True, node_size=node_size, 
            node_color=node_color, edge_color=edge_color, 
            font_size=10, font_weight='bold')
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def network_statistics(C: np.ndarray) -> dict:
    """
    Calculate basic network statistics.
    
    Args:
        C: Adjacency matrix
    
    Returns:
        Dictionary containing network statistics
    """
    n = C.shape[0]
    n_edges = np.sum(C) // 2 if not np.any(C != C.T) else np.sum(C)
    
    stats = {
        'n_nodes': n,
        'n_edges': n_edges,
        'density': n_edges / (n * (n - 1) / 2),
        'avg_degree': 2 * n_edges / n,
        'is_directed': np.any(C != C.T),
        'has_self_loops': np.any(np.diag(C) != 0)
    }
    
    # Calculate degree distribution
    degrees = np.sum(C, axis=1)
    stats['degree_distribution'] = degrees
    stats['max_degree'] = np.max(degrees)
    stats['min_degree'] = np.min(degrees)
    
    return stats


def load_connectivity(name: str, **kwargs) -> np.ndarray:
    """
    Load a predefined connectivity by name.
    
    Args:
        name: Connectivity name
        **kwargs: Additional parameters for the connectivity function
    
    Returns:
        n x n adjacency matrix
    """
    name = name.lower()
    
    # Basic topologies
    if name == 'tnet9x':
        return tnet9x()
    elif name == 'tnet9d':
        return tnet9d()
    
    # Standard topologies
    elif name.startswith('ring'):
        n = kwargs.get('n', 10)
        directed = kwargs.get('directed', False)
        return ring_network(n, directed)
    
    elif name.startswith('star'):
        n = kwargs.get('n', 10)
        center = kwargs.get('center', 0)
        directed = kwargs.get('directed', False)
        return star_network(n, center, directed)
    
    elif name.startswith('fully_connected'):
        n = kwargs.get('n', 10)
        directed = kwargs.get('directed', False)
        self_loops = kwargs.get('self_loops', False)
        return fully_connected(n, directed, self_loops)
    
    elif name.startswith('lattice'):
        rows = kwargs.get('rows', 5)
        cols = kwargs.get('cols', 5)
        periodic = kwargs.get('periodic', False)
        return lattice_network(rows, cols, periodic)
    
    elif name.startswith('small_world'):
        n = kwargs.get('n', 20)
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.1)
        directed = kwargs.get('directed', False)
        return small_world_network(n, k, p, directed)
    
    elif name.startswith('scale_free'):
        n = kwargs.get('n', 20)
        m = kwargs.get('m', 2)
        directed = kwargs.get('directed', False)
        return scale_free_network(n, m, directed)
    
    elif name.startswith('modular'):
        n_modules = kwargs.get('n_modules', 3)
        nodes_per_module = kwargs.get('nodes_per_module', 5)
        p_within = kwargs.get('p_within', 0.8)
        p_between = kwargs.get('p_between', 0.1)
        directed = kwargs.get('directed', False)
        return modular_network(n_modules, nodes_per_module, p_within, p_between, directed)
    
    else:
        raise ValueError(f"Unknown connectivity '{name}'. Available options: "
                        f"tnet9x, tnet9d, ring, star, fully_connected, lattice, "
                        f"small_world, scale_free, modular")


def get_available_connectivities() -> List[str]:
    """
    Get list of available connectivity types.
    
    Returns:
        List of connectivity names
    """
    return [
        'tnet9x', 'tnet9d', 'ring', 'star', 'fully_connected', 
        'lattice', 'small_world', 'scale_free', 'modular'
    ]


# ============================================================================
# EXAMPLE USAGE AND TUTORIAL FUNCTIONS
# ============================================================================

def tutorial_examples():
    """
    Demonstrate various connectivity patterns with examples.
    """
    print("=== Connectivity Toolbox Tutorial Examples ===\n")
    
    # Example 1: Basic topologies
    print("1. Basic Topologies:")
    print("   - Ring network (10 nodes):", ring_network(10).shape)
    print("   - Star network (8 nodes):", star_network(8).shape)
    print("   - Fully connected (5 nodes):", fully_connected(5).shape)
    
    # Example 2: Complex topologies
    print("\n2. Complex Topologies:")
    print("   - Small-world network (20 nodes):", small_world_network(20, 4, 0.1).shape)
    print("   - Scale-free network (15 nodes):", scale_free_network(15, 2).shape)
    print("   - Modular network (3 modules, 4 nodes each):", 
          modular_network(3, 4, 0.8, 0.1).shape)
    
    # Example 3: Custom patterns
    print("\n3. Custom Patterns:")
    print("   - Random network (10 nodes, density 0.3):", 
          custom_connectivity(10, "random", 0.3).shape)
    print("   - Banded network (8 nodes, bandwidth 2):", 
          custom_connectivity(8, "banded", bandwidth=2).shape)
    
    # Example 4: Network statistics
    print("\n4. Network Statistics:")
    C = small_world_network(10, 4, 0.2)
    stats = network_statistics(C)
    print(f"   - Nodes: {stats['n_nodes']}")
    print(f"   - Edges: {stats['n_edges']}")
    print(f"   - Density: {stats['density']:.3f}")
    print(f"   - Average degree: {stats['avg_degree']:.1f}")
    
    print("\n=== Tutorial Complete ===")


if __name__ == "__main__":
    # Run tutorial examples
    tutorial_examples()

