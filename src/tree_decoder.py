"""
Tree Decoder - Updated Version with ILP Support
================================================

This module provides methods to decode tree structures from distance matrices.

Methods:
1. Minimum Spanning Tree (MST) - Fast, guaranteed to work
2. Local Integer Linear Program (ILP) - Optimization-based approach

Author: Based on Hewitt & Manning (2019)
"""

import numpy as np
import networkx as nx


class TreeDecoder:
    """
    Decodes tree structures from distance matrices using various algorithms.
    
    This class implements the decoding methods discussed in:
    "A Structural Probe for Finding Syntax in Word Representations"
    """
    
    def __init__(self):
        """Initialize the TreeDecoder."""
        pass
    
    def edges_to_distance_matrix(self, edges, n):
        """
        Convert a tree (given as edges) to a distance matrix.
        
        The distance between two nodes is the length of the path between them
        in the tree (i.e., the number of edges on the path).
        
        Args:
            edges: list of tuples (i, j) representing tree edges
            n: number of nodes in the tree
            
        Returns:
            distance_matrix: numpy array [n x n] where distance_matrix[i][j]
                           is the tree distance between nodes i and j
        """
        # Initialize distance matrix with infinity
        dist = np.full((n, n), np.inf)
        
        # Distance from each node to itself is 0
        for i in range(n):
            dist[i][i] = 0
        
        # Set direct edge distances to 1
        for i, j in edges:
            dist[i][j] = 1
            dist[j][i] = 1
        
        # Floyd-Warshall algorithm to compute all-pairs shortest paths
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        
        return dist
    
    def minimum_spanning_tree(self, distance_matrix):
        """
        Decode tree using Minimum Spanning Tree (Prim's algorithm).
        
        This is the primary decoding method used in the paper. It's fast,
        guaranteed to produce a valid tree, and works well in practice.
        
        Args:
            distance_matrix: numpy array [n x n] of pairwise distances
            
        Returns:
            edges: list of tuples (i, j) representing tree edges
            
        Time Complexity: O(E log V) where E = n^2 edges, V = n nodes
        """
        n = len(distance_matrix)
        
        if n == 1:
            return []
        
        # Create complete graph with distances as edge weights
        G = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                G.add_edge(i, j, weight=distance_matrix[i][j])
        
        # Compute minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        edges = list(mst.edges())
        
        return edges
    
    def local_ilp(self, distance_matrix, timeout=30):
        """
        Decode tree using Local Integer Linear Program.
        
        This formulation enforces:
        1. Exactly n-1 edges (tree property)
        2. Connectivity via flow constraints
        3. Minimizes total edge weight (sum of distances)
        
        Note: This requires PuLP to be installed. If not available,
        it falls back to MST.
        
        Args:
            distance_matrix: numpy array [n x n] of pairwise distances
            timeout: maximum time in seconds for solver (default: 30)
            
        Returns:
            edges: list of tuples (i, j) representing tree edges
            
        Raises:
            No exceptions - falls back to MST if ILP fails
        """
        n = len(distance_matrix)
        
        if n == 1:
            return []
        
        # Check if PuLP is available
        try:
            import pulp
        except ImportError:
            print("⚠️  Warning: PuLP not installed. Falling back to MST.")
            print("   Install with: pip install pulp")
            return self.minimum_spanning_tree(distance_matrix)
        
        # ====================================================================
        # CREATE OPTIMIZATION PROBLEM
        # ====================================================================
        
        prob = pulp.LpProblem("Tree_Decoding", pulp.LpMinimize)
        
        # Decision variables: x[i,j] = 1 if edge (i,j) is in the tree
        edge_vars = {}
        for i in range(n):
            for j in range(i+1, n):
                edge_vars[(i,j)] = pulp.LpVariable(
                    f"x_{i}_{j}", 
                    cat='Binary'
                )
        
        # ====================================================================
        # OBJECTIVE: Minimize total distance
        # ====================================================================
        
        objective_terms = []
        for (i,j), var in edge_vars.items():
            objective_terms.append(distance_matrix[i][j] * var)
        prob += pulp.lpSum(objective_terms)
        
        # ====================================================================
        # CONSTRAINT 1: Exactly n-1 edges (tree property)
        # ====================================================================
        
        prob += pulp.lpSum(edge_vars.values()) == n - 1, "tree_edges"
        
        # ====================================================================
        # CONSTRAINT 2: Connectivity using network flow
        # ====================================================================
        # 
        # We use a single-commodity flow formulation:
        # - Node 0 is the source that sends out n-1 units of flow
        # - Each other node consumes exactly 1 unit of flow
        # - Flow can only occur on selected edges
        #
        # This ensures the tree is connected.
        # ====================================================================
        
        # Flow variables: f[i,j] = flow from i to j
        flow_vars = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    flow_vars[(i,j)] = pulp.LpVariable(
                        f"f_{i}_{j}",
                        lowBound=0,
                        upBound=n-1,
                        cat='Continuous'
                    )
        
        # Flow conservation: each non-root node consumes 1 unit
        for k in range(1, n):
            inflow = pulp.lpSum([flow_vars[(i,k)] for i in range(n) if i != k])
            outflow = pulp.lpSum([flow_vars[(k,j)] for j in range(n) if j != k])
            prob += outflow == inflow - 1, f"flow_conservation_{k}"
        
        # Root node generates n-1 units of flow
        root_outflow = pulp.lpSum([flow_vars[(0,j)] for j in range(1, n)])
        root_inflow = pulp.lpSum([flow_vars[(i,0)] for i in range(1, n)])
        prob += root_outflow == root_inflow + (n - 1), "root_flow"
        
        # Flow can only occur on selected edges
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Get the undirected edge variable
                    edge_var = edge_vars.get((min(i,j), max(i,j)), None)
                    if edge_var is not None:
                        # Flow on (i,j) can be at most (n-1) if edge is selected
                        prob += flow_vars[(i,j)] <= (n - 1) * edge_var, \
                                f"flow_edge_{i}_{j}"
        
        # ====================================================================
        # SOLVE THE ILP
        # ====================================================================
        
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=timeout)
        
        try:
            prob.solve(solver)
        except Exception as e:
            print(f"⚠️  ILP solver error: {e}. Falling back to MST.")
            return self.minimum_spanning_tree(distance_matrix)
        
        # ====================================================================
        # EXTRACT SOLUTION
        # ====================================================================
        
        if prob.status == pulp.LpStatusOptimal:
            result_edges = []
            for (i,j), var in edge_vars.items():
                if pulp.value(var) > 0.5:  # Binary variable is essentially 1
                    result_edges.append((i,j))
            
            # Sanity check: we should have exactly n-1 edges
            if len(result_edges) == n - 1:
                return result_edges
            else:
                print(f"⚠️  ILP returned {len(result_edges)} edges, expected {n-1}.")
                print("   Falling back to MST.")
                return self.minimum_spanning_tree(distance_matrix)
        else:
            # ILP did not find optimal solution within timeout
            status_name = pulp.LpStatus.get(prob.status, f"Unknown({prob.status})")
            print(f"⚠️  ILP status: {status_name}. Falling back to MST.")
            return self.minimum_spanning_tree(distance_matrix)
    
    def decode(self, distance_matrix, method='mst', **kwargs):
        """
        Unified interface for tree decoding.
        
        Args:
            distance_matrix: numpy array [n x n] of pairwise distances
            method: 'mst' or 'ilp' (default: 'mst')
            **kwargs: additional arguments passed to the decoder
            
        Returns:
            edges: list of tuples (i, j) representing tree edges
        """
        if method.lower() == 'mst':
            return self.minimum_spanning_tree(distance_matrix)
        elif method.lower() == 'ilp':
            return self.local_ilp(distance_matrix, **kwargs)
        else:
            raise ValueError(f"Unknown decoding method: {method}")


# Convenience functions for backward compatibility
def decode_mst(distance_matrix):
    """Decode using MST (convenience function)"""
    decoder = TreeDecoder()
    return decoder.minimum_spanning_tree(distance_matrix)


def decode_ilp(distance_matrix, timeout=30):
    """Decode using ILP (convenience function)"""
    decoder = TreeDecoder()
    return decoder.local_ilp(distance_matrix, timeout=timeout)