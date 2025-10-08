"""
Methods for decoding trees from distance matrices
"""
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, to_tree

class TreeDecoder:
    """Decode tree structures from distance matrices"""
    
    def __init__(self):
        pass
    
    def minimum_spanning_tree(self, distance_matrix):
        """
        Find minimum spanning tree using Prim's algorithm
        
        Args:
            distance_matrix: numpy array [n x n]
        
        Returns:
            edges: list of tuples (i, j) representing edges
        """
        n = len(distance_matrix)
        
        # Create weighted graph
        G = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=distance_matrix[i][j])
        
        # Find MST
        mst = nx.minimum_spanning_tree(G)
        
        # Extract edges
        edges = list(mst.edges())
        
        return edges
    
    def hierarchical_clustering(self, distance_matrix):
        """
        Agglomerative hierarchical clustering
        
        Args:
            distance_matrix: numpy array [n x n]
        
        Returns:
            linkage_matrix: scipy linkage matrix
        """
        # Convert distance matrix to condensed form
        from scipy.spatial.distance import squareform
        
        # Make sure matrix is symmetric
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Convert to condensed format
        condensed = squareform(distance_matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed, method='average')
        
        return linkage_matrix
    
    def edges_to_distance_matrix(self, edges, n):
        """
        Convert edge list to distance matrix
        
        Args:
            edges: list of tuples (i, j)
            n: number of nodes
        
        Returns:
            distance_matrix: numpy array [n x n]
        """
        # Build graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        
        # Compute all shortest paths
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    distances[i][j] = 0
                else:
                    try:
                        distances[i][j] = nx.shortest_path_length(G, i, j)
                    except nx.NetworkXNoPath:
                        distances[i][j] = float('inf')
        
        return distances
    
    def evaluate_tree_fit(self, predicted_distances, gold_distances):
        """
        Compute Frobenius norm between predicted and gold distances
        
        Args:
            predicted_distances: numpy array [n x n]
            gold_distances: numpy array [n x n]
        
        Returns:
            frobenius_norm: float
        """
        diff = predicted_distances - gold_distances
        return np.linalg.norm(diff, ord='fro')
    
    def weighted_edge_accuracy(self, predicted_edges, gold_edges_list, weights):
        """
        Compute weighted edge accuracy for mixture of trees
        
        Args:
            predicted_edges: set of edges
            gold_edges_list: list of sets of edges
            weights: list of weights for each gold tree
        
        Returns:
            accuracy: float
        """
        predicted_set = set(predicted_edges)
        
        total_overlap = 0
        total_weight = sum(weights)
        
        for gold_edges, weight in zip(gold_edges_list, weights):
            overlap = len(predicted_set & set(gold_edges))
            n_edges = len(gold_edges)
            total_overlap += weight * (overlap / n_edges if n_edges > 0 else 0)
        
        return total_overlap / total_weight if total_weight > 0 else 0