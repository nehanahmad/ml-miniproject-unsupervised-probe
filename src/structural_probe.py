"""
Structural probe implementation
"""
import torch
import torch.nn as nn
import numpy as np

class StructuralProbe(nn.Module):
    """
    Structural probe that learns a projection matrix B
    """
    
    def __init__(self, model_dim=768, probe_rank=64):
        """
        Args:
            model_dim: dimension of input embeddings (768 for BERT-base)
            probe_rank: rank of projection matrix
        """
        super(StructuralProbe, self).__init__()
        
        self.model_dim = model_dim
        self.probe_rank = probe_rank
        
        # Projection matrix B - better initialization
        self.proj = nn.Parameter(torch.randn(model_dim, probe_rank) * 0.01)  # Smaller init
        
    def forward(self, embeddings):
        """
        Compute distance matrix from embeddings
        
        Args:
            embeddings: tensor of shape [seq_len, model_dim]
        
        Returns:
            distances: tensor of shape [seq_len, seq_len]
        """
        # Project embeddings: [seq_len, model_dim] @ [model_dim, probe_rank] -> [seq_len, probe_rank]
        projected = torch.matmul(embeddings, self.proj)
        
        # Compute pairwise squared distances
        # diff[i, j] = projected[i] - projected[j]
        diff = projected.unsqueeze(1) - projected.unsqueeze(0)  # [seq_len, seq_len, probe_rank]
        
        # Squared L2 distance
        distances = torch.sum(diff ** 2, dim=-1)  # [seq_len, seq_len]
        
        # Add small epsilon to diagonal to avoid numerical issues
        distances = distances + torch.eye(distances.size(0)) * 1e-8
        
        return distances


class ImprovedUnsupervisedProbe:
    """
    Improved unsupervised probe trainer with proper EM-style objectives
    """
    
    def __init__(self, probe, tree_decoder, learning_rate=0.001):
        self.probe = probe
        self.decoder = tree_decoder
        self.optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        self.loss_history = []
        
    def train_step_mst_proper(self, embeddings):
        """
        Proper MST training: minimize distance to MST-implied tree distances
        Not just minimizing edge distances!
        """
        # Get predicted distances
        pred_distances = self.probe(embeddings)
        n = len(embeddings)
        
        # Decode tree using MST
        dist_matrix = pred_distances.detach().cpu().numpy()
        edges = self.decoder.minimum_spanning_tree(dist_matrix)
        
        # Create TARGET distance matrix from the tree
        # This is the key: we want our distances to match tree distances
        target_distances = self.decoder.edges_to_distance_matrix(edges, n)
        target_distances = torch.tensor(target_distances, dtype=torch.float32)
        
        # MSE loss between predicted distances and tree-implied distances
        loss = torch.nn.functional.mse_loss(pred_distances, target_distances)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_step_weighted_trees(self, embeddings, n_samples=50):
        """
        Proper EM-style training with multiple weighted trees
        This is closer to what the paper describes
        """
        # Get predicted distances
        pred_distances = self.probe(embeddings)
        n = len(embeddings)
        
        # Sample multiple trees using random projections
        dist_matrix = pred_distances.detach().cpu().numpy()
        sampled_trees = self.sample_trees_via_projections(embeddings, n_samples)
        
        # Weight each tree by quality of fit (inverse of total distance)
        weights = []
        for tree_edges in sampled_trees:
            tree_weight = self.compute_tree_weight(tree_edges, dist_matrix)
            weights.append(tree_weight)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)
        
        # Create weighted average target distance matrix
        target_distances = np.zeros((n, n))
        for tree_edges, weight in zip(sampled_trees, weights):
            tree_dist = self.decoder.edges_to_distance_matrix(tree_edges, n)
            target_distances += weight * tree_dist
        
        target_distances = torch.tensor(target_distances, dtype=torch.float32)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(pred_distances, target_distances)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def sample_trees_via_projections(self, embeddings, n_samples=50):
        """
        Sample trees using random projections (as described in paper)
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        n = len(embeddings_np)
        
        # Mean center
        embeddings_centered = embeddings_np - embeddings_np.mean(axis=0)
        
        trees = []
        for _ in range(n_samples):
            # Random unit vector
            random_vec = np.random.randn(embeddings_np.shape[1])
            random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-8)
            
            # Project and sort
            projections = embeddings_centered @ random_vec
            ordering = np.argsort(projections)
            
            # Create linear tree from ordering
            edges = [(ordering[i], ordering[i+1]) for i in range(n-1)]
            trees.append(edges)
        
        return trees
    
    def compute_tree_weight(self, edges, dist_matrix):
        """
        Compute weight for a tree: 1 / sum of distances along edges
        """
        total_dist = sum(dist_matrix[i, j] for i, j in edges)
        weight = 1.0 / (total_dist + 1e-8)
        return weight
    
    def train_step_distance_weighted_walk(self, embeddings):
        """
        Random walk that respects distances (proper implementation)
        """
        pred_distances = self.probe(embeddings)
        dist_matrix = pred_distances.detach().cpu().numpy()
        n = len(embeddings)
        
        # Build tree via distance-weighted random walk
        visited = {0}
        edges = []
        unvisited = set(range(1, n))
        
        while unvisited:
            # Pick random visited node
            from_node = np.random.choice(list(visited))
            
            # Pick unvisited node with probability inversely proportional to distance
            unvisited_list = list(unvisited)
            distances = np.array([dist_matrix[from_node, j] for j in unvisited_list])
            
            # Convert distances to probabilities (closer = higher prob)
            # Use inverse distances with temperature
            inv_distances = 1.0 / (distances + 0.1)  # add epsilon
            probs = inv_distances / inv_distances.sum()
            
            # Sample
            to_node_idx = np.random.choice(len(unvisited_list), p=probs)
            to_node = unvisited_list[to_node_idx]
            
            edges.append((from_node, to_node))
            visited.add(to_node)
            unvisited.remove(to_node)
        
        # Create target distances from this tree
        target_distances = self.decoder.edges_to_distance_matrix(edges, n)
        target_distances = torch.tensor(target_distances, dtype=torch.float32)
        
        # MSE loss
        loss = torch.nn.functional.mse_loss(pred_distances, target_distances)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, embeddings_list, method='mst_proper'):
        """
        Train for one epoch with improved methods
        """
        total_loss = 0
        count = 0
        
        for embeddings in embeddings_list:
            if method == 'mst_proper':
                loss = self.train_step_mst_proper(embeddings)
            elif method == 'weighted_trees':
                loss = self.train_step_weighted_trees(embeddings, n_samples=50)
            elif method == 'weighted_walk':
                loss = self.train_step_distance_weighted_walk(embeddings)
            else:
                raise NotImplementedError(f"Method {method} not implemented")
            
            total_loss += loss
            count += 1
        
        avg_loss = total_loss / max(count, 1)
        self.loss_history.append(avg_loss)
        
        return avg_loss


# Additional helper: Add regularization to prevent collapse
class StructuralProbeWithRegularization(nn.Module):
    """
    Structural probe with regularization to prevent collapse
    """
    
    def __init__(self, model_dim=768, probe_rank=64, reg_strength=0.001):
        super().__init__()
        self.model_dim = model_dim
        self.probe_rank = probe_rank
        self.reg_strength = reg_strength
        
        # Better initialization: orthogonal
        self.proj = nn.Parameter(torch.randn(model_dim, probe_rank))
        nn.init.orthogonal_(self.proj)
        self.proj.data *= 0.01  # Scale down
        
    def forward(self, embeddings):
        """Compute distance matrix with regularization"""
        # Project
        projected = torch.matmul(embeddings, self.proj)
        
        # Pairwise distances
        diff = projected.unsqueeze(1) - projected.unsqueeze(0)
        distances = torch.sum(diff ** 2, dim=-1)
        
        # Add small epsilon to diagonal
        distances = distances + torch.eye(distances.size(0)) * 1e-8
        
        return distances
    
    def get_regularization_loss(self):
        """
        Regularization to encourage orthogonality and prevent collapse
        """
        # Encourage orthogonality of projection matrix
        proj_normalized = self.proj / (torch.norm(self.proj, dim=0, keepdim=True) + 1e-8)
        gram = torch.matmul(proj_normalized.T, proj_normalized)
        identity = torch.eye(self.probe_rank)
        ortho_loss = torch.norm(gram - identity) ** 2
        
        return self.reg_strength * ortho_loss