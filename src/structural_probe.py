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
        
        # Projection matrix B
        self.proj = nn.Parameter(torch.randn(model_dim, probe_rank))
        
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
        
        return distances


class UnsupervisedProbe:
    """
    Train structural probe in unsupervised manner using EM-style algorithm
    """
    
    def __init__(self, probe, tree_decoder, learning_rate=0.001):
        """
        Args:
            probe: StructuralProbe instance
            tree_decoder: TreeDecoder instance
            learning_rate: learning rate for optimizer
        """
        self.probe = probe
        self.decoder = tree_decoder
        self.optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        self.loss_history = []
        
    def train_step_mst(self, embeddings):
        """
        Single training step using MST as pseudo-labels
        
        Args:
            embeddings: tensor of shape [seq_len, model_dim]
        
        Returns:
            loss: float
        """
        # Get predicted distances
        pred_distances = self.probe(embeddings)
        
        # Decode tree using MST
        dist_matrix = pred_distances.detach().cpu().numpy()
        edges = self.decoder.minimum_spanning_tree(dist_matrix)
        
        # Convert edges back to distance matrix (gold labels)
        n = len(embeddings)
        gold_distances = self.decoder.edges_to_distance_matrix(edges, n)
        gold_distances = torch.tensor(gold_distances, dtype=torch.float32)
        
        # Compute loss (mean squared error)
        loss = torch.mean((pred_distances - gold_distances) ** 2)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, embeddings_list, method='mst'):
        """
        Train for one epoch
        
        Args:
            embeddings_list: list of embedding tensors
            method: 'mst' or 'random_walk'
        
        Returns:
            avg_loss: average loss for the epoch
        """
        total_loss = 0
        
        for embeddings in embeddings_list:
            if method == 'mst':
                loss = self.train_step_mst(embeddings)
            else:
                raise NotImplementedError(f"Method {method} not implemented")
            
            total_loss += loss
        
        avg_loss = total_loss / len(embeddings_list)
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, embeddings, gold_edges):
        """
        Evaluate probe on a sentence
        
        Args:
            embeddings: tensor of shape [seq_len, model_dim]
            gold_edges: list of tuples (i, j) representing gold tree edges
        
        Returns:
            uuas: float (edge accuracy)
        """
        # Get predicted distances
        pred_distances = self.probe(embeddings)
        dist_matrix = pred_distances.detach().cpu().numpy()
        
        # Decode tree
        pred_edges = self.decoder.minimum_spanning_tree(dist_matrix)
        
        # Compute UUAS
        from src.utils import compute_uuas
        score = compute_uuas(set(pred_edges), set(gold_edges))
        
        return score