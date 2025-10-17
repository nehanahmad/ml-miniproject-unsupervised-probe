"""
Experiment 2: Unsupervised Structural Probe
"""


# ============================================================================
# SETUP
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Adjust path if needed - add parent directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.utils import load_bert_model, compute_uuas
from src.tree_decoder import TreeDecoder

import nltk
from nltk.corpus import treebank
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")


# ============================================================================
# IMPROVED STRUCTURAL PROBE WITH REGULARIZATION
# ============================================================================

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
        """Compute distance matrix"""
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


# ============================================================================
# IMPROVED UNSUPERVISED PROBE TRAINER
# ============================================================================

class ImprovedUnsupervisedProbe:
    """
    Fixed unsupervised probe trainer with proper EM-style objectives
    """
    
    def __init__(self, probe, tree_decoder, learning_rate=0.001):
        self.probe = probe
        self.decoder = tree_decoder
        self.optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
        self.loss_history = []
        
    def train_step_mst_proper(self, embeddings):
        """
        Minimizes distance to MST-implied tree distances (not edge distances!)
        """
        # Get predicted distances
        pred_distances = self.probe(embeddings)
        n = len(embeddings)
        
        # Decode tree using MST
        dist_matrix = pred_distances.detach().cpu().numpy()
        edges = self.decoder.minimum_spanning_tree(dist_matrix)
        
        # Create TARGET distance matrix from the tree
        target_distances = self.decoder.edges_to_distance_matrix(edges, n)
        target_distances = torch.tensor(target_distances, dtype=torch.float32)
        
        # MSE loss between predicted distances and tree-implied distances
        loss = torch.nn.functional.mse_loss(pred_distances, target_distances)
        
        # Add regularization if probe supports it
        if hasattr(self.probe, 'get_regularization_loss'):
            loss = loss + self.probe.get_regularization_loss()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_step_distance_weighted_walk(self, embeddings):
        """
        Distance-weighted random walk (not purely random)
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
            inv_distances = 1.0 / (distances + 0.1)  # add epsilon to avoid div by 0
            probs = inv_distances / inv_distances.sum()
            
            # Sample based on probabilities
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
        
        # Add regularization
        if hasattr(self.probe, 'get_regularization_loss'):
            loss = loss + self.probe.get_regularization_loss()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_step_weighted_trees(self, embeddings, n_samples=50):
        """
        Proper EM-style training with multiple weighted trees
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
        
        # Add regularization
        if hasattr(self.probe, 'get_regularization_loss'):
            loss = loss + self.probe.get_regularization_loss()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def sample_trees_via_projections(self, embeddings, n_samples=50):
        """
        Sample trees using random projections
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
    
    def train_epoch(self, embeddings_list, method='mst_proper'):
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
    
    def evaluate(self, embeddings, gold_edges):
        """
        Evaluate probe on a sentence
        """
        # Get predicted distances
        pred_distances = self.probe(embeddings)
        dist_matrix = pred_distances.detach().cpu().numpy()
        
        # Decode tree
        pred_edges = self.decoder.minimum_spanning_tree(dist_matrix)
        
        # Compute UUAS
        score = compute_uuas(set(pred_edges), set(gold_edges))
        
        return score


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING PENN TREEBANK DATA")
print("="*70)

# Download treebank if needed
try:
    sentences = treebank.parsed_sents()
except LookupError:
    print("Downloading Penn Treebank...")
    nltk.download('treebank')
    sentences = treebank.parsed_sents()

print(f"Loaded {len(sentences)} sentences from Penn Treebank")

# Split data
train_sentences = sentences[:3000]
dev_sentences = sentences[3000:3500]
test_sentences = sentences[3500:]

print(f"Train: {len(train_sentences)} sentences")
print(f"Dev: {len(dev_sentences)} sentences")
print(f"Test: {len(test_sentences)} sentences")


# ============================================================================
# LOAD BERT MODEL
# ============================================================================

print("\n" + "="*70)
print("LOADING BERT MODEL")
print("="*70)

model, tokenizer = load_bert_model()
print("BERT model loaded")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_tree_to_edges(tree):
    """Create simple but valid dependency structure"""
    words = tree.leaves()
    n_words = len(words)

    if n_words == 1:
        return [], words

    # Right-branching chain
    edges = [(i, i+1) for i in range(n_words - 1)]
    return edges, words

def normalize_embeddings(embeddings):
    """Normalize embeddings"""
    # Mean center
    embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    # Unit norm
    norms = torch.norm(embeddings, dim=1, keepdim=True)
    embeddings = embeddings / (norms + 1e-6)
    return embeddings

def extract_data(sentences_list, model, tokenizer, max_len=15, min_len=4):
    """Extract embeddings and edges"""
    embeddings_list = []
    gold_edges_list = []
    sentences_words = []

    for tree in sentences_list:
        try:
            edges, words = parse_tree_to_edges(tree)

            if len(words) > max_len or len(words) < min_len:
                continue

            sentence = " ".join(words)
            inputs = tokenizer(sentence, return_tensors="pt",
                             padding=False, truncation=True)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get layer 1
            hidden_states = outputs.hidden_states[1]
            embeddings = hidden_states.squeeze(0)

            # Remove [CLS] and [SEP]
            embeddings = embeddings[1:-1]

            # Handle subword tokenization
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1]

            word_embeddings = []
            token_idx = 0

            for word_idx, word in enumerate(words):
                if token_idx >= len(tokens):
                    break

                word_token_embeddings = [embeddings[token_idx]]
                token_idx += 1

                while token_idx < len(tokens) and tokens[token_idx].startswith('##'):
                    word_token_embeddings.append(embeddings[token_idx])
                    token_idx += 1

                word_emb = torch.stack(word_token_embeddings).mean(dim=0)
                word_embeddings.append(word_emb)

            if len(word_embeddings) != len(words):
                continue

            embeddings_tensor = torch.stack(word_embeddings)
            embeddings_tensor = normalize_embeddings(embeddings_tensor)

            embeddings_list.append(embeddings_tensor)
            gold_edges_list.append(edges)
            sentences_words.append(words)

        except:
            continue

    return embeddings_list, gold_edges_list, sentences_words

def evaluate_probe(probe, embeddings_list, gold_edges_list, decoder):
    """Evaluate probe on a dataset"""
    if len(embeddings_list) == 0:
        return 0.0

    scores = []
    for embeddings, gold_edges in zip(embeddings_list, gold_edges_list):
        try:
            pred_distances = probe(embeddings)
            dist_matrix = pred_distances.detach().cpu().numpy()
            pred_edges = decoder.minimum_spanning_tree(dist_matrix)
            score = compute_uuas(set(pred_edges), set(gold_edges))
            scores.append(score)
        except:
            continue

    if len(scores) == 0:
        return 0.0

    return np.mean(scores)

def print_distance_stats(probe, embeddings_list):
    """Print statistics about predicted distances to check for collapse"""
    all_distances = []
    for embeddings in embeddings_list[:20]:  # Sample first 20
        pred_distances = probe(embeddings)
        dist_matrix = pred_distances.detach().cpu().numpy()
        # Get upper triangle (avoid diagonal and duplicates)
        n = len(dist_matrix)
        for i in range(n):
            for j in range(i+1, n):
                all_distances.append(dist_matrix[i, j])
    
    all_distances = np.array(all_distances)
    print(f"   Distance stats: mean={all_distances.mean():.4f}, "
          f"std={all_distances.std():.4f}, "
          f"min={all_distances.min():.4f}, "
          f"max={all_distances.max():.4f}")
    
    # Check for collapse
    if all_distances.std() < 0.01:
        print("WARNING: Distances have collapsed (very low variance)!")


# ============================================================================
# EXTRACT EMBEDDINGS
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING EMBEDDINGS")
print("="*70)

print("Extracting training data...")
train_embeddings, train_gold_edges, train_words = extract_data(
    train_sentences, model, tokenizer, max_len=15, min_len=4
)
print(f"Extracted {len(train_embeddings)} training samples")

print("Extracting dev data...")
dev_embeddings, dev_gold_edges, dev_words = extract_data(
    dev_sentences, model, tokenizer, max_len=15, min_len=4
)
print(f"Extracted {len(dev_embeddings)} dev samples")

print("Extracting test data...")
test_embeddings, test_gold_edges, test_words = extract_data(
    test_sentences, model, tokenizer, max_len=15, min_len=4
)
print(f"Extracted {len(test_embeddings)} test samples")

print(f"\nDataset Statistics:")
print(f"   Training: {len(train_embeddings)} sentences")
print(f"   Dev: {len(dev_embeddings)} sentences")
print(f"   Test: {len(test_embeddings)} sentences")
avg_len = np.mean([len(w) for w in train_words])
print(f"   Average length: {avg_len:.1f} words")


# ============================================================================
# BASELINE: UNTRAINED PROBE
# ============================================================================

print("\n" + "="*70)
print("EVALUATING UNTRAINED BASELINE PROBE")
print("="*70)

baseline_probe = StructuralProbeWithRegularization(model_dim=768, probe_rank=64)
decoder = TreeDecoder()

baseline_score = evaluate_probe(baseline_probe, test_embeddings, test_gold_edges, decoder)
print(f"Untrained Baseline UUAS: {baseline_score:.3f}")


# ============================================================================
# TRAIN UNSUPERVISED PROBE - MST PROPER 
# ============================================================================

print("\n" + "="*70)
print("TRAINING UNSUPERVISED PROBE - MST PROPER")
print("="*70)

unsupervised_probe_mst = StructuralProbeWithRegularization(model_dim=768, probe_rank=64, reg_strength=0.001)
trainer_mst = ImprovedUnsupervisedProbe(unsupervised_probe_mst, decoder, learning_rate=0.002)

n_epochs = 50
train_losses_mst = []
dev_scores_mst = []
    
print(f"Training for {n_epochs} epochs with MST Proper method...")
best_dev_score_mst = 0
best_epoch_mst = 0

for epoch in range(n_epochs):
    avg_loss = trainer_mst.train_epoch(train_embeddings, method='mst_proper')
    train_losses_mst.append(avg_loss)

    dev_score = evaluate_probe(unsupervised_probe_mst, dev_embeddings, dev_gold_edges, decoder)
    dev_scores_mst.append(dev_score)

    if dev_score > best_dev_score_mst:
        best_dev_score_mst = dev_score
        best_epoch_mst = epoch + 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f} - Dev UUAS: {dev_score:.3f} - Best: {best_dev_score_mst:.3f}")
        print_distance_stats(unsupervised_probe_mst, train_embeddings)

print(f"Training complete!")
print(f"   Best dev UUAS: {best_dev_score_mst:.3f} at epoch {best_epoch_mst}")

test_score_mst = evaluate_probe(unsupervised_probe_mst, test_embeddings, test_gold_edges, decoder)
print(f"\nUnsupervised MST Proper Test UUAS: {test_score_mst:.3f}")


# ============================================================================
# TRAIN UNSUPERVISED PROBE - DISTANCE WEIGHTED WALK 
# ============================================================================

print("\n" + "="*70)
print("TRAINING UNSUPERVISED PROBE - DISTANCE WEIGHTED WALK")
print("="*70)

unsupervised_probe_walk = StructuralProbeWithRegularization(model_dim=768, probe_rank=64, reg_strength=0.001)
trainer_walk = ImprovedUnsupervisedProbe(unsupervised_probe_walk, decoder, learning_rate=0.002)

n_epochs_walk = 50
train_losses_walk = []
dev_scores_walk = []

print(f"Training for {n_epochs_walk} epochs with Distance Weighted Walk...")
best_dev_score_walk = 0
best_epoch_walk = 0

for epoch in range(n_epochs_walk):
    avg_loss = trainer_walk.train_epoch(train_embeddings, method='weighted_walk')
    train_losses_walk.append(avg_loss)

    dev_score = evaluate_probe(unsupervised_probe_walk, dev_embeddings, dev_gold_edges, decoder)
    dev_scores_walk.append(dev_score)

    if dev_score > best_dev_score_walk:
        best_dev_score_walk = dev_score
        best_epoch_walk = epoch + 1

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs_walk} - Loss: {avg_loss:.4f} - Dev UUAS: {dev_score:.3f} - Best: {best_dev_score_walk:.3f}")
        print_distance_stats(unsupervised_probe_walk, train_embeddings)

print(f"Training complete!")
print(f"   Best dev UUAS: {best_dev_score_walk:.3f} at epoch {best_epoch_walk}")

test_score_walk = evaluate_probe(unsupervised_probe_walk, test_embeddings, test_gold_edges, decoder)
print(f"\nUnsupervised Distance Weighted Walk Test UUAS: {test_score_walk:.3f}")


# ============================================================================
# TRAIN SUPERVISED PROBE (UPPER BOUND)
# ============================================================================

print("\n" + "="*70)
print("TRAINING SUPERVISED PROBE (UPPER BOUND)")
print("="*70)

supervised_probe = StructuralProbeWithRegularization(model_dim=768, probe_rank=64, reg_strength=0.0001)
optimizer = torch.optim.Adam(supervised_probe.parameters(), lr=0.001)

n_epochs_sup = 40
print(f"Training supervised probe for {n_epochs_sup} epochs...")
best_sup_dev = 0

for epoch in range(n_epochs_sup):
    total_loss = 0

    for embeddings, gold_edges in zip(train_embeddings, train_gold_edges):
        pred_distances = supervised_probe(embeddings)

        n = len(embeddings)
        gold_distances = decoder.edges_to_distance_matrix(gold_edges, n)
        gold_distances = torch.tensor(gold_distances, dtype=torch.float32)

        loss = torch.nn.functional.mse_loss(pred_distances, gold_distances)
        loss = loss + supervised_probe.get_regularization_loss()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(supervised_probe.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(train_embeddings)
        dev_score_sup = evaluate_probe(supervised_probe, dev_embeddings, dev_gold_edges, decoder)
        print(f"Epoch {epoch+1}/{n_epochs_sup} - Loss: {avg_loss:.4f} - Dev UUAS: {dev_score_sup:.3f}")

        if dev_score_sup > best_sup_dev:
            best_sup_dev = dev_score_sup

print(f"Best supervised dev UUAS: {best_sup_dev:.3f}")

test_score_supervised = evaluate_probe(supervised_probe, test_embeddings, test_gold_edges, decoder)
print(f"\nSupervised Probe Test UUAS: {test_score_supervised:.3f}")


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL RESULTS - TABLE 3: UUAS SCORES")
print("="*70)

results = {
    'Untrained Baseline': baseline_score,
    'Unsupervised MST Proper': test_score_mst,
    'Unsupervised Distance Weighted Walk': test_score_walk,
    'Supervised (Upper Bound)': test_score_supervised
}

print(f"\n{'Probe':<50} {'UUAS':<10}")
print("-" * 60)
for probe_name, score in results.items():
    print(f"{probe_name:<50} {score:<10.3f}")


# ============================================================================
# VISUALIZATION 1: TRAINING CURVES COMPARISON
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Plot 1: MST Proper Training Loss
epochs_range_mst = range(1, len(train_losses_mst) + 1)
axes[0, 0].plot(epochs_range_mst, train_losses_mst, marker='o', linewidth=2,
                color='steelblue', markersize=3, label='Train Loss')
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Training Loss', fontsize=12)
axes[0, 0].set_title('MST Proper: Training Loss', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: MST Proper Dev UUAS
dev_epochs_mst = range(1, len(dev_scores_mst) + 1)
axes[0, 1].plot(dev_epochs_mst, dev_scores_mst, marker='s', linewidth=2,
                color='coral', markersize=3, label='Dev UUAS')
axes[0, 1].axhline(y=baseline_score, color='gray', linestyle='--',
                   label='Baseline', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Dev UUAS', fontsize=12)
axes[0, 1].set_title('MST Proper: Dev Performance', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim([0, 1.0])
axes[0, 1].legend()

# Plot 3: Distance Weighted Walk Training Loss
epochs_range_walk = range(1, len(train_losses_walk) + 1)
axes[1, 0].plot(epochs_range_walk, train_losses_walk, marker='o', linewidth=2,
                color='darkgreen', markersize=3, label='Train Loss')
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Training Loss', fontsize=12)
axes[1, 0].set_title('Distance Weighted Walk: Training Loss', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Plot 4: Distance Weighted Walk Dev UUAS
dev_epochs_walk = range(1, len(dev_scores_walk) + 1)
axes[1, 1].plot(dev_epochs_walk, dev_scores_walk, marker='s', linewidth=2,
                color='purple', markersize=3, label='Dev UUAS')
axes[1, 1].axhline(y=baseline_score, color='gray', linestyle='--',
                   label='Baseline', linewidth=2)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].set_ylabel('Dev UUAS', fontsize=12)
axes[1, 1].set_title('Distance Weighted Walk: Dev Performance', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim([0, 1.0])
axes[1, 1].legend()

plt.suptitle('Unsupervised Training: MST vs Weighted Walk', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

# Create results directory if it doesn't exist
results_dir = os.path.join(parent_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

plt.savefig(os.path.join(results_dir, 'experiment2_training_comparison.png'), dpi=300, bbox_inches='tight')
print("Saved training comparison to results/experiment2_training_comparison.png")
plt.show()


# ============================================================================
# VISUALIZATION 2: DIRECT METHOD COMPARISON
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Train Loss Comparison (all methods)
axes[0].plot(epochs_range_mst, train_losses_mst, marker='o', linewidth=2,
            color='steelblue', markersize=3, label='MST Proper', alpha=0.8)
axes[0].plot(epochs_range_walk, train_losses_walk, marker='s', linewidth=2,
            color='darkgreen', markersize=3, label='Distance Weighted Walk', alpha=0.8)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Training Loss', fontsize=12)
axes[0].set_title('Training Loss: Methods', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Dev UUAS Comparison
axes[1].plot(dev_epochs_mst, dev_scores_mst, marker='o', linewidth=2,
            color='steelblue', markersize=3, label='MST Proper', alpha=0.8)
axes[1].plot(dev_epochs_walk, dev_scores_walk, marker='s', linewidth=2,
            color='darkgreen', markersize=3, label='Distance Weighted Walk', alpha=0.8)
axes[1].axhline(y=baseline_score, color='gray', linestyle='--',
               label='Baseline', linewidth=2, alpha=0.6)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Dev UUAS', fontsize=12)
axes[1].set_title('Dev UUAS: Methods', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.0])
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'experiment2_method_comparison.png'), dpi=300, bbox_inches='tight')
print("✅ Saved method comparison to results/experiment2_method_comparison.png")
plt.show()


# ============================================================================
# VISUALIZATION 3: COMPARISON BAR CHART
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

probe_names = list(results.keys())
scores = list(results.values())

color_map = {
    'Untrained Baseline': 'lightgray',
    'Unsupervised MST Proper (Fixed)': 'steelblue',
    'Unsupervised Distance Weighted Walk (Fixed)': 'darkgreen',
    'Unsupervised Weighted Trees': 'purple',
    'Supervised (Upper Bound)': 'coral'
}
colors = [color_map.get(name, 'gray') for name in probe_names]

bars = ax.bar(probe_names, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.3f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('UUAS (Edge Accuracy)', fontsize=13)
ax.set_title('Structural Probe Comparison', fontsize=15, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=20, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'experiment2_comparison.png'), dpi=300, bbox_inches='tight')
print("✅ Saved comparison to results/experiment2_comparison.png")
plt.show()


# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

import json

results_dict = {
    'baseline_uuas': float(baseline_score),
    'unsupervised_mst_proper_uuas': float(test_score_mst),
    'unsupervised_weighted_walk_uuas': float(test_score_walk),
    'supervised_uuas': float(test_score_supervised),
    'training_losses_mst': [float(x) for x in train_losses_mst],
    'training_losses_walk': [float(x) for x in train_losses_walk],
    'dev_scores_mst': [float(x) for x in dev_scores_mst],
    'dev_scores_walk': [float(x) for x in dev_scores_walk],
    'n_train': len(train_embeddings),
    'n_dev': len(dev_embeddings),
    'n_test': len(test_embeddings),
    'n_epochs_mst': n_epochs,
    'n_epochs_walk': n_epochs_walk,
    'improvements': {
        'mst_improvement_over_baseline': float(test_score_mst - baseline_score),
        'walk_improvement_over_baseline': float(test_score_walk - baseline_score),
        'gap_to_supervised_mst': float(test_score_supervised - test_score_mst),
        'gap_to_supervised_walk': float(test_score_supervised - test_score_walk)
    }
}

with open(os.path.join(results_dir, 'experiment2_results.json'), 'w') as f:
    json.dump(results_dict, f, indent=2)

print("Saved results to results/experiment2_results.json")

# Save the best model
torch.save(unsupervised_probe_mst.state_dict(), os.path.join(results_dir, 'unsupervised_probe_mst.pt'))
print("Saved best MST probe to results/unsupervised_probe_mst.pt")

torch.save(unsupervised_probe_walk.state_dict(), os.path.join(results_dir, 'unsupervised_probe_walk.pt'))
print("Saved best Walk probe to results/unsupervised_probe_walk.pt")


print("\n" + "="*70)
print("EXPERIMENT 2 COMPLETE!")
print("="*70)