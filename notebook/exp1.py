# Experiment 1: Tree Decoding with MST and ILP
# This notebook replicates the results from the poster

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# SETUP
# ============================================================================

try:
    import pulp
    print("PuLP is installed")
except ImportError:
    print("📦 Installing PuLP...")
    import subprocess
    subprocess.check_call(['pip', 'install', '-q', 'pulp'])
    import pulp
    print("PuLP installed successfully")

# Import our tree decoder
import sys
import os

# Get absolute path to parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from src.tree_decoder import TreeDecoder

# Create results directory
results_dir = os.path.join(parent_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

np.random.seed(42)
print("Setup complete!")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_random_tree(n=5):
    """Generate a random tree with n nodes"""
    if n == 1:
        return []
    G = nx.complete_graph(n)
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.random()
    tree = nx.minimum_spanning_tree(G)
    return list(tree.edges())

def tree_to_distance_matrix(edges, n):
    """Convert tree edges to distance matrix"""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i][j] = nx.shortest_path_length(G, i, j)
    return distances

def mix_distance_matrices(distance_matrices, weights=None):
    """Create mixture of distance matrices"""
    if weights is None:
        weights = [1.0 / len(distance_matrices)] * len(distance_matrices)
    weights = np.array(weights)
    weights = weights / weights.sum()
    mixed = np.zeros_like(distance_matrices[0])
    for dist, w in zip(distance_matrices, weights):
        mixed += w * dist
    return mixed

def compute_weighted_edge_accuracy(predicted_edges, gold_edges_list, weights):
    """Compute weighted edge accuracy"""
    predicted_set = set()
    for u, v in predicted_edges:
        predicted_set.add((min(u, v), max(u, v)))
    
    total_overlap = 0
    total_weight = sum(weights)
    
    for gold_edges, weight in zip(gold_edges_list, weights):
        gold_set = set()
        for u, v in gold_edges:
            gold_set.add((min(u, v), max(u, v)))
        overlap = len(predicted_set & gold_set)
        n_edges = len(gold_set)
        total_overlap += weight * (overlap / n_edges if n_edges > 0 else 0)
    
    return total_overlap / total_weight if total_weight > 0 else 0

def compute_frobenius_norm(pred_distances, gold_distances):
    """Compute Frobenius norm of difference"""
    diff = pred_distances - gold_distances
    return np.linalg.norm(diff, ord='fro')


# ============================================================================
# GENERATE DATA
# ============================================================================

print("\n" + "="*70)
print("GENERATING SYNTHETIC TREES")
print("="*70)

n_nodes = 5
n_trees = 125

all_trees = []
for i in range(n_trees):
    edges = generate_random_tree(n_nodes)
    all_trees.append(edges)

print(f"Generated {len(all_trees)} random trees with {n_nodes} nodes each")


# ============================================================================
# EXPERIMENT: Single Tree
# ============================================================================

print("\n" + "="*70)
print("SCENARIO 1: SINGLE TREE")
print("="*70)

decoder = TreeDecoder()

results_single = {
    'mst': {'edge_acc': [], 'frobenius': []},
    'ilp': {'edge_acc': [], 'frobenius': []}
}

for tree_edges in all_trees[:125]:
    dist_matrix = tree_to_distance_matrix(tree_edges, n_nodes)
    
    # Method 1: MST
    pred_edges_mst = decoder.minimum_spanning_tree(dist_matrix)
    pred_dist_mst = tree_to_distance_matrix(pred_edges_mst, n_nodes)
    edge_acc_mst = compute_weighted_edge_accuracy(pred_edges_mst, [tree_edges], [1.0])
    frob_mst = compute_frobenius_norm(pred_dist_mst, dist_matrix)
    results_single['mst']['edge_acc'].append(edge_acc_mst)
    results_single['mst']['frobenius'].append(frob_mst)
    
    # Method 2: ILP
    pred_edges_ilp = decoder.local_ilp(dist_matrix, timeout=10)
    pred_dist_ilp = tree_to_distance_matrix(pred_edges_ilp, n_nodes)
    edge_acc_ilp = compute_weighted_edge_accuracy(pred_edges_ilp, [tree_edges], [1.0])
    frob_ilp = compute_frobenius_norm(pred_dist_ilp, dist_matrix)
    results_single['ilp']['edge_acc'].append(edge_acc_ilp)
    results_single['ilp']['frobenius'].append(frob_ilp)

avg_edge_acc_mst_single = np.mean(results_single['mst']['edge_acc'])
avg_frob_mst_single = np.mean(results_single['mst']['frobenius'])
avg_edge_acc_ilp_single = np.mean(results_single['ilp']['edge_acc'])
avg_frob_ilp_single = np.mean(results_single['ilp']['frobenius'])

print(f"\nMST Results (n={len(results_single['mst']['edge_acc'])} trees):")
print(f"  Average Edge Accuracy: {avg_edge_acc_mst_single:.3f}")
print(f"  Average Frobenius Norm: {avg_frob_mst_single:.3f}")
print(f"\nILP Results (n={len(results_single['ilp']['edge_acc'])} trees):")
print(f"  Average Edge Accuracy: {avg_edge_acc_ilp_single:.3f}")
print(f"  Average Frobenius Norm: {avg_frob_ilp_single:.3f}")


# ============================================================================
# EXPERIMENT: Two Trees (Mixture)
# ============================================================================

print("\n" + "="*70)
print("SCENARIO 2: MIXTURE OF TWO TREES")
print("="*70)

results_two = {
    'mst': {'edge_acc': [], 'frobenius': []},
    'ilp': {'edge_acc': [], 'frobenius': []}
}

for i in range(62):
    tree1 = all_trees[i*2]
    tree2 = all_trees[i*2 + 1]
    weights = np.random.dirichlet([1, 1])
    
    dist1 = tree_to_distance_matrix(tree1, n_nodes)
    dist2 = tree_to_distance_matrix(tree2, n_nodes)
    mixed_dist = mix_distance_matrices([dist1, dist2], weights)
    
    # MST
    pred_edges_mst = decoder.minimum_spanning_tree(mixed_dist)
    pred_dist_mst = tree_to_distance_matrix(pred_edges_mst, n_nodes)
    edge_acc_mst = compute_weighted_edge_accuracy(pred_edges_mst, [tree1, tree2], weights)
    frob_mst = compute_frobenius_norm(pred_dist_mst, mixed_dist)
    results_two['mst']['edge_acc'].append(edge_acc_mst)
    results_two['mst']['frobenius'].append(frob_mst)
    
    # ILP
    pred_edges_ilp = decoder.local_ilp(mixed_dist, timeout=10)
    pred_dist_ilp = tree_to_distance_matrix(pred_edges_ilp, n_nodes)
    edge_acc_ilp = compute_weighted_edge_accuracy(pred_edges_ilp, [tree1, tree2], weights)
    frob_ilp = compute_frobenius_norm(pred_dist_ilp, mixed_dist)
    results_two['ilp']['edge_acc'].append(edge_acc_ilp)
    results_two['ilp']['frobenius'].append(frob_ilp)

avg_edge_acc_mst_two = np.mean(results_two['mst']['edge_acc'])
avg_frob_mst_two = np.mean(results_two['mst']['frobenius'])
avg_edge_acc_ilp_two = np.mean(results_two['ilp']['edge_acc'])
avg_frob_ilp_two = np.mean(results_two['ilp']['frobenius'])

print(f"\nMST Results (n={len(results_two['mst']['edge_acc'])} mixtures):")
print(f"  Average Edge Accuracy: {avg_edge_acc_mst_two:.3f}")
print(f"  Average Frobenius Norm: {avg_frob_mst_two:.3f}")
print(f"\nILP Results (n={len(results_two['ilp']['edge_acc'])} mixtures):")
print(f"  Average Edge Accuracy: {avg_edge_acc_ilp_two:.3f}")
print(f"  Average Frobenius Norm: {avg_frob_ilp_two:.3f}")


# ============================================================================
# EXPERIMENT: Five Trees (Mixture)
# ============================================================================

print("\n" + "="*70)
print("SCENARIO 3: MIXTURE OF FIVE TREES")
print("="*70)

results_five = {
    'mst': {'edge_acc': [], 'frobenius': []},
    'ilp': {'edge_acc': [], 'frobenius': []}
}

for i in range(25):
    trees = [all_trees[i*5 + j] for j in range(5)]
    weights = np.random.dirichlet([1, 1, 1, 1, 1])
    
    dist_matrices = [tree_to_distance_matrix(t, n_nodes) for t in trees]
    mixed_dist = mix_distance_matrices(dist_matrices, weights)
    
    # MST
    pred_edges_mst = decoder.minimum_spanning_tree(mixed_dist)
    pred_dist_mst = tree_to_distance_matrix(pred_edges_mst, n_nodes)
    edge_acc_mst = compute_weighted_edge_accuracy(pred_edges_mst, trees, weights)
    frob_mst = compute_frobenius_norm(pred_dist_mst, mixed_dist)
    results_five['mst']['edge_acc'].append(edge_acc_mst)
    results_five['mst']['frobenius'].append(frob_mst)
    
    # ILP
    pred_edges_ilp = decoder.local_ilp(mixed_dist, timeout=10)
    pred_dist_ilp = tree_to_distance_matrix(pred_edges_ilp, n_nodes)
    edge_acc_ilp = compute_weighted_edge_accuracy(pred_edges_ilp, trees, weights)
    frob_ilp = compute_frobenius_norm(pred_dist_ilp, mixed_dist)
    results_five['ilp']['edge_acc'].append(edge_acc_ilp)
    results_five['ilp']['frobenius'].append(frob_ilp)

avg_edge_acc_mst_five = np.mean(results_five['mst']['edge_acc'])
avg_frob_mst_five = np.mean(results_five['mst']['frobenius'])
avg_edge_acc_ilp_five = np.mean(results_five['ilp']['edge_acc'])
avg_frob_ilp_five = np.mean(results_five['ilp']['frobenius'])

print(f"\nMST Results (n={len(results_five['mst']['edge_acc'])} mixtures):")
print(f"  Average Edge Accuracy: {avg_edge_acc_mst_five:.3f}")
print(f"  Average Frobenius Norm: {avg_frob_mst_five:.3f}")
print(f"\nILP Results (n={len(results_five['ilp']['edge_acc'])} mixtures):")
print(f"  Average Edge Accuracy: {avg_edge_acc_ilp_five:.3f}")
print(f"  Average Frobenius Norm: {avg_frob_ilp_five:.3f}")


# ============================================================================
# SUMMARY TABLES
# ============================================================================

print("\n" + "="*70)
print("FINAL RESULTS - TABLE 1: WEIGHTED EDGE ACCURACY")
print("="*70)

print(f"\n{'Method':<10} {'Single Tree':<15} {'Two Trees':<15} {'Five Trees':<15}")
print("-" * 60)
print(f"{'MST':<10} {avg_edge_acc_mst_single:<15.3f} {avg_edge_acc_mst_two:<15.3f} {avg_edge_acc_mst_five:<15.3f}")
print(f"{'ILP':<10} {avg_edge_acc_ilp_single:<15.3f} {avg_edge_acc_ilp_two:<15.3f} {avg_edge_acc_ilp_five:<15.3f}")

print("\n" + "="*70)
print("FINAL RESULTS - TABLE 2: AVERAGE FROBENIUS NORM")
print("="*70)

print(f"\n{'Method':<10} {'Single Tree':<15} {'Two Trees':<15} {'Five Trees':<15}")
print("-" * 60)
print(f"{'MST':<10} {avg_frob_mst_single:<15.3f} {avg_frob_mst_two:<15.3f} {avg_frob_mst_five:<15.3f}")
print(f"{'ILP':<10} {avg_frob_ilp_single:<15.3f} {avg_frob_ilp_two:<15.3f} {avg_frob_ilp_five:<15.3f}")


# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scenarios = ['Single\nTree', 'Two\nTrees', 'Five\nTrees']

# Plot 1: Edge Accuracy Comparison
mst_accs = [avg_edge_acc_mst_single, avg_edge_acc_mst_two, avg_edge_acc_mst_five]
ilp_accs = [avg_edge_acc_ilp_single, avg_edge_acc_ilp_two, avg_edge_acc_ilp_five]

x = np.arange(len(scenarios))
width = 0.35

axes[0,0].bar(x - width/2, mst_accs, width, label='MST', color='steelblue', alpha=0.7)
axes[0,0].bar(x + width/2, ilp_accs, width, label='ILP', color='darkgreen', alpha=0.7)
axes[0,0].set_ylabel('Weighted Edge Accuracy', fontsize=12)
axes[0,0].set_title('Edge Accuracy by Scenario', fontsize=14, fontweight='bold')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(scenarios)
axes[0,0].set_ylim([0, 1.0])
axes[0,0].legend()
axes[0,0].grid(axis='y', alpha=0.3)

# Plot 2: Frobenius Norm Comparison
mst_frobs = [avg_frob_mst_single, avg_frob_mst_two, avg_frob_mst_five]
ilp_frobs = [avg_frob_ilp_single, avg_frob_ilp_two, avg_frob_ilp_five]

axes[0,1].bar(x - width/2, mst_frobs, width, label='MST', color='coral', alpha=0.7)
axes[0,1].bar(x + width/2, ilp_frobs, width, label='ILP', color='darkred', alpha=0.7)
axes[0,1].set_ylabel('Average Frobenius Norm', fontsize=12)
axes[0,1].set_title('Distance Distortion by Scenario', fontsize=14, fontweight='bold')
axes[0,1].set_xticks(x)
axes[0,1].set_xticklabels(scenarios)
axes[0,1].legend()
axes[0,1].grid(axis='y', alpha=0.3)

# Plot 3: Example Tree Reconstruction (MST)
example_tree = all_trees[0]
example_dist = tree_to_distance_matrix(example_tree, n_nodes)

G_orig = nx.Graph()
G_orig.add_edges_from(example_tree)
pos = nx.spring_layout(G_orig, seed=42)

axes[1,0].set_title('Original Tree → MST Reconstruction', fontsize=12, fontweight='bold')
pred_edges_mst = decoder.minimum_spanning_tree(example_dist)
G_mst = nx.Graph()
G_mst.add_edges_from(pred_edges_mst)
nx.draw(G_mst, pos, ax=axes[1,0], with_labels=True, node_color='lightblue',
        node_size=500, font_size=10, font_weight='bold')

# Plot 4: Example Tree Reconstruction (ILP)
axes[1,1].set_title('Original Tree → ILP Reconstruction', fontsize=12, fontweight='bold')
pred_edges_ilp = decoder.local_ilp(example_dist, timeout=10)
G_ilp = nx.Graph()
G_ilp.add_edges_from(pred_edges_ilp)
nx.draw(G_ilp, pos, ax=axes[1,1], with_labels=True, node_color='lightgreen',
        node_size=500, font_size=10, font_weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'experiment1.png'), dpi=300, bbox_inches='tight')
print("✅ Saved plot to results/experiment1.png")
plt.show()

print("\n" + "="*70)
print("EXPERIMENT 1 COMPLETE!")
print("="*70)
