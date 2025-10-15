#
# train.py
#
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from tqdm import tqdm
import numpy as np

from probe_model import StructuralProbe

# -- Helper Functions --
def compute_distance_matrix(projected_embeddings):
    """Computes squared L2 distance matrix."""
    n_words = projected_embeddings.shape[0]
    dist_matrix = torch.zeros((n_words, n_words))
    for i in range(n_words):
        for j in range(i + 1, n_words):
            dist = torch.sum((projected_embeddings[i] - projected_embeddings[j])**2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix

def decode_mst(distance_matrix):
    """Decodes the MST and computes its path distance matrix."""
    # Scipy's MST function works on a CSR matrix
    graph = csr_matrix(distance_matrix.detach().numpy())
    mst = minimum_spanning_tree(graph)
    
    # Compute path distances on the MST
    mst_path_distances = shortest_path(csgraph=mst, directed=False, unweighted=True)
    # The result can contain inf for unconnected nodes, handle it.
    mst_path_distances[mst_path_distances == np.inf] = 0
    
    return torch.from_numpy(mst_path_distances)

# -- Main Training Script --
def main():
    # 1. Hyperparameters
    MODEL_DIM = 768 # For bert-base
    PROBE_RANK = 64 # As mentioned in the paper 
    LEARNING_RATE = 0.001 # As mentioned in the paper 
    EPOCHS = 10

    # 2. Load Data and Initialize Model
    embeddings = torch.load('ptb_bert_embeddings.pt')
    probe = StructuralProbe(MODEL_DIM, PROBE_RANK)
    optimizer = optim.Adam(probe.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.L1Loss() # L1 Loss (Mean Absolute Error)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for sentence_embeddings in tqdm(embeddings, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            if sentence_embeddings.shape[0] < 2: continue

            optimizer.zero_grad()
            
            # Project embeddings
            projected = probe(sentence_embeddings)
            
            # Calculate probe's distance matrix
            d_probe = compute_distance_matrix(projected)
            
            # E-Step: Decode MST to get pseudo-labels
            d_mst = decode_mst(d_probe)
            
            # M-Step: Compute loss and update
            loss = loss_function(d_probe, d_mst)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(embeddings):.4f}")

    # Save the trained probe
    torch.save(probe.state_dict(), 'unsupervised_probe.pt')
    print("Training complete and model saved.")

if __name__ == '__main__':
    main()