"""
Utility functions for loading data and models
"""
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import nltk
from nltk.corpus import treebank

def load_bert_model():
    """Load pretrained BERT model and tokenizer"""
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set to evaluation mode
    print("✅ BERT model loaded successfully")
    return model, tokenizer

def get_embeddings(model, tokenizer, sentence, layer=1):
    """
    Get BERT embeddings for a sentence
    
    Args:
        model: BERT model
        tokenizer: BERT tokenizer
        sentence: string or list of words
        layer: which layer to extract (0-12 for BERT base)
    
    Returns:
        embeddings: numpy array of shape [seq_len, hidden_dim]
    """
    if isinstance(sentence, list):
        sentence = " ".join(sentence)
    
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", padding=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract specific layer (layer 1 like in the paper)
    hidden_states = outputs.hidden_states[layer]
    
    # Remove batch dimension and convert to numpy
    embeddings = hidden_states.squeeze(0).numpy()
    
    return embeddings

def load_treebank_data():
    """Load Penn Treebank data"""
    try:
        sentences = treebank.parsed_sents()
        print(f"✅ Loaded {len(sentences)} sentences from Penn Treebank")
        return sentences
    except LookupError:
        print("Downloading Penn Treebank...")
        nltk.download('treebank')
        sentences = treebank.parsed_sents()
        return sentences

def tree_to_distance_matrix(tree):
    """
    Convert a parse tree to a distance matrix
    
    Args:
        tree: NLTK Tree object
    
    Returns:
        distance_matrix: numpy array [n x n] where n is number of words
    """
    # Get words from tree
    words = tree.leaves()
    n = len(words)
    
    # Initialize distance matrix
    distances = np.zeros((n, n))
    
    # Compute distances between all pairs
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i][j] = 0
            else:
                # Find path length in tree
                distances[i][j] = compute_tree_distance(tree, i, j)
    
    return distances

def compute_tree_distance(tree, idx1, idx2):
    """
    Compute distance between two word indices in a tree
    This is a simplified version - you may need to implement proper tree path finding
    """
    # For now, return simple heuristic
    return abs(idx1 - idx2)

def compute_uuas(predicted_edges, gold_edges):
    """
    Compute Undirected Unlabeled Attachment Score
    
    Args:
        predicted_edges: set of tuples (i, j)
        gold_edges: set of tuples (i, j)
    
    Returns:
        score: float between 0 and 1
    """
    if len(gold_edges) == 0:
        return 0.0
    
    # Make edges undirected
    pred_undirected = set()
    for u, v in predicted_edges:
        pred_undirected.add((min(u, v), max(u, v)))
    
    gold_undirected = set()
    for u, v in gold_edges:
        gold_undirected.add((min(u, v), max(u, v)))
    
    # Compute overlap
    correct = len(pred_undirected & gold_undirected)
    total = len(gold_undirected)
    
    return correct / total