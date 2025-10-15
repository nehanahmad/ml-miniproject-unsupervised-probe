import torch
import spacy
from datasets import load_dataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import treebank

# Import your existing modules
from probe_model import StructuralProbe
from feature_extractor import extract_bert_features # We'll reuse parts of this
from train import compute_distance_matrix # Reuse the distance function

# --- Helper function to get edges from an adjacency matrix ---
def get_edges_from_mst(mst_matrix):
    """Converts a CSR matrix MST into a set of edge tuples."""
    edges = set()
    # tocoo() gives row, col indices of non-zero elements
    rows, cols = mst_matrix.tocoo().row, mst_matrix.tocoo().col
    for i in range(len(rows)):
        # Ensure canonical order (min, max) for undirected comparison
        edge = tuple(sorted((rows[i], cols[i])))
        edges.add(edge)
    return edges

# --- Main Evaluation Function ---
def evaluate_probe(probe, sentences, tokenizer, bert_model):
    """Calculates the average UUAS for a given probe."""
    probe.eval()
    nlp = spacy.load("en_core_web_sm")
    total_uuas = 0
    total_sentences = 0

    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Evaluating"):
            if not sentence or len(sentence.split()) < 2:
                continue

            words = sentence.split()
            
            # 1. Get Gold Edges using spaCy
            doc = nlp(sentence)
            gold_edges = set()
            for token in doc:
                # Exclude self-loops from root tokens
                if token.head.i != token.i:
                    edge = tuple(sorted((token.i, token.head.i)))
                    gold_edges.add(edge)
            
            if not gold_edges: continue

            # 2. Get Predicted Edges from the Probe
            # (This logic is adapted from your feature_extractor.py)
            encoded_input = tokenizer(sentence, return_tensors='pt')
            outputs = bert_model(**encoded_input)
            hidden_states = outputs.hidden_states[-1].squeeze(0)
            
            word_ids = encoded_input.word_ids()
            word_vectors = []
            for i in range(len(words)):
                token_indices = [j for j, word_id in enumerate(word_ids) if word_id == i]
                if token_indices:
                    word_vectors.append(hidden_states[token_indices].mean(dim=0))
            
            if not word_vectors: continue
            sentence_embeddings = torch.stack(word_vectors)
            
            projected = probe(sentence_embeddings)
            d_probe = compute_distance_matrix(projected)
            
            mst_matrix = minimum_spanning_tree(csr_matrix(d_probe.detach().numpy()))
            pred_edges = get_edges_from_mst(mst_matrix)
            
            # 3. Calculate UUAS for the sentence
            correct_edges = len(gold_edges.intersection(pred_edges))
            uuas = correct_edges / len(gold_edges)
            total_uuas += uuas
            total_sentences += 1

    return total_uuas / total_sentences if total_sentences > 0 else 0

# --- Script Entry Point ---
if __name__ == '__main__':
    # -- 1. Setup Models --
    MODEL_DIM = 768
    PROBE_RANK = 64
    
    # Load BERT for feature extraction during eval
    print("Loading BERT and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    bert_model = AutoModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    bert_model.eval()

    # Load your TRAINED probe
    print("Loading Trained Probe...")
    trained_probe = StructuralProbe(MODEL_DIM, PROBE_RANK)
    trained_probe.load_state_dict(torch.load('unsupervised_probe.pt'))

    # Create an UNTRAINED probe for baseline comparison
    print("Initializing Untrained Probe...")
    untrained_probe = StructuralProbe(MODEL_DIM, PROBE_RANK)

    # -- 2. Load Test Data --
    print("Loading test data...")
    # Use the validation or test split for evaluation
    test_sentences = [" ".join(sent) for sent in treebank.sents()][:500]

    # -- 3. Run Evaluations --
    print("\n--- Running Evaluation on Trained Probe ---")
    trained_uuas = evaluate_probe(trained_probe, test_sentences, tokenizer, bert_model)
    print(f"\n✅ Trained Probe Average UUAS: {trained_uuas:.4f}")

    print("\n--- Running Evaluation on Untrained Probe (Baseline) ---")
    untrained_uuas = evaluate_probe(untrained_probe, test_sentences, tokenizer, bert_model)
    print(f"\n✅ Untrained Probe Average UUAS: {untrained_uuas:.4f}")

    # The paper's baseline for the MST probe was 39% 
    print(f"\nComparison: The paper's unsupervised MST probe achieved a UUAS of 0.39. Your results show a change of {trained_uuas - untrained_uuas:+.4f} from the baseline.")
