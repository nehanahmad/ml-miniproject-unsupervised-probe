#
# feature_extractor.py
#
import torch
from datasets import load_dataset
from transformers import BertModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
from nltk.corpus import treebank

def extract_bert_features():
    # 1. Load Model and Data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval() # Disable dropout

    # Load Penn Treebank sentences via NLTK (avoids local dataset script conflict)
    sentences = [" ".join(sent) for sent in treebank.sents()]

    all_word_embeddings = []

    # 2. Process each sentence
    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Extracting Features"):
            if not sentence: continue

            words = sentence.split()
            encoded_input = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**encoded_input)
            hidden_states = outputs.hidden_states[-1].squeeze(0) # Get last hidden layer

            # 3. Align subwords to words
            word_ids = encoded_input.word_ids()
            word_vectors = []
            
            for i in range(len(words)):
                # Find all subword token indices for the current word
                token_indices = [j for j, word_id in enumerate(word_ids) if word_id == i]
                if token_indices:
                    # Average the embeddings of the subword tokens
                    word_vector = hidden_states[token_indices].mean(dim=0)
                    word_vectors.append(word_vector)
            
            if word_vectors:
                all_word_embeddings.append(torch.stack(word_vectors))

    # 4. Save features to disk
    torch.save(all_word_embeddings, 'ptb_bert_embeddings.pt')
    print("Features extracted and saved.")

if __name__ == '__main__':
    extract_bert_features()