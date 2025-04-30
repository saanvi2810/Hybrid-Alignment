import torch
import numpy as np
import os
from transformers import EsmModel, EsmTokenizer

EMBEDDING_FOLDER = "./embeddings" 

def load_embedding_from_disk(seq_name):
    """
    Loads the precomputed embedding tensor for a given sequence.
    The embedding should be saved under: ./embeddings/{seq_name}.pt
    """
    path = os.path.join(EMBEDDING_FOLDER, f"{seq_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    return torch.load(path) 

def get_dynamic_cosine_similarity_matrix(seq_a, seq_b):
    emb_a = load_embedding_from_disk(seq_a)  #(m, d)
    emb_b = load_embedding_from_disk(seq_b)  #(n, d)

    #normalize embeddings
    emb_a_norm = emb_a / emb_a.norm(dim=1, keepdim=True)
    emb_b_norm = emb_b / emb_b.norm(dim=1, keepdim=True)

    #cosine similarity matrix computation (m, n) 
    similarity_matrix = torch.matmul(emb_a_norm, emb_b_norm.T)

    return similarity_matrix.numpy()
