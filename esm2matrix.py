import torch
import numpy as np
from transformers import EsmForProtein, EsmTokenizer

model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = EsmTokenizer.from_pretrained(model_name)
model = EsmForProtein.from_pretrained(model_name)
model.eval()

def get_embeddings(sequence):
    sequence = ' '.join(list(sequence))
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[0] 

    return embeddings[1:-1]  #remove CLS and EOS tokens

def get_dynamic_cosine_similarity_matrix(seq_a, seq_b):
    emb_a = get_embeddings(seq_a)  #(m, d)
    emb_b = get_embeddings(seq_b)  #(n, d)

    #normalize embeddings
    emb_a_norm = emb_a / emb_a.norm(dim=1, keepdim=True)
    emb_b_norm = emb_b / emb_b.norm(dim=1, keepdim=True)

    #cosine similarity matrix computation (m, n) 
    similarity_matrix = torch.matmul(emb_a_norm, emb_b_norm.T)

    return similarity_matrix.numpy()
