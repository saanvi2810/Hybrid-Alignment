import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam
import json
import pickle

from esm2matrix import get_dynamic_cosine_similarity_matrix
from granthammatrix import normalized_grantham_matrix
from local_alignment_affine_weighted import smith_waterman_affine_with_output
from loss_functions import compute_alignment_accuracy

#for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

with open('./sequences.json', 'r') as f:
    sequences = json.load(f)
    
with open('./pair_to_ground_truth.pkl', 'rb') as f:
    pair_to_ground_truth = pickle.load(f)

def load_ground_truth_pairs(file_path='./ground_truth_pairs.csv'):
    df = pd.read_csv(file_path)
    pairs = list(zip(df['Seq1'], df['Seq2']))
    return pairs

ground_truth_pairs = load_ground_truth_pairs('ground_truth_pairs.csv')
random.shuffle(ground_truth_pairs)

split = int(0.7 * len(ground_truth_pairs))
train_pairs = ground_truth_pairs[:split]
val_pairs = ground_truth_pairs[split:]

print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

w_static = torch.tensor(0.5, requires_grad=True)
gap_open = torch.tensor(-10.0, requires_grad=True)
gap_extend = torch.tensor(-1.0, requires_grad=True)

params = [w_static, gap_open, gap_extend]
optimizer = Adam(params, lr=0.01)

def sample_hard_pairs(pairs, batch_size):
    difficulties = []
    for a, b in pairs:
        seq1, seq2 = sequences[a], sequences[b]
        dynamic = get_dynamic_cosine_similarity_matrix(seq1, seq2)
        difficulties.append((a, b, -np.mean(dynamic))) #harder = lower similarity
    difficulties.sort(key=lambda x: x[2], reverse=True)
    sampled = random.sample(difficulties[:len(difficulties)//2], batch_size)
    return [(a, b) for (a, b, _) in sampled]

#training Loop
num_epochs = 300
batch_size = 8

train_losses = []
val_losses = []

best_val_loss = float('inf')
best_params = None

for epoch in range(num_epochs):
    batch_loss = 0.0
    optimizer.zero_grad()

    sampled_pairs = sample_hard_pairs(train_pairs, batch_size)

    for seq1_name, seq2_name in sampled_pairs:
        seq1 = sequences[seq1_name]
        seq2 = sequences[seq2_name]

        dynamic_matrix = get_dynamic_cosine_similarity_matrix(seq1, seq2)

        pred_alignment = smith_waterman_affine_with_output(
            seq1, seq2,
            static_matrix=normalized_grantham_matrix,
            dynamic_matrix=dynamic_matrix,
            w_static=w_static,
            w_dynamic=1 - w_static,
            gap_open=gap_open,
            gap_extend=gap_extend
        )

        gt_alignment = pair_to_ground_truth(seq1_name, seq2_name) 

        if gt_alignment is None:
            continue

        accuracy = compute_alignment_accuracy(pred_alignment, gt_alignment)
        loss = 1 - accuracy
        batch_loss += loss

    batch_loss /= len(sampled_pairs)
    batch_loss.backward()
    optimizer.step()

    #clamp values
    with torch.no_grad():
        w_static.clamp_(0.0, 1.0)
        gap_open.clamp_(-20.0, -0.1)
        gap_extend.clamp_(-10.0, -0.01)

    train_losses.append(batch_loss.item())

    #validation
    sampled_val_pairs = random.sample(val_pairs, batch_size)
    val_loss = 0.0

    for seq1_name, seq2_name in sampled_val_pairs:
        seq1 = sequences[seq1_name]
        seq2 = sequences[seq2_name]

        dynamic_matrix = get_dynamic_cosine_similarity_matrix(seq1, seq2)

        pred_alignment = smith_waterman_affine_with_output(
            seq1, seq2,
            static_matrix=normalized_grantham_matrix,
            dynamic_matrix=dynamic_matrix,
            w_static=w_static,
            w_dynamic=1 - w_static,
            gap_open=gap_open,
            gap_extend=gap_extend
        )

        gt_alignment = pair_to_ground_truth(seq1_name, seq2_name)

        if gt_alignment is None:
            continue

        accuracy = compute_alignment_accuracy(pred_alignment, gt_alignment)
        val_loss += (1 - accuracy)

    val_loss /= len(sampled_val_pairs)
    val_losses.append(val_loss.item())

    #save best
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_params = {
            'w_static': w_static.item(),
            'gap_open': gap_open.item(),
            'gap_extend': gap_extend.item()
        }

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:04d} | Train loss: {batch_loss.item():.4f} | Val loss: {val_loss.item():.4f} | w_static: {w_static.item():.3f} | gap_open: {gap_open.item():.2f} | gap_extend: {gap_extend.item():.2f}")

with open('best_alignment_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)

print(f"\nBest Params Found: {best_params}")

#plot
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig('loss_curve.png')
plt.show()
