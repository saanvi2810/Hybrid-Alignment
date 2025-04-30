import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.optim import Adam
import json
import pickle
import torch.nn.functional as F

from esm2matrix import get_dynamic_cosine_similarity_matrix
from granthammatrix import normalized_grantham_matrix
from local_alignment_affine_weighted import smith_waterman_affine_with_output
from evaluation import compute_alignment_accuracy

#for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def alignment_df_to_matrix(df, len1, len2):
    matrix = torch.zeros((len1, len2), dtype=torch.float32)
    for _, row in df.iterrows():
        i = int(row['Pos1'])
        j = int(row['Pos2'])
        if 0 <= i < len1 and 0 <= j < len2:
            matrix[i-1, j-1] = 1.0
    return matrix

def kl_alignment_loss(pred_scores, gt_matrix):
    pred_log_probs = F.log_softmax(pred_scores, dim=1) 
    gt_probs = F.softmax(gt_matrix, dim=1)
    return F.kl_div(pred_log_probs, gt_probs, reduction='batchmean')

print("Loading data...")
with open('./sequences.json', 'r') as f:
    sequences = json.load(f)
    
with open('./pair_to_ground_truth.pkl', 'rb') as f:
    pair_to_ground_truth = pickle.load(f)
    
name_map = {}
with open('./name_map.txt', 'r') as f:
    for line in f:
        if '->' in line:
            original, mapped = line.strip().split('->')
            original = original.strip()
            mapped = mapped.strip()
            name_map[original] = mapped

print("Data loaded.")

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
        
        mapped_seq1 = name_map.get(a, None)
        mapped_seq2 = name_map.get(b, None)

        if mapped_seq1 is None or mapped_seq2 is None:
            continue

        mapped_seq1 += '_HUMAN'
        mapped_seq2 += '_HUMAN'

        if mapped_seq1 not in sequences or mapped_seq2 not in sequences:
            continue

        seq1 = sequences[mapped_seq1]
        seq2 = sequences[mapped_seq2]
        dynamic = get_dynamic_cosine_similarity_matrix(mapped_seq1, mapped_seq2)
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
    optimizer.zero_grad()
    losses = []
    sampled_pairs = sample_hard_pairs(train_pairs, batch_size)

    for seq1_name, seq2_name in sampled_pairs:
        mapped_seq1 = name_map.get(seq1_name, None)
        mapped_seq2 = name_map.get(seq2_name, None)

        if mapped_seq1 is None or mapped_seq2 is None:
            continue  # skip if mapping missing

        mapped_seq1 += '_HUMAN'
        mapped_seq2 += '_HUMAN'

        if mapped_seq1 not in sequences or mapped_seq2 not in sequences:
            continue  # skip if sequence missing

        seq1 = sequences[mapped_seq1]
        seq2 = sequences[mapped_seq2]

        dynamic_matrix = get_dynamic_cosine_similarity_matrix(mapped_seq1, mapped_seq2)

        pred_alignment, pred_scores = smith_waterman_affine_with_output(
            seq1, seq2, mapped_seq1, mapped_seq2,
            static_matrix=normalized_grantham_matrix,
            dynamic_matrix=dynamic_matrix,
            w_static=w_static,
            w_dynamic=1 - w_static,
            gap_open=gap_open,
            gap_extend=gap_extend
        )

        gt_alignment = pair_to_ground_truth.get((seq1_name, seq2_name))


        if gt_alignment is None:
            continue

        len1, len2 = pred_scores.shape
        gt_matrix = alignment_df_to_matrix(gt_alignment, len1, len2)
        loss = kl_alignment_loss(pred_scores, gt_matrix)
        losses.append(loss)

    batch_loss = torch.stack(losses).mean()
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
    val_losses_epoch = []

    for seq1_name, seq2_name in sampled_val_pairs:
        mapped_seq1 = name_map.get(seq1_name, None)
        mapped_seq2 = name_map.get(seq2_name, None)

        if mapped_seq1 is None or mapped_seq2 is None:
            continue

        mapped_seq1 += '_HUMAN'
        mapped_seq2 += '_HUMAN'

        if mapped_seq1 not in sequences or mapped_seq2 not in sequences:
            continue

        seq1 = sequences[mapped_seq1]
        seq2 = sequences[mapped_seq2]

        dynamic_matrix = get_dynamic_cosine_similarity_matrix(mapped_seq1, mapped_seq2)

        pred_alignment, pred_scores = smith_waterman_affine_with_output(
            seq1, seq2, mapped_seq1, mapped_seq2,
            static_matrix=normalized_grantham_matrix,
            dynamic_matrix=dynamic_matrix,
            w_static=w_static,
            w_dynamic=1 - w_static,
            gap_open=gap_open,
            gap_extend=gap_extend
        )

        gt_alignment = pair_to_ground_truth.get((seq1_name, seq2_name))

        if gt_alignment is None:
            continue

        accuracy = compute_alignment_accuracy(pred_alignment, gt_alignment)
        val_losses_epoch.append(1 - accuracy)

    val_loss = np.mean(val_losses_epoch) if val_losses_epoch else float('inf')
    val_losses.append(val_loss)

    #save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {
            'w_static': w_static.item(),
            'gap_open': gap_open.item(),
            'gap_extend': gap_extend.item()
        }

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:04d} | Train loss: {batch_loss.item():.4f} | Val loss: {val_loss:.4f} | w_static: {w_static.item():.3f} | gap_open: {gap_open.item():.2f} | gap_extend: {gap_extend.item():.2f}")

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
