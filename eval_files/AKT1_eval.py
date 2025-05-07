import os
import torch
import json
import pickle
import pandas as pd
from tqdm import tqdm

from evaluation import compute_alignment_accuracy
from local_alignment_affine_weighted import smith_waterman_affine_with_output
from granthammatrix import normalized_grantham_matrix

use_blosum = True  #Switch to False to use Grantham

if use_blosum:
    blosum_path = "./BLOSUM_Substitution_Matrix.csv"
    blosum_df = pd.read_csv(blosum_path, index_col=0)
    blosum_matrix = blosum_df.astype(float)
    normalized_static_matrix = blosum_matrix / blosum_matrix.to_numpy().max()
    print("Using BLOSUM substitution matrix.")
else:
    from granthammatrix import normalized_grantham_matrix
    normalized_static_matrix = normalized_grantham_matrix
    print("Using Grantham substitution matrix.")
# Set to CPU only
device = torch.device("cpu")

# Constants
SEQ_NAME = 'AGC_AKT1'
SEQ_TAGGED = SEQ_NAME.replace('AGC_', '') + '_HUMAN'

SIM_DIR = '/Users/saanviaima/Desktop/AKT1' 
EMBEDDING_DIR = './embeddings'

print("Loading sequences...")
with open('./sequences.json') as f:
    sequences = json.load(f)
print(f"Total sequences loaded: {len(sequences)}")

print("Loading ground truth...")
with open('./pair_to_ground_truth.pkl', 'rb') as f:
    pair_to_ground_truth = pickle.load(f)
print(f"Total ground truth pairs loaded: {len(pair_to_ground_truth)}")

results = []
skipped = 0

def strip_prefix(name):
    """Strip everything before the first underscore (e.g., 'AGC_AKT1' -> 'AKT1')"""
    return name.split('_', 1)[-1]

for fname in tqdm(os.listdir(SIM_DIR), desc="Evaluating AGC_AKT1"):
    if not fname.startswith(f"{SEQ_NAME}_") or not fname.endswith('.pt'):
        continue

    other_full = fname[len(SEQ_NAME) + 1:-3]

    seq1_tagged = strip_prefix(SEQ_NAME) + '_HUMAN'
    seq2_tagged = strip_prefix(other_full) + '_HUMAN'

    sim_path = os.path.join(SIM_DIR, fname)

    if seq1_tagged not in sequences or seq2_tagged not in sequences:
        print(f"Missing sequence for: {seq1_tagged} or {seq2_tagged}")
        continue

    seq1 = sequences[seq1_tagged]
    seq2 = sequences[seq2_tagged]

    sim_matrix = torch.load(sim_path, map_location=device)

    pred_df, _ = smith_waterman_affine_with_output(
        seq1, seq2, seq1_tagged, seq2_tagged,
        static_matrix=normalized_static_matrix,
        dynamic_matrix=sim_matrix,
        w_static=0.75,
        w_dynamic=0.25,
        gap_open=-11.0,
        gap_extend=-1.0,
        device=device
    )

    gt = pair_to_ground_truth.get((SEQ_NAME, other_full))
    if gt is None:
        print(f"No ground truth for: ({SEQ_NAME}, {other_full})")
        continue

    acc = compute_alignment_accuracy(gt, pred_df)
    results.append((other_full, acc))


print(f"\nSaving results... {len(results)} successful, {skipped} skipped.")
results_df = pd.DataFrame(results, columns=['Seq2', 'Accuracy'])
results_df.sort_values(by='Accuracy', ascending=False, inplace=True)
results_df.to_csv('AGC_AKT1_eval_results.csv', index=False)

print("\nTop 5 results:")
print(results_df.head())
