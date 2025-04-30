import pandas as pd
import json
import pickle
import torch
from tqdm import tqdm
import os

from local_alignment_affine_weighted import smith_waterman_affine_with_output
from evaluation import compute_alignment_accuracy

# Force CPU
device = torch.device("cpu")
torch.set_default_device(device)
print("Running on CPU.")

# Load data
with open("sequences.json") as f:
    sequences = json.load(f)

with open("pair_to_ground_truth.pkl", "rb") as f:
    pair_to_gt = pickle.load(f)

name_map = {}
with open("name_map.txt") as f:
    for line in f:
        if '->' in line:
            a, b = line.strip().split("->")
            name_map[a.strip()] = b.strip()

blosum_df = pd.read_csv("BLOSUM_Substitution_Matrix.csv", index_col=0)
all_pairs = list(pair_to_gt.keys())

# Start fresh
partial_file = "blosum_all_partial.csv"
results = []

# Main loop
for i, (s1, s2) in enumerate(tqdm(all_pairs, desc="Evaluating all pairs")):
    m1, m2 = name_map.get(s1), name_map.get(s2)
    if not m1 or not m2:
        continue

    m1 += "_HUMAN"
    m2 += "_HUMAN"

    if m1 not in sequences or m2 not in sequences:
        continue

    seq1 = sequences[m1]
    seq2 = sequences[m2]
    gt = pair_to_gt.get((s1, s2))

    if gt is None or gt.empty:
        continue

    dummy_dynamic = torch.zeros((len(seq1), len(seq2)), device=device)

    try:
        pred, _ = smith_waterman_affine_with_output(
            seq1, seq2, m1, m2,
            static_matrix=blosum_df,
            dynamic_matrix=dummy_dynamic,
            w_static=1.0,
            w_dynamic=0.0,
            gap_open=-10.0,
            gap_extend=-1.0
        )
        acc = compute_alignment_accuracy(gt, pred)
    except Exception as e:
        acc = None
        print(f"Error on {s1}, {s2}: {e}")

    results.append({"Seq1": s1, "Seq2": s2, "Accuracy": acc})

    if (i + 1) % 250 == 0:
        pd.DataFrame(results).to_csv(partial_file, index=False)
        print(f"Saved {len(results)} results to {partial_file}")

# Final save
pd.DataFrame(results).to_csv("blosum_all_eval_final.csv", index=False)
print("Done — final results saved to blosum_all_eval_final.csv")
