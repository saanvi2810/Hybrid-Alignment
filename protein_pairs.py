import pandas as pd

# Load the ground truth alignments
ground_truth = pd.read_csv('clean_ground_truth.csv')

# Create a DataFrame of unique (Seq1, Seq2) pairs
protein_pairs = ground_truth[['Seq1', 'Seq2']].drop_duplicates().reset_index(drop=True)

# Print to check
print(protein_pairs)

# Save to CSV
protein_pairs.to_csv('ground_truth_pairs.csv', index=False)
