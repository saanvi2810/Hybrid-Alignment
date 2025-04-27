import pandas as pd

def split_ground_truth_by_pair(csv_path):
    """
    Splits the ground truth alignment CSV into a dictionary 
    where each key is a (Seq1, Seq2) pair and each value is a DataFrame 
    containing the corresponding aligned residue pairs.

    Args:
        csv_path (str): Path to the clean_ground_truth.csv file.
        
    Returns:
        dict: { (Seq1, Seq2): DataFrame of corresponding alignments }
    """
    # Load the full ground truth CSV
    df = pd.read_csv(csv_path)

    # Create a new 'pair' column
    df['pair'] = list(zip(df['Seq1'], df['Seq2']))

    # Group by the 'pair' and build the dictionary
    pair_to_df = {pair: group.drop(columns=['pair']).reset_index(drop=True) 
                  for pair, group in df.groupby('pair')}

    return pair_to_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split clean ground truth into per-pair alignments (in memory).")
    parser.add_argument('--csv', required=True, help="Path to clean_ground_truth.csv")
    args = parser.parse_args()

    pair_to_ground_truth = split_ground_truth_by_pair(args.csv)

    print(f"Total number of protein pairs: {len(pair_to_ground_truth)}")

    # Just preview a few pairs
    for pair in list(pair_to_ground_truth.keys())[:5]:  # Show first 5
        print(f"{pair[0]} vs {pair[1]}: {len(pair_to_ground_truth[pair])} aligned residues")
