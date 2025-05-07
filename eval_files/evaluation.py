import pandas as pd

def compute_alignment_accuracy(ground_truth_df, prediction_df):
    """
    Evaluates prediction vs ground truth for a single protein pair.

    Args:
        ground_truth_df (DataFrame): Ground truth alignment for this pair.
        prediction_df (DataFrame): Predicted alignment for this pair.

    Returns:
        true_positive_percentage (float): Percentage of correctly aligned residue pairs.
    """
    gt_pairs = set(zip(ground_truth_df['Pos1'], ground_truth_df['Pos2']))
    pred_pairs = set(zip(prediction_df['Pos1'], prediction_df['Pos2']))

    true_positives = len(gt_pairs & pred_pairs)
    total_ground_truth = len(gt_pairs)

    true_positive_percentage = (true_positives / total_ground_truth) * 100 if total_ground_truth > 0 else 0

    return true_positive_percentage