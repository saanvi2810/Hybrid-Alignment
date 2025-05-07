import pandas as pd
import torch
import os

def smith_waterman_affine_with_output(seq1, seq2, 
                                      seq1_name, seq2_name,
                                      static_matrix, dynamic_matrix, 
                                      w_static=0.5, w_dynamic=0.5, 
                                      gap_open=-10.0, gap_extend=-1.0, device='cpu'):
    
    m, n = len(seq1), len(seq2)

    score = torch.zeros((m + 1, n + 1), dtype=torch.float32)
    gap_a = torch.full((m + 1, n + 1), float('-inf'))
    gap_b = torch.full((m + 1, n + 1), float('-inf'))
    pointer = torch.zeros((m + 1, n + 1), dtype=torch.int32)

    max_score = 0.0
    max_pos = None

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            aa1, aa2 = seq1[i - 1], seq2[j - 1]

            static_score = static_matrix.at[aa1, aa2] if (aa1 in static_matrix.index and aa2 in static_matrix.columns) else 0.0
            dynamic_score = dynamic_matrix[i - 1, j - 1]

            match_score = w_static * static_score + w_dynamic * dynamic_score

            diag = score[i - 1, j - 1] + match_score
            gap_a[i, j] = max(score[i - 1, j] + gap_open, gap_a[i - 1, j] + gap_extend)
            gap_b[i, j] = max(score[i, j - 1] + gap_open, gap_b[i, j - 1] + gap_extend)

            best = max(torch.tensor(0.0), diag, gap_a[i, j], gap_b[i, j])
            score[i, j] = best

            if best == 0:
                pointer[i, j] = 0
            elif best == diag:
                pointer[i, j] = 1
            elif best == gap_a[i, j]:
                pointer[i, j] = 2
            else:
                pointer[i, j] = 3

            if best > max_score:
                max_score = best
                max_pos = (i, j)

    aligned_pairs = []
    if max_pos is not None:
        i, j = max_pos
        while pointer[i, j] != 0:
            if pointer[i, j] == 1:
                aligned_pairs.append({
                    "Seq1": seq1_name,
                    "Pos1": i,
                    "Res1": seq1[i - 1],
                    "Seq2": seq2_name,
                    "Pos2": j,
                    "Res2": seq2[j - 1],
                    "Label": 1
                })
                i -= 1
                j -= 1
            elif pointer[i, j] == 2:
                i -= 1
            elif pointer[i, j] == 3:
                j -= 1
        aligned_pairs = aligned_pairs[::-1]

    df_output = pd.DataFrame(aligned_pairs)
    
    return df_output, score[1:, 1:] 