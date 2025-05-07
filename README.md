# Hybrid Protein Sequence Alignment using ESM2 and Grantham Distances

This project explores dynamic protein alignment by combining ESM2 embeddings with Grantham distance-based similarity. We benchmark our method against traditional approaches such as BLOSUM62.

---

## Overview

Traditional alignment tools like BLOSUM62 use static substitution matrices. In contrast, we introduce a hybrid strategy that dynamically combines:

- **Contextual embeddings** from ESM2 (a transformer-based protein language model)
- **Physicochemical similarity** via normalized Grantham distances

We use a weighted combination of these matrices in the Smith-Waterman local alignment algorithm with affine gap penalties, optimized via KL divergence loss using structural ground truth alignments.

---

## Project Structure

.
├── README.md
├── __pycache__
├── data # Input sequences and ground truth alignments
│   ├── AKT1_precomputed_simlarities
│   │   ├── AGC_AKT1_AGC_AKT2.pt
│   │   ├── AGC_AKT1_AGC_DMPK.pt
│   │   │   ...
│   │   └── AGC_AKT1_TYR_ZAP70.pt
│   ├── BLOSUM_Substitution_Matrix.csv
│   ├── Benchmark-aligned-residue-pairs-SE.out
│   ├── all_sequences.fasta
│   ├── clean_ground_truth.csv
│   ├── ground_truth_pairs.csv
│   ├── name_map.txt
│   ├── pair_to_ground_truth.pkl
│   └── sequences.json
├── embeddings # Precomputed ESM2 residue-level embeddings
│   ├── AAK1_HUMAN.pt
│   ├── AAPK1_HUMAN.pt
│   │     ...
│   └── ZAP70_HUMAN.pt
├── eval_files # Scripts and notebooks for evaluation and visualization
│   ├── AKT1_BLOSUM_graphs.ipynb
│   ├── AKT1_eval.py
│   ├── alignment_method_summary_stats.csv
│   ├── evaluate_blosum_alignments.py
│   └── evaluation.py
├── main_and_dependancies # Core alignment and scoring code
│   ├── esm2matrix.py
│   ├── granthammatrix.py
│   ├── local_alignment_affine_weighted.py
│   └── main.py
├── requirements.txt
├── results # Evaluation results, trained weights, and plots
│   ├── akt1_alignment_by_family.csv
│   ├── best_alignment_params.json
│   ├── blosum + esm2
│   │   ├── blosum_AGC_AKT1_eval_results_100static.csv
│   │   ├── blosum_AGC_AKT1_eval_results_25static.csv
│   │   └── blosum_AGC_AKT1_eval_results_75static.csv
│   ├── esm_AGC_AKT1_eval_results_0static.csv
│   ├── grantham + esm2
│   │   ├── grantham_AGC_AKT1_eval_results_100static.csv
│   │   ├── grantham_AGC_AKT1_eval_results_25static.csv
│   │   ├── grantham_AGC_AKT1_eval_results_50static.csv
│   │   └── grantham_AGC_AKT1_eval_results_75static.csv
│   └── training_summary.png
└── sequence_extractor.ipynb

## Running the Project

Due to file size limitations, some data (e.g., `.pt` embedding files, `.pkl` alignments) are hosted on [Google Drive](https://drive.google.com/drive/folders/1xt80QRTA24enXvQK5tuuTx7ePSAobj19?usp=drive_link).

### 1. Install Dependencies

```bash
pip install -r requirements.txt

### 2. Download Required Files
Manually download the following from Google Drive:

pair_to_ground_truth.pkl

sequences.json

*.pt ESM2 embeddings

clean_ground_truth.csv

Place them in the appropriate folders (data/ and embeddings/) as shown above.

### 3. Train and Evaluate

Run the main script:
python main_and_dependancies/main.py

Train parameters (static weight, gap penalties)

Optimize via KL Divergence loss

Output training/validation curves (results/training_summary.png)

Save best weights in results/best_alignment_params.json

### 4. Evaluation & Visualization
Use the Jupyter notebooks in eval_files/ to:

Compare BLOSUM vs. ESM2 vs. Grantham-based alignments

Visualize accuracy distributions and performance metrics

Compute per-family alignment accuracy (e.g., AKT1 vs. other kinases)

System Requirements

Python 3.10 or 3.11

Compatible with MacOS and Linux

CPU is sufficient, but GPU (CUDA) training significantly. 

