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

```
.
├── README.md
├── __pycache__
├── data                      # Input sequences and ground truth alignments
│   ├── AKT1_precomputed_simlarities/
│   ├── BLOSUM_Substitution_Matrix.csv
│   ├── Benchmark-aligned-residue-pairs-SE.out
│   ├── all_sequences.fasta
│   ├── clean_ground_truth.csv
│   ├── ground_truth_pairs.csv
│   ├── name_map.txt
│   ├── pair_to_ground_truth.pkl
│   └── sequences.json
├── embeddings               # Precomputed ESM2 residue-level embeddings (*.pt)
├── eval_files               # Scripts and notebooks for evaluation and visualization
│   ├── AKT1_BLOSUM_graphs.ipynb
│   ├── AKT1_eval.py
│   ├── alignment_method_summary_stats.csv
│   ├── evaluate_blosum_alignments.py
│   └── evaluation.py
├── main_and_dependancies    # Core alignment and scoring code
│   ├── esm2matrix.py
│   ├── granthammatrix.py
│   ├── local_alignment_affine_weighted.py
│   └── main.py
├── requirements.txt
├── results                  # Evaluation results, trained weights, and plots
│   ├── akt1_alignment_by_family.csv
│   ├── best_alignment_params.json
│   ├── blosum + esm2/
│   ├── grantham + esm2/
│   ├── esm_AGC_AKT1_eval_results_0static.csv
│   └── training_summary.png
└── sequence_extractor.ipynb
```

---

## Running the Project

Due to file size limitations, some data (e.g., `.pt` embedding files, `.pkl` alignments) are hosted on **[Google Drive](https://drive.google.com/drive/folders/1xt80QRTA24enXvQK5tuuTx7ePSAobj19?usp=drive_link)**.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Required Files

Manually download the following from Google Drive and place them in the correct folders:

- `pair_to_ground_truth.pkl`
- `sequences.json`
- `*.pt` ESM2 embeddings (into `embeddings/`)
- `clean_ground_truth.csv`

### 3. Train and Evaluate

Run the main script to begin training and evaluation:

```bash
python main_and_dependancies/main.py
```

- Trains on difficult sequence pairs using KL Divergence loss
- Optimizes `w_static`, gap open, and gap extension penalties
- Saves best weights in `results/best_alignment_params.json`
- Outputs training and validation curves in `results/training_summary.png`

### 4. Evaluate & Visualize

Use the Jupyter notebooks in `eval_files/` to:

- Compare ESM2, Grantham, and BLOSUM alignments
- Visualize performance distributions (e.g., box plots)
- Compute per-family alignment accuracy (e.g., for AKT1)

---

## System Requirements

- Python 3.10 or 3.11
- MacOS or Linux recommended
- CPU is sufficient, but GPU (CUDA) will significantly accelerate ESM2-based training

---

## Training Summary

Our training approach uses KL divergence to align predicted soft alignment matrices with structural ground truth. Only a few parameters (similarity weighting and gap penalties) are learned.

Future extensions may include fine-tuning ESM2 layers and incorporating bilinear similarity scoring for more adaptive alignment.

---

## Citation

This project was completed as part of a final computational genomics research effort. Refer to `comp gen final paper.pdf` for a detailed explanation of our methods and results.
