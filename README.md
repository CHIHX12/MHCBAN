# MHCBAN — MHC-Peptide Binding Affinity Prediction

> **Language:** English (primary) · [中文輔助說明](README_zh.md)

MHCBAN is a deep learning framework for binary MHC class I–peptide binding affinity prediction.
It adapts the **Bilinear Attention Network (BAN)** architecture from
[DrugBAN](https://github.com/pz-white/DrugBAN) (Bai *et al.*, *Nature Machine Intelligence* 2023),
replacing the molecular-graph and protein-CNN encoders with **bidirectional LSTM (BiLSTM)**
encoders suited to amino acid sequence inputs.
The model supports HLA-A, HLA-B, and HLA-C alleles and is evaluated under four training strategies,
including a **Pan-HLA universal model** that covers all three HLA types with a single set of weights.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Data Preparation](#data-preparation)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Gray Zone Inference](#gray-zone-inference)
10. [Discussion](#discussion)
11. [Citation](#citation)
12. [Environment](#environment)

---

## Architecture

```
Peptide sequence (≤30 aa) ──► PeptideBiLSTM ──► Feature matrix  (batch × 30 × 256)
                                                         │
                                                    BAN Layer          ← identical to DrugBAN
                                                         │
MHC pseudo-sequence (34 aa) ──► MHCBiLSTM ──► Feature matrix  (batch × 34 × 256)
                                                         │
                                                   MLPDecoder          ← identical to DrugBAN
                                                         │
                                              Binding probability  [0, 1]
```

See `figures/fig0_model_architecture.pdf` for the full block diagram with per-layer dimensions
and parameter counts.

### Correspondence with DrugBAN

| Module | DrugBAN | MHCBAN |
|--------|---------|--------|
| Branch 1 encoder | MolecularGCN (molecular graph) | PeptideBiLSTM (amino acid sequence) |
| Branch 2 encoder | ProteinCNN (protein sequence) | MHCBiLSTM (34-position pseudo-sequence) |
| Attention | BANLayer ✓ identical | BANLayer ✓ identical |
| Decoder | MLPDecoder ✓ identical | MLPDecoder ✓ identical |

### Parameter Summary

| Module | Structure | Parameters | Share |
|--------|-----------|------------|-------|
| PeptideBiLSTM | Embedding + BiLSTM ×2 | 662,656 | 30.3% |
| MHCBiLSTM | Embedding + BiLSTM ×2 | 662,656 | 30.3% |
| BANLayer | FC ×2 + h_mat + BN | 397,574 | 18.2% |
| MLPDecoder | FC ×4 + BN ×3 | 462,337 | 21.2% |
| **Total** | **15 parametric layers** | **2,185,223** | 100% |

**Key hyperparameters:** vocab = 25, embedding dim = 128, BiLSTM hidden = 128 (bidirectional → 256),
BiLSTM layers = 2, BAN heads = 3, MLP: 256 → 512 → 512 → 128 → 1.

---

## Dataset

### Source and Labeling

- **Source:** IEDB (Immune Epitope Database)
- **Positive label (binder):** IC50 < 100 nM
- **Negative label (non-binder):** IC50 > 10,000 nM
- **MHC representation:** 34-position pseudo-sequence (NetMHCpan convention)

### Data Statistics (clean dataset)

| HLA Type | Raw rows | PTM removed | Clean rows | Unique alleles | Peptide length |
|----------|---------|------------|------------|---------------|----------------|
| HLA-A | 50,899 | 178 | **50,721** | 42 | 8–30 aa |
| HLA-B | 28,930 | 15 | **28,915** | 48 | 7–26 aa |
| HLA-C | 1,033 | 1 | **1,032** | 11 | 8–21 aa |
| **Total** | **80,862** | **194** | **80,668** | **101** | |

### Data Quality: PTM Removal

The raw dataset contains **194 peptides** with post-translational modification (PTM) annotations
(e.g., `VLHDDLCEA+SCM(C7)`, `KIEDFGSAK(ac)`).
These entries are problematic because their IC50 labels correspond to the *modified* peptides,
not the bare amino acid sequences encoded by the model.
Special characters (`+`, `(`, digits) would be mapped to `X` (unknown residue),
creating a sequence–label mismatch.

**Decision:** All 194 PTM-annotated entries were **removed** rather than converted,
because the IC50 labels themselves are not transferable to the unmodified sequences.

**Validation:** Removal has zero impact on random-split AUROC (Δ = 0.000) and a small
positive impact on cluster-split AUROC (HLA-B cluster: Δ = +0.033), confirming that PTM
labels introduced noise into allele-specific training signals.

### Gray Zone (not used in training)

Peptides in the IC50 range 100–10,000 nM were withheld from training and used exclusively
for post-hoc gray zone inference (see [Gray Zone Inference](#gray-zone-inference)).

| HLA Type | Gray-zone peptides |
|----------|--------------------|
| HLA-A | 32,638 |
| HLA-B | 11,565 |
| HLA-C | 912 |
| **Total** | **45,115** |

---

## Repository Structure

```
MHCBAN/
├── configs/
│   ├── mhcban_config.yaml               # Reference config (original, do not use directly)
│   └── mhcban_config_clean.yaml         # ✅ Official config (OUTPUT_DIR: ./results_clean)
├── datasets/                            # ⚠️ Original splits (includes PTM — archived)
├── datasets_clean/                      # ✅ Official splits (PTM removed)
│   ├── HLA_A/ HLA_B/ HLA_C/
│   │   ├── random/   train.csv / val.csv / test.csv
│   │   └── cluster/  train.csv / val.csv / test.csv / allele_assignment.txt
│   └── cross_hla/
│       ├── A_to_B/   source_train.csv / target_train.csv / target_test.csv
│       └── A_to_C/
├── models/
│   ├── ban.py                           # BANLayer + FCNet (ported from DrugBAN, unchanged)
│   ├── encoders.py                      # PeptideBiLSTM + MHCBiLSTM
│   └── mhcban.py                        # MHCBAN model + MLPDecoder
├── domain_adaptator.py                  # Gradient Reversal Layer + Discriminator (CDAN)
├── trainer.py                           # Trainer (standard) + DATrainer (CDAN)
├── train.py                             # Plan 1 entry point
├── train_da.py                          # Plan 2 entry point (cross-HLA DA)
├── train_pretrain.py                    # Plan 3/4 Phase 1 entry point
├── train_finetune.py                    # Plan 3 Phase 2 entry point
├── run_stability_clean.sh               # ✅ Plan 1 & 2 stability test (clean data)
├── run_stability_plan3_plan4_clean.sh   # ✅ Plan 3 & 4 stability test (clean data)
├── aggregate_stability.py               # Aggregate results (--clean / --both)
├── infer_gray_zone.py                   # Gray zone inference script
├── plot_results.py                      # Generate publication figures (fig1–fig5)
├── results_table.txt                    # ✅ Full results + reproducibility + architecture record
├── gray_zone_inference.csv             # Input: 45,115 gray-zone peptides
├── gray_zone_results.csv               # Output: + predicted binding probability columns
├── figures/                            # ✅ Publication figures (fig0–8, 600 DPI, PDF+PNG+SVG)
│   ├── fig0_model_architecture.*       # Architecture block diagram
│   ├── fig1_auroc_overview.*           # AUROC overview (all 13 experiments)
│   ├── fig2_metrics_heatmap.*          # Full metrics heatmap
│   ├── fig3_random_vs_cluster.*        # Random vs. Cluster split comparison
│   ├── fig4_cold_start_comparison.*    # Cold-start scenario comparison
│   ├── fig5_pan_vs_individual.*        # Pan-HLA vs. individual models
│   ├── fig6_gray_scatter.*             # Gray zone: log(IC50) vs. predicted probability
│   ├── fig7_gray_boxplot.*             # Gray zone: per-bin box plots
│   ├── fig8_gray_histogram.*           # Gray zone: per-HLA histograms
│   └── figure_captions.txt            # Full English captions for all figures
└── results_clean/                      # ✅ Official results (clean data, N=10 seeds)
    ├── <exp>_seed{42..51}/             # 13 experiments × 10 seeds = 130 directories
    │   ├── result_metrics.pt           # Test metrics (auroc / auprc / f1 / sens / spec / acc)
    │   ├── best_model_epoch_*.pth      # Best checkpoint (by validation AUROC) [not tracked]
    │   └── {train,valid,test}_markdowntable.txt
    ├── pretrain_HLAA_random_seed{42..51}   # Phase 1 checkpoint for Plan 3 FT HLA-B
    └── pretrain_HLAAB_random_seed{42..51}  # Phase 1 checkpoint for Plan 3 FT HLA-C
```

> **Note:** Model weight files (`*.pth`) are excluded from version control due to size (~2.5 GB total).
> All `result_metrics.pt` files (test-set metrics) are tracked and sufficient for reproducing the tables.

---

## Requirements

```bash
# Recommended: create a dedicated conda environment
conda create -n mhcban python=3.10
conda activate mhcban
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn prettytable tqdm pandas numpy PyYAML

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

---

## Data Preparation

### Prerequisites

The raw source CSV files should be placed as follows:

```
TCR_A_B_C/
├── HLA_A/
│   └── binary_train_HLA_A_clean.csv   # Clean version (PTM removed)
├── HLA_B/
│   └── binary_train_HLA_B_clean.csv
├── HLA_C/
│   └── binary_train_HLA_C_clean.csv
└── MHCBAN/                            # This repository root
```

CSV columns:
- `Peptide` — amino acid sequence (8–30 aa)
- `Pseudo_Sequence` — 34-position MHC fingerprint (NetMHCpan)
- `Label` — 1 = binder (IC50 < 100 nM), 0 = non-binder (IC50 > 10,000 nM)

### Generate dataset splits (run once)

```bash
cd MHCBAN

# Generate datasets_clean/ from the clean source CSVs
python datasets/prepare_splits.py \
    --data .. \
    --out_root datasets_clean \
    --suffix _clean

# Build cross-HLA DA splits (copy from random splits)
mkdir -p datasets_clean/cross_hla/A_to_B datasets_clean/cross_hla/A_to_C
cp datasets_clean/HLA_A/random/train.csv datasets_clean/cross_hla/A_to_B/source_train.csv
cp datasets_clean/HLA_B/random/train.csv datasets_clean/cross_hla/A_to_B/target_train.csv
cp datasets_clean/HLA_B/random/test.csv  datasets_clean/cross_hla/A_to_B/target_test.csv
cp datasets_clean/HLA_A/random/train.csv datasets_clean/cross_hla/A_to_C/source_train.csv
cp datasets_clean/HLA_C/random/train.csv datasets_clean/cross_hla/A_to_C/target_train.csv
cp datasets_clean/HLA_C/random/test.csv  datasets_clean/cross_hla/A_to_C/target_test.csv
```

---

## Training

### Training Plans Overview

| Plan | Method | Entry point |
|------|--------|-------------|
| **Plan 1** | Standard training (Random / Cluster split) | `train.py` |
| **Plan 2** | Cross-HLA Domain Adaptation (CDAN) | `train_da.py` |
| **Plan 3** | Transfer Learning (Pre-train → Fine-tune) | `train_pretrain.py` + `train_finetune.py` |
| **Plan 4** | Pan-HLA universal model (A+B+C combined) | `train_pretrain.py --hla A B C` |

### Quick start (single run)

```bash
CFG=configs/mhcban_config_clean.yaml
DATA=datasets_clean

# Plan 1 — Standard training
python -u train.py --cfg $CFG --hla A --split random  --dataset_dir $DATA
python -u train.py --cfg $CFG --hla A --split cluster --dataset_dir $DATA

# Plan 2 — Cross-HLA domain adaptation
python -u train_da.py --cfg $CFG --source A --target B --dataset_dir $DATA

# Plan 4 — Pan-HLA universal model
python -u train_pretrain.py --cfg $CFG --hla A B C --split random --dataset_dir $DATA
```

### Plan 3 — Transfer Learning

> ⚠️ Phase 2 (fine-tuning) must be run **after** Phase 1 completes.
> Use `run_plan3_plan4.sh` which chains them with `&&`.

```bash
# Phase 1: Pre-train on source HLA
python -u train_pretrain.py --cfg configs/mhcban_config_clean.yaml \
    --hla A --split random --dataset_dir datasets_clean

# Phase 2: Fine-tune on target HLA
python -u train_finetune.py --cfg configs/mhcban_config_clean.yaml \
    --checkpoint results_clean/pretrain_HLAA_random_seed42 \
    --hla B --split cluster --freeze none --lr 1e-4 --epochs 50 \
    --dataset_dir datasets_clean
```

**Freeze strategy selection:**

| Strategy | Frozen layers | Recommended for |
|----------|--------------|-----------------|
| `encoder` | PeptideBiLSTM + MHCBiLSTM | HLA-C (~1k samples) |
| `all_but_mlp` | encoder + BANLayer | Very small datasets (<500 samples) |
| `none` | None (full fine-tune at low LR) | HLA-B (~29k samples) |

### Full stability test (N = 10 seeds, seeds 42–51)

```bash
mkdir -p logs/stability_clean logs/stability_plan3_plan4_clean

# Plan 1 & 2 (8 experiments, 4 GPUs)
nohup bash run_stability_clean.sh \
    > logs/stability_clean/run_all.log 2>&1 &

# Plan 3 & 4 (simultaneously, 4 GPUs)
nohup bash run_stability_plan3_plan4_clean.sh \
    > logs/stability_plan3_plan4_clean/run_all.log 2>&1 &
```

---

## Evaluation

```bash
# Print full results table (clean dataset)
python aggregate_stability.py --clean

# Compare clean vs. original (PTM-contaminated) results
python aggregate_stability.py --both

# Regenerate publication figures (fig1–fig5)
python plot_results.py

# Gray zone inference (requires gray_zone_inference.csv)
python infer_gray_zone.py
```

---

## Results

> All results use the **PTM-cleaned dataset** (`datasets_clean/`).
> Each experiment was repeated with **10 random seeds (42–51)**; values reported as mean ± SD.
> Full per-seed metrics are stored in `results_clean/` and can be reproduced with
> `python aggregate_stability.py --clean`.

### Plan 1 — Standard Training

| Experiment | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------------|-------|-------|----|-------------|-------------|----------|
| HLA-A Random | 0.977±0.001 | 0.953±0.002 | 0.933±0.003 | 0.927±0.011 | 0.937±0.010 | 0.931±0.005 |
| HLA-B Random | 0.976±0.002 | 0.929±0.007 | 0.929±0.004 | 0.918±0.008 | 0.938±0.007 | 0.922±0.006 |
| HLA-C Random | 0.873±0.010 | 0.905±0.012 | 0.840±0.019 | 0.776±0.073 | 0.886±0.047 | 0.846±0.020 |
| HLA-A Cluster | 0.929±0.014 | 0.912±0.020 | 0.870±0.018 | 0.827±0.034 | 0.902±0.012 | 0.862±0.021 |
| HLA-B Cluster | 0.782±0.030 | 0.545±0.034 | 0.738±0.022 | 0.498±0.115 | 0.878±0.041 | 0.593±0.078 |
| HLA-C Cluster | 0.403±0.066 | 0.746±0.040 | 0.678±0.007 | 0.080±0.048 | 0.984±0.021 | 0.786±0.010 |

### Plan 2 — Cross-HLA Domain Adaptation (CDAN)

| Experiment | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------------|-------|-------|----|-------------|-------------|----------|
| DA: A→B | 0.647±0.019 | 0.316±0.014 | 0.686±0.017 | 0.215±0.152 | 0.930±0.060 | 0.371±0.107 |
| DA: A→C | 0.638±0.057 | 0.767±0.035 | 0.688±0.021 | 0.218±0.175 | 0.930±0.056 | 0.670±0.032 |

### Plan 3 — Transfer Learning (Pre-train → Fine-tune)

| Experiment | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------------|-------|-------|----|-------------|-------------|----------|
| FT HLA-B Cluster (A→B, freeze=none) | 0.768±0.024 | 0.543±0.056 | 0.732±0.015 | 0.546±0.072 | 0.839±0.040 | 0.619±0.047 |
| FT HLA-C Cluster (AB→C, freeze=encoder) | 0.593±0.101 | 0.841±0.043 | 0.688±0.032 | 0.280±0.232 | 0.899±0.099 | 0.764±0.041 |
| FT HLA-C Random (AB→C, freeze=encoder) | 0.885±0.040 | 0.914±0.037 | 0.847±0.024 | 0.726±0.082 | 0.933±0.022 | 0.858±0.023 |

### Plan 4 — Pan-HLA Universal Model

| Experiment | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------------|-------|-------|----|-------------|-------------|----------|
| Pan A+B+C Random | **0.979±0.001** | **0.951±0.002** | **0.934±0.002** | 0.922±0.010 | 0.945±0.007 | 0.928±0.005 |
| Pan A+B+C Cluster | **0.896±0.013** | 0.853±0.015 | 0.827±0.013 | 0.777±0.034 | 0.862±0.024 | 0.811±0.017 |

### Best Method per Scenario (AUROC)

| Scenario | Best method | AUROC |
|----------|-------------|-------|
| HLA-A allele cold-start (unseen allele) | Plan 1 — HLA-A Cluster | 0.929±0.014 |
| HLA-B allele cold-start | Plan 1 — HLA-B Cluster | 0.782±0.030 |
| HLA-C allele cold-start | **Plan 4 — Pan-HLA Cluster** | **0.896±0.013** |
| Universal prediction (all HLA types) | **Plan 4 — Pan-HLA Random** | **0.979±0.001** |

---

## Gray Zone Inference

The model is trained exclusively on the two extremes of binding affinity:
strong binders (IC50 < 100 nM, label = 1) and confirmed non-binders (IC50 > 10,000 nM, label = 0).
We assessed whether the trained Pan-HLA model learned a **continuous affinity ranking** by
applying it to 45,115 gray-zone peptides (IC50 100–10,000 nM) that were never seen during training.

### Correlation with IC50

| Model | Spearman r (−log₁₀ IC50 vs predicted prob.) | *p*-value |
|-------|---------------------------------------------|-----------|
| Pan-HLA Random | **+0.455** | ≈ 0 |
| Pan-HLA Cluster | **+0.383** | ≈ 0 |

A positive Spearman correlation indicates that peptides with lower IC50 (stronger binders)
receive higher predicted binding probabilities, consistent with a learned affinity ordering.

### Mean Predicted Probability by IC50 Bin (Pan-HLA Random)

| IC50 range (nM) | Mean prob. | SD | n |
|-----------------|-----------|-----|---|
| 100–200 | 0.737 | 0.355 | 6,409 |
| 200–500 | 0.636 | 0.393 | 8,142 |
| 500–1,000 | 0.519 | 0.417 | 6,675 |
| 1,000–2,000 | 0.423 | 0.411 | 7,217 |
| 2,000–5,000 | 0.281 | 0.373 | 9,926 |
| 5,000–10,000 | 0.226 | 0.345 | 6,613 |

The predicted probability crosses the 0.5 decision boundary in the 500–1,000 nM bin
(mean = 0.519), which aligns precisely with the **500 nM clinical binder/non-binder threshold**
widely used in immunology.

```bash
# Run gray zone inference (requires gray_zone_inference.csv)
python infer_gray_zone.py
# Output: gray_zone_results.csv, figures/fig6–fig8
```

---

## Discussion

### Why HLA-C Cluster AUROC Is Low

HLA-C has only 11 unique alleles in the dataset; the cluster split assigns only 2 alleles
to the test set (137 samples total). The model learns allele-specific representations during
training that do not transfer to unseen alleles, resulting in poor cold-start generalization
(AUROC 0.403±0.066).

**Solution — Plan 4 Pan-HLA** (AUROC 0.896±0.013): training on all 101 alleles (A+B+C)
simultaneously exposes the model to a much wider range of MHC pseudo-sequences,
substantially improving cold-start performance.

### Why Pan-HLA Works

The 34-position MHC pseudo-sequence encodes the **allele identity** as a fixed-length
amino acid fingerprint. When trained jointly on A+B+C data, the model learns to map
(peptide sequence, MHC pseudo-sequence) pairs to binding probability without any
HLA-type label — inference requires only the peptide and the allele pseudo-sequence.
This design allows **zero-shot extrapolation** to novel alleles not seen during training,
provided their 34-position pseudo-sequence is available (e.g., from NetMHCpan).

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Peptide max length | 30 aa |
| MHC pseudo-seq length | 34 aa |
| Vocabulary size | 25 |
| BiLSTM embedding dim | 128 |
| BiLSTM hidden dim | 128 (bidirectional → 256) |
| BiLSTM layers | 2 |
| BAN heads (glimpses) | 3 |
| MLP dimensions | 256 → 512 → 512 → 128 → 1 |
| Batch size | 64 |
| Learning rate | 1×10⁻³ |
| Epochs | 100 |
| Optimizer | Adam |
| Dropout | 0.2 |
| DA warm-up epochs | 10 |
| DA lambda | 1.0 |
| Fine-tune LR (Plan 3) | 1×10⁻⁴ |
| Fine-tune epochs (Plan 3) | 50 |

---

## Citation

If you use MHCBAN or build upon this work, please cite the following:

```bibtex
@article{bai2023drugban,
  title   = {A bilinear attention network with domain adaptation for drug-target interaction prediction},
  author  = {Bai, Peizhen and others},
  journal = {Nature Machine Intelligence},
  year    = {2023}
}

@article{kim2018ban,
  title   = {Bilinear Attention Networks},
  author  = {Kim, Jin-Hwa and others},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2018}
}

@article{jurtz2017netmhcpan,
  title   = {NetMHCpan-4.0: Improved Peptide-MHC Class I Interaction Predictions
             Integrating Eluted Ligand and Peptide Binding Affinity Data},
  author  = {Jurtz, Vanessa and others},
  journal = {Journal of Immunology},
  year    = {2017}
}
```

---

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.0 |
| GPU | Tesla V100-SXM2-16GB × 4 |
| scikit-learn | — |
| pandas / numpy / PyYAML / tqdm / prettytable | — |
