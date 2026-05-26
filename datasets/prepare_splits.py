"""
Pre-split binary CSVs and save to disk, mirroring DrugBAN datasets structure.

Supports two split strategies:
  random  — sample-level random split (7:2:1)
  cluster — allele-level cold-start split (train/val/test contain DIFFERENT alleles)

Usage:
    python datasets/prepare_splits.py --data ..               # both strategies
    python datasets/prepare_splits.py --data .. --split random
    python datasets/prepare_splits.py --data .. --split cluster
"""
import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data',        default='..', type=str)
parser.add_argument('--split',       default='both',
                    choices=['random', 'cluster', 'both'])
parser.add_argument('--train_ratio', default=0.7, type=float)
parser.add_argument('--val_ratio',   default=0.2, type=float)
parser.add_argument('--seed',        default=42,  type=int)
args = parser.parse_args()

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(args.seed)


def save_split(df_train, df_val, df_test, out_dir, hla, tag):
    os.makedirs(out_dir, exist_ok=True)
    df_train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    df_val.to_csv(  os.path.join(out_dir, 'val.csv'),   index=False)
    df_test.to_csv( os.path.join(out_dir, 'test.csv'),  index=False)
    print(f"  [{tag}] train={len(df_train)} | val={len(df_val)} | test={len(df_test)}"
          f"  → {out_dir}")


def random_split(df, hla_dir):
    df_s = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n       = len(df_s)
    n_train = int(n * args.train_ratio)
    n_val   = int(n * args.val_ratio)
    save_split(df_s.iloc[:n_train],
               df_s.iloc[n_train:n_train + n_val],
               df_s.iloc[n_train + n_val:],
               os.path.join(hla_dir, 'random'), '', 'random')


def cluster_split(df, hla_dir):
    """
    Allele-level cold-start split.
    Train / val / test contain COMPLETELY DIFFERENT alleles (no overlap).
    Allele ratio: 70% train | 20% val | 10% test  (by number of alleles)
    """
    alleles = df['MHC'].unique()
    alleles = rng.permutation(alleles)          # shuffle alleles

    n_alleles = len(alleles)
    n_train_a = max(1, int(n_alleles * args.train_ratio))
    n_val_a   = max(1, int(n_alleles * args.val_ratio))
    # ensure at least 1 allele in test
    n_train_a = min(n_train_a, n_alleles - 2)
    n_val_a   = min(n_val_a,   n_alleles - n_train_a - 1)

    train_alleles = set(alleles[:n_train_a])
    val_alleles   = set(alleles[n_train_a:n_train_a + n_val_a])
    test_alleles  = set(alleles[n_train_a + n_val_a:])

    df_train = df[df['MHC'].isin(train_alleles)].reset_index(drop=True)
    df_val   = df[df['MHC'].isin(val_alleles)].reset_index(drop=True)
    df_test  = df[df['MHC'].isin(test_alleles)].reset_index(drop=True)

    out_dir = os.path.join(hla_dir, 'cluster')
    save_split(df_train, df_val, df_test, out_dir, '', 'cluster')

    # print allele assignment for traceability (hospital audit)
    audit_path = os.path.join(out_dir, 'allele_assignment.txt')
    with open(audit_path, 'w') as f:
        f.write(f"seed={args.seed}\n\n")
        f.write(f"TRAIN alleles ({len(train_alleles)}):\n")
        f.write('\n'.join(sorted(train_alleles)) + '\n\n')
        f.write(f"VAL alleles ({len(val_alleles)}):\n")
        f.write('\n'.join(sorted(val_alleles)) + '\n\n')
        f.write(f"TEST alleles ({len(test_alleles)}):\n")
        f.write('\n'.join(sorted(test_alleles)) + '\n')
    print(f"    allele_assignment.txt saved → {audit_path}")


# ── Main ──────────────────────────────────────────────────────────────
for hla in ['A', 'B', 'C']:
    src_csv = os.path.join(args.data, f'HLA_{hla}', f'binary_train_HLA_{hla}.csv')
    hla_dir = os.path.join(OUT_ROOT, f'HLA_{hla}')
    os.makedirs(hla_dir, exist_ok=True)

    df = pd.read_csv(src_csv)
    print(f"\nHLA-{hla}: {len(df)} samples | {df['MHC'].nunique()} alleles")

    # full.csv — always saved
    df.to_csv(os.path.join(hla_dir, 'full.csv'), index=False)

    if args.split in ('random', 'both'):
        random_split(df, hla_dir)
    if args.split in ('cluster', 'both'):
        cluster_split(df, hla_dir)

print("\nAll splits saved.")
