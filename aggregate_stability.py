"""
Aggregate stability test results (seeds 42–51) into mean ± std table.

Usage:
    python aggregate_stability.py                    # original dirty data (results/)
    python aggregate_stability.py --clean            # clean data (results_clean/)
    python aggregate_stability.py --both             # side-by-side comparison
"""
import torch, os, glob, argparse, numpy as np
from collections import defaultdict

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIRTY = os.path.join(BASE_DIR, 'results')
RESULTS_CLEAN = os.path.join(BASE_DIR, 'results_clean')
SEEDS = list(range(42, 52))

EXPERIMENTS_1_2 = [
    ('HLA_A_random',    'HLA-A Random'),
    ('HLA_B_random',    'HLA-B Random'),
    ('HLA_C_random',    'HLA-C Random'),
    ('HLA_A_cluster',   'HLA-A Cluster'),
    ('HLA_B_cluster',   'HLA-B Cluster'),
    ('HLA_C_cluster',   'HLA-C Cluster'),
    ('DA_HLAA_to_HLAB', 'DA: A→B'),
    ('DA_HLAA_to_HLAC', 'DA: A→C'),
]

EXPERIMENTS_3_4 = [
    ('finetune_HLAB_cluster_freezenone',       'FT HLA-B cluster (A→B)'),
    ('finetune_HLAC_cluster_freezeencoder',    'FT HLA-C cluster (AB→C)'),
    ('finetune_HLAC_random_freezeencoder',     'FT HLA-C random  (AB→C)'),
    ('pretrain_HLAABC_random',                 'Pan A+B+C random'),
    ('pretrain_HLAABC_cluster',                'Pan A+B+C cluster'),
]

METRICS = ['auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'accuracy']


def load_experiments(experiment_list, results_dir):
    data = defaultdict(lambda: defaultdict(list))
    for exp_key, _ in experiment_list:
        for seed in SEEDS:
            pattern = os.path.join(results_dir, f'{exp_key}_seed{seed}', 'result_metrics.pt')
            matches = glob.glob(pattern)
            if not matches:
                continue
            d = torch.load(matches[0], map_location='cpu', weights_only=False)
            m = d['test_metrics']
            for metric in METRICS:
                if metric in m:
                    data[exp_key][metric].append(float(m[metric]))
    return data


def print_table(title, experiment_list, data):
    print(f"\n{'='*len(title)}")
    print(title)
    print('='*len(title))
    header = f"{'Experiment':<28} {'AUROC':>14} {'AUPRC':>14} {'F1':>14} {'Sens':>14} {'Spec':>14} {'Acc':>14} {'N':>4}"
    print(header)
    print('-' * len(header))
    for exp_key, exp_name in experiment_list:
        vals = data[exp_key]
        if not vals:
            print(f"{exp_name:<28}  (no results)")
            continue
        n = len(vals.get('auroc', []))
        row = f"{exp_name:<28}"
        for metric in METRICS:
            v = vals.get(metric, [])
            if v:
                row += f"  {np.mean(v):.3f}±{np.std(v):.3f}"
            else:
                row += f"  {'N/A':>12}"
        row += f"  {n:>4}"
        print(row)


def print_comparison(title, experiment_list, data_dirty, data_clean):
    print(f"\n{'='*len(title)}")
    print(title)
    print('='*len(title))
    header = f"{'Experiment':<28} {'Dirty AUROC':>16} {'Clean AUROC':>16} {'Delta':>8} {'N_d':>4} {'N_c':>4}"
    print(header)
    print('-' * len(header))
    for exp_key, exp_name in experiment_list:
        d_vals = data_dirty[exp_key].get('auroc', [])
        c_vals = data_clean[exp_key].get('auroc', [])
        d_str = f"{np.mean(d_vals):.3f}±{np.std(d_vals):.3f}" if d_vals else "N/A"
        c_str = f"{np.mean(c_vals):.3f}±{np.std(c_vals):.3f}" if c_vals else "N/A"
        delta = f"{np.mean(c_vals)-np.mean(d_vals):+.3f}" if d_vals and c_vals else "N/A"
        print(f"{exp_name:<28} {d_str:>16} {c_str:>16} {delta:>8} {len(d_vals):>4} {len(c_vals):>4}")


parser = argparse.ArgumentParser()
parser.add_argument('--clean', action='store_true', help='show clean data results (results_clean/)')
parser.add_argument('--both',  action='store_true', help='show dirty vs clean comparison')
args = parser.parse_args()

if args.both:
    d12_dirty = load_experiments(EXPERIMENTS_1_2, RESULTS_DIRTY)
    d34_dirty = load_experiments(EXPERIMENTS_3_4, RESULTS_DIRTY)
    d12_clean = load_experiments(EXPERIMENTS_1_2, RESULTS_CLEAN)
    d34_clean = load_experiments(EXPERIMENTS_3_4, RESULTS_CLEAN)
    print_comparison("方案1&2 — Dirty vs Clean (AUROC)", EXPERIMENTS_1_2, d12_dirty, d12_clean)
    print_comparison("方案3&4 — Dirty vs Clean (AUROC)", EXPERIMENTS_3_4, d34_dirty, d34_clean)
elif args.clean:
    results_dir = RESULTS_CLEAN
    data_12 = load_experiments(EXPERIMENTS_1_2, results_dir)
    data_34 = load_experiments(EXPERIMENTS_3_4, results_dir)
    print_table("方案1&2 — 標準訓練 & DA（乾淨資料）", EXPERIMENTS_1_2, data_12)
    print_table("方案3&4 — Transfer Learning & Pan-HLA（乾淨資料）", EXPERIMENTS_3_4, data_34)
    print(f"\nResults from: {results_dir}")
else:
    results_dir = RESULTS_DIRTY
    data_12 = load_experiments(EXPERIMENTS_1_2, results_dir)
    data_34 = load_experiments(EXPERIMENTS_3_4, results_dir)
    print_table("方案1&2 — 標準訓練 & Domain Adaptation", EXPERIMENTS_1_2, data_12)
    print_table("方案3&4 — Transfer Learning & Pan-HLA",   EXPERIMENTS_3_4, data_34)
    print(f"\nResults from: {results_dir}")

print(f"Seeds used: {SEEDS}")
