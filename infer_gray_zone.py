"""
Run Pan-HLA model inference on gray-zone peptides (IC50 100–10,000 nM).
Uses both Pan-HLA Random and Pan-HLA Cluster best checkpoints (seed 42).

Output: gray_zone_results.csv  +  gray_zone_figures/
"""
import torch, os, yaml, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr, spearmanr

from models.mhcban import MHCBAN
from datasets.dataloader import MHCDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CFG_PATH  = 'configs/mhcban_config_clean.yaml'
GRAY_CSV  = 'gray_zone_inference.csv'
OUT_CSV   = 'gray_zone_results.csv'
OUT_DIR   = 'gray_zone_figures'
os.makedirs(OUT_DIR, exist_ok=True)

CHECKPOINTS = {
    'Pan-HLA Random':  'results_clean/pretrain_HLAABC_random_seed42/best_model_epoch_15.pth',
    'Pan-HLA Cluster': 'results_clean/pretrain_HLAABC_cluster_seed42/best_model_epoch_12.pth',
}

# ── Global plot style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'legend.fontsize': 11, 'savefig.dpi': 600,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
with open(CFG_PATH) as f:
    config = yaml.safe_load(f)

df_gray = pd.read_csv(GRAY_CSV)
df_gray = df_gray[['Peptide', 'Pseudo_Sequence', 'IC50_nM', 'MHC', 'HLA_type']].copy()

# Add dummy Label column (not used for loss, just for MHCDataset compatibility)
df_gray['Label'] = 0
dataset = MHCDataset(df_gray)
loader  = DataLoader(dataset, batch_size=256, shuffle=False)

# ── Inference function ────────────────────────────────────────────────────────
def run_inference(ckpt_path):
    model = MHCBAN(**config).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    preds = []
    with torch.no_grad():
        for v_pep, v_mhc, _ in loader:
            v_pep, v_mhc = v_pep.to(device), v_mhc.to(device)
            _, _, score, _ = model(v_pep, v_mhc, mode="eval")
            prob = torch.sigmoid(score)
            preds.extend(prob.cpu().numpy().flatten().tolist())
    return np.array(preds)

# ── Run both models ───────────────────────────────────────────────────────────
print(f'Gray zone peptides: {len(df_gray)}')
print(f'Device: {device}\n')

for model_name, ckpt in CHECKPOINTS.items():
    print(f'Running {model_name} ...')
    preds = run_inference(ckpt)
    col = model_name.replace(' ', '_').replace('-', '_')
    df_gray[col] = preds
    r, p = spearmanr(-np.log10(df_gray['IC50_nM']), preds)
    print(f'  Spearman r (log-IC50 vs pred): {r:.4f}  (p={p:.2e})')

df_gray.to_csv(OUT_CSV, index=False)
print(f'\nResults saved: {OUT_CSV}')

# ── Bin analysis ──────────────────────────────────────────────────────────────
bins   = [100, 200, 500, 1000, 2000, 5000, 10000]
labels = ['100–200', '200–500', '500–1k', '1k–2k', '2k–5k', '5k–10k']
df_gray['IC50_bin'] = pd.cut(df_gray['IC50_nM'], bins=bins, labels=labels)

print('\nPrediction by IC50 bin (Pan-HLA Random):')
g = df_gray.groupby('IC50_bin', observed=True)['Pan_HLA_Random']
for bin_label, grp in g:
    print(f'  {bin_label} nM: mean={grp.mean():.3f}  std={grp.std():.3f}  n={len(grp)}')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Scatter — log(IC50) vs predicted probability
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (model_name, ckpt) in zip(axes, CHECKPOINTS.items()):
    col = model_name.replace(' ', '_').replace('-', '_')
    x = np.log10(df_gray['IC50_nM'])
    y = df_gray[col]

    # colour by HLA type
    colors_map = {'A': '#2196F3', 'B': '#FF9800', 'C': '#4CAF50'}
    for hla, grp in df_gray.groupby('HLA_type'):
        xi = np.log10(grp['IC50_nM'])
        yi = grp[col]
        ax.scatter(xi, yi, c=colors_map[hla], alpha=0.15, s=4, label=f'HLA-{hla}',
                   rasterized=True)

    # trend line
    z = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    ax.plot(xline, np.polyval(z, xline), 'k-', lw=1.5, zorder=5)

    r, _ = spearmanr(x, y)
    ax.set_xlabel('log$_{10}$(IC50 nM)', fontsize=13)
    ax.set_ylabel('Predicted Binding Probability', fontsize=13)
    ax.set_title(f'{model_name}\nSpearman r = {r:.3f}', fontsize=13, fontweight='bold')

    # reference lines
    ax.axhline(0.5, color='gray', linestyle='--', lw=1.0, alpha=0.7, label='P = 0.5')
    ax.axvline(np.log10(500),  color='#E91E63', linestyle=':', lw=1.2, alpha=0.8,
               label='500 nM')
    ax.axvline(np.log10(2000), color='#9C27B0', linestyle=':', lw=1.2, alpha=0.8,
               label='2,000 nM')

    ax.set_xlim(np.log10(95), np.log10(10500))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right', fontsize=10, markerscale=4, framealpha=0.9)

    # x-tick labels as nM values
    xticks = np.log10([100, 200, 500, 1000, 2000, 5000, 10000])
    ax.set_xticks(xticks)
    ax.set_xticklabels(['100', '200', '500', '1k', '2k', '5k', '10k'])

fig.suptitle('Pan-HLA Model Inference on Gray Zone Peptides (IC50 100–10,000 nM)\n'
             'All HLA-A + B + C  |  n = {:,}'.format(len(df_gray)),
             fontsize=14, fontweight='bold')
fig.tight_layout()
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT_DIR, f'gray1_scatter.{ext}'),
                dpi=600, bbox_inches='tight')
plt.close(fig)
print('gray1_scatter saved.')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Box plot — predicted probability per IC50 bin
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (model_name, _) in zip(axes, CHECKPOINTS.items()):
    col = model_name.replace(' ', '_').replace('-', '_')
    groups = [df_gray[df_gray['IC50_bin'] == lbl][col].dropna().values
              for lbl in labels]
    bp = ax.boxplot(groups, patch_artist=True, medianprops=dict(color='black', lw=2),
                    flierprops=dict(marker='.', markersize=2, alpha=0.3),
                    whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))

    cmap = plt.cm.RdYlGn_r
    for patch, lbl in zip(bp['boxes'], labels):
        mid_ic50 = {'100–200': 150, '200–500': 350, '500–1k': 750,
                    '1k–2k': 1500, '2k–5k': 3500, '5k–10k': 7500}[lbl]
        norm_val = (np.log10(mid_ic50) - np.log10(100)) / (np.log10(10000) - np.log10(100))
        patch.set_facecolor(cmap(norm_val))
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_xlabel('IC50 Range (nM)', fontsize=13)
    ax.set_ylabel('Predicted Binding Probability', fontsize=13)
    ax.set_title(model_name, fontsize=13, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', lw=1.0, alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)

    # annotate n
    for i, grp in enumerate(groups):
        ax.text(i + 1, -0.04, f'n={len(grp):,}', ha='center', fontsize=8.5, color='gray')

fig.suptitle('Predicted Binding Probability by IC50 Bin — Gray Zone\n'
             'Pan-HLA model (trained on IC50 < 100 nM  and  > 10,000 nM only)',
             fontsize=14, fontweight='bold')
fig.tight_layout()
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT_DIR, f'gray2_boxplot.{ext}'),
                dpi=600, bbox_inches='tight')
plt.close(fig)
print('gray2_boxplot saved.')

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Histogram of predicted probabilities (Random model)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)

for ax, hla in zip(axes, ['A', 'B', 'C']):
    sub = df_gray[df_gray['HLA_type'] == hla]
    ax.hist(sub['Pan_HLA_Random'], bins=40, color='#2196F3', alpha=0.75,
            edgecolor='white', linewidth=0.4)
    ax.axvline(0.5, color='red', linestyle='--', lw=1.5, label='P = 0.5')
    ax.set_xlabel('Predicted Binding Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'HLA-{hla}  (n={len(sub):,})', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)

    pct_above = (sub['Pan_HLA_Random'] > 0.5).mean() * 100
    ax.text(0.98, 0.97, f'{pct_above:.1f}% > 0.5', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold',
            color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle('Pan-HLA Random — Predicted Probability Distribution for Gray Zone\n'
             '(IC50 100–10,000 nM, model never saw these during training)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(os.path.join(OUT_DIR, f'gray3_histogram.{ext}'),
                dpi=600, bbox_inches='tight')
plt.close(fig)
print('gray3_histogram saved.')

print('\nAll gray zone figures saved to:', OUT_DIR)
