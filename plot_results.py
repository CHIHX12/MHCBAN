"""
Publication-quality figures for MHCBAN results.
600 DPI, journal-standard formatting, all English.

Output: figures/
  fig1_auroc_overview.pdf/.png/.svg   — All experiments AUROC comparison
  fig2_metrics_heatmap.pdf/.png/.svg  — Full metrics heatmap
  fig3_random_vs_cluster.pdf/.png/.svg — Random vs Cluster per HLA type
  fig4_cold_start_comparison.pdf/.png/.svg — Cold-start scenario best methods
  fig5_pan_vs_individual.pdf/.png/.svg — Pan-HLA vs individual models
"""

import torch, os, glob, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         12,
    'axes.titlesize':    14,
    'axes.labelsize':    13,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   11,
    'figure.dpi':        150,
    'savefig.dpi':       600,
    'axes.linewidth':    1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
    'lines.linewidth':   1.5,
    'errorbar.capsize':  4,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_clean')
SEEDS = list(range(42, 52))
METRICS = ['auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'accuracy']
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_exp(exp_key):
    data = defaultdict(list)
    for seed in SEEDS:
        pattern = os.path.join(RESULTS_DIR, f'{exp_key}_seed{seed}', 'result_metrics.pt')
        matches = glob.glob(pattern)
        if not matches:
            continue
        m = torch.load(matches[0], map_location='cpu', weights_only=False)['test_metrics']
        for metric in METRICS:
            if metric in m:
                data[metric].append(float(m[metric]))
    return data


EXPERIMENTS = {
    # (label, exp_key, group, color_idx)
    'HLA-A Random':           ('HLA_A_random',                    'Random',      0),
    'HLA-B Random':           ('HLA_B_random',                    'Random',      1),
    'HLA-C Random':           ('HLA_C_random',                    'Random',      2),
    'HLA-A Cluster':          ('HLA_A_cluster',                   'Cluster',     0),
    'HLA-B Cluster':          ('HLA_B_cluster',                   'Cluster',     1),
    'HLA-C Cluster':          ('HLA_C_cluster',                   'Cluster',     2),
    'DA: A→B':                ('DA_HLAA_to_HLAB',                 'DA',          3),
    'DA: A→C':                ('DA_HLAA_to_HLAC',                 'DA',          4),
    'FT: A→B Cluster':        ('finetune_HLAB_cluster_freezenone','Transfer',    1),
    'FT: AB→C Cluster':       ('finetune_HLAC_cluster_freezeencoder','Transfer', 2),
    'FT: AB→C Random':        ('finetune_HLAC_random_freezeencoder', 'Transfer', 5),
    'Pan-HLA Random':         ('pretrain_HLAABC_random',          'Pan-HLA',     6),
    'Pan-HLA Cluster':        ('pretrain_HLAABC_cluster',         'Pan-HLA',     7),
}

data = {label: load_exp(key) for label, (key, _, _) in EXPERIMENTS.items()}


# ── Colour palette ─────────────────────────────────────────────────────────────
GROUP_COLORS = {
    'Random':   '#2196F3',   # blue
    'Cluster':  '#FF9800',   # orange
    'DA':       '#9C27B0',   # purple
    'Transfer': '#4CAF50',   # green
    'Pan-HLA':  '#F44336',   # red
}


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — AUROC Overview (all 13 experiments)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_auroc_overview():
    labels = list(EXPERIMENTS.keys())
    groups = [EXPERIMENTS[l][1] for l in labels]
    means  = [np.mean(data[l]['auroc']) if data[l]['auroc'] else 0 for l in labels]
    stds   = [np.std(data[l]['auroc'])  if data[l]['auroc'] else 0 for l in labels]
    colors = [GROUP_COLORS[g] for g in groups]

    fig, ax = plt.subplots(figsize=(13, 5.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  error_kw=dict(ecolor='#333333', lw=1.5, capsize=5, capthick=1.5),
                  edgecolor='white', linewidth=0.5, zorder=3)

    # value labels on top of bars
    for xi, (m, s) in enumerate(zip(means, stds)):
        if m > 0:
            ax.text(xi, m + s + 0.008, f'{m:.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#222222')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10.5)
    ax.set_ylabel('AUROC', fontsize=13)
    ax.set_title('MHCBAN Performance Overview — AUROC (Mean ± SD, n = 10 seeds)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(0.25, 1.08)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.025))
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.0, zorder=2, label='Random (0.5)')

    # legend for groups
    legend_patches = [mpatches.Patch(color=c, label=g, alpha=0.85)
                      for g, c in GROUP_COLORS.items()]
    legend_patches.append(plt.Line2D([0], [0], color='gray', linestyle=':', lw=1.0,
                                     label='Random classifier (0.5)'))
    ax.legend(handles=legend_patches, loc='lower right', framealpha=0.9,
              edgecolor='#cccccc', fontsize=10)

    fig.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'fig1_auroc_overview.{ext}'),
                    dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('fig1 saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Full Metrics Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def fig2_metrics_heatmap():
    labels      = list(EXPERIMENTS.keys())
    metric_names = ['AUROC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy']
    metric_keys  = ['auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'accuracy']

    matrix = np.zeros((len(labels), len(metric_keys)))
    std_mat = np.zeros_like(matrix)
    for i, label in enumerate(labels):
        for j, mk in enumerate(metric_keys):
            v = data[label].get(mk, [])
            matrix[i, j]  = np.mean(v) if v else np.nan
            std_mat[i, j] = np.std(v)  if v else np.nan

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)

    # cell text: mean ± std
    for i in range(len(labels)):
        for j in range(len(metric_keys)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = 'white' if (val < 0.42 or val > 0.88) else 'black'
            ax.text(j, i, f'{val:.3f}\n±{std_mat[i,j]:.3f}',
                    ha='center', va='center', fontsize=8.0,
                    fontweight='bold', color=text_color)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)

    # colour-coded row labels by group
    for i, label in enumerate(labels):
        group = EXPERIMENTS[label][1]
        ax.get_yticklabels()[i].set_color(GROUP_COLORS[group])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Score', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title('MHCBAN — All Metrics Summary (Mean ± SD, n = 10 seeds)',
                 fontsize=14, fontweight='bold', pad=12)

    # group legend
    legend_patches = [mpatches.Patch(color=c, label=g, alpha=0.85)
                      for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_patches, loc='lower right',
              bbox_to_anchor=(1.22, 0.0), framealpha=0.9, fontsize=10)

    fig.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'fig2_metrics_heatmap.{ext}'),
                    dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('fig2 saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Random vs Cluster per HLA type (Plan 1 focus)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_random_vs_cluster():
    hla_types  = ['HLA-A', 'HLA-B', 'HLA-C']
    split_keys = {
        'Random':  ['HLA-A Random',  'HLA-B Random',  'HLA-C Random'],
        'Cluster': ['HLA-A Cluster', 'HLA-B Cluster', 'HLA-C Cluster'],
    }
    metric_keys  = ['auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    metric_names = ['AUROC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    x = np.arange(len(metric_keys))
    width = 0.35

    for ax, hla, rkey, ckey in zip(axes, hla_types,
                                    split_keys['Random'],
                                    split_keys['Cluster']):
        r_means = [np.mean(data[rkey].get(mk, [0])) for mk in metric_keys]
        r_stds  = [np.std( data[rkey].get(mk, [0])) for mk in metric_keys]
        c_means = [np.mean(data[ckey].get(mk, [0])) for mk in metric_keys]
        c_stds  = [np.std( data[ckey].get(mk, [0])) for mk in metric_keys]

        ax.bar(x - width/2, r_means, width, yerr=r_stds, label='Random',
               color='#2196F3', alpha=0.85,
               error_kw=dict(ecolor='#333', lw=1.2, capsize=4), zorder=3)
        ax.bar(x + width/2, c_means, width, yerr=c_stds, label='Cluster',
               color='#FF9800', alpha=0.85,
               error_kw=dict(ecolor='#333', lw=1.2, capsize=4), zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=35, ha='right', fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(hla, fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
        ax.legend(fontsize=10, framealpha=0.9, loc='lower right')
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    fig.suptitle('Random Split vs. Allele Cold-Start (Cluster Split) — Plan 1\n'
                 'Mean ± SD, n = 10 seeds',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'fig3_random_vs_cluster.{ext}'),
                    dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('fig3 saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Cold-Start Scenarios: Best Method per HLA Type
# ══════════════════════════════════════════════════════════════════════════════
def fig4_cold_start():
    scenarios = {
        'HLA-A\nCold-Start': [
            ('Plan 1\nCluster',   'HLA-A Cluster',   GROUP_COLORS['Cluster']),
        ],
        'HLA-B\nCold-Start': [
            ('Plan 1\nCluster',   'HLA-B Cluster',   GROUP_COLORS['Cluster']),
            ('Plan 3\nFine-Tune', 'FT: A→B Cluster', GROUP_COLORS['Transfer']),
            ('Plan 4\nPan-HLA',   'Pan-HLA Cluster', GROUP_COLORS['Pan-HLA']),
        ],
        'HLA-C\nCold-Start': [
            ('Plan 1\nCluster',        'HLA-C Cluster',   GROUP_COLORS['Cluster']),
            ('Plan 3\nFine-Tune',      'FT: AB→C Cluster',GROUP_COLORS['Transfer']),
            ('Plan 4\nPan-HLA',        'Pan-HLA Cluster', GROUP_COLORS['Pan-HLA']),
        ],
        'Universal\n(All HLA)': [
            ('Plan 1 A\nRandom',   'HLA-A Random',    GROUP_COLORS['Random']),
            ('Plan 1 B\nRandom',   'HLA-B Random',    GROUP_COLORS['Random']),
            ('Plan 4\nPan-HLA',    'Pan-HLA Random',  GROUP_COLORS['Pan-HLA']),
        ],
    }

    fig, axes = plt.subplots(1, 4, figsize=(15, 5.5))
    fig.suptitle('Cold-Start & Universal Prediction: Method Comparison (AUROC)\n'
                 'Mean ± SD, n = 10 seeds',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, (scenario, methods) in zip(axes, scenarios.items()):
        labels_m = [m[0] for m in methods]
        means_m  = [np.mean(data[m[1]]['auroc']) if data[m[1]]['auroc'] else 0
                    for m in methods]
        stds_m   = [np.std(data[m[1]]['auroc'])  if data[m[1]]['auroc'] else 0
                    for m in methods]
        colors_m = [m[2] for m in methods]

        x = np.arange(len(methods))
        bars = ax.bar(x, means_m, yerr=stds_m, color=colors_m, alpha=0.85, width=0.55,
                      error_kw=dict(ecolor='#333', lw=1.5, capsize=5), zorder=3,
                      edgecolor='white', linewidth=0.5)

        for xi, (m, s) in enumerate(zip(means_m, stds_m)):
            ax.text(xi, m + s + 0.012, f'{m:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels_m, fontsize=10)
        ax.set_ylim(0.25, 1.12)
        ax.set_title(scenario, fontsize=12, fontweight='bold')
        ax.set_ylabel('AUROC', fontsize=12)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.0, zorder=2)
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    # unified legend — placed in right margin to avoid x-axis overlap
    legend_patches = [mpatches.Patch(color=c, label=g, alpha=0.85)
                      for g, c in GROUP_COLORS.items()
                      if g in ('Random', 'Cluster', 'Transfer', 'Pan-HLA')]
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    fig.legend(handles=legend_patches, loc='center left', ncol=1,
               bbox_to_anchor=(0.89, 0.5), framealpha=0.9, fontsize=11)

    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'fig4_cold_start_comparison.{ext}'),
                    dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('fig4 saved.')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Pan-HLA vs Individual Models (all 6 metrics)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_pan_vs_individual():
    compare = [
        ('HLA-A\nRandom',  'HLA-A Random',   GROUP_COLORS['Random']),
        ('HLA-B\nRandom',  'HLA-B Random',   GROUP_COLORS['Random']),
        ('HLA-C\nRandom',  'HLA-C Random',   GROUP_COLORS['Random']),
        ('Pan-HLA\nRandom','Pan-HLA Random', GROUP_COLORS['Pan-HLA']),
        ('HLA-A\nCluster', 'HLA-A Cluster',  GROUP_COLORS['Cluster']),
        ('HLA-B\nCluster', 'HLA-B Cluster',  GROUP_COLORS['Cluster']),
        ('HLA-C\nCluster', 'HLA-C Cluster',  GROUP_COLORS['Cluster']),
        ('Pan-HLA\nCluster','Pan-HLA Cluster',GROUP_COLORS['Pan-HLA']),
    ]
    metric_keys  = ['auroc', 'auprc', 'f1', 'sensitivity', 'specificity', 'accuracy']
    metric_names = ['AUROC', 'AUPRC', 'F1', 'Sensitivity', 'Specificity', 'Accuracy']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, mk, mn in zip(axes, metric_keys, metric_names):
        labels_p = [c[0] for c in compare]
        means_p  = [np.mean(data[c[1]].get(mk, [0])) for c in compare]
        stds_p   = [np.std( data[c[1]].get(mk, [0])) for c in compare]
        colors_p = [c[2] for c in compare]

        x = np.arange(len(compare))
        ax.bar(x, means_p, yerr=stds_p, color=colors_p, alpha=0.85, width=0.6,
               error_kw=dict(ecolor='#333', lw=1.2, capsize=4), zorder=3,
               edgecolor='white', linewidth=0.5)

        # highlight Pan-HLA bars
        for xi, (c_info, m_val) in enumerate(zip(compare, means_p)):
            if 'Pan-HLA' in c_info[0]:
                ax.get_children()[xi].set_linewidth(2.0)
                ax.get_children()[xi].set_edgecolor('#B71C1C')

        ax.set_xticks(x)
        ax.set_xticklabels(labels_p, fontsize=8.5, rotation=30, ha='right')
        ax.set_ylim(0, 1.12)
        ax.set_title(mn, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11)
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

        # divider between random and cluster groups
        ax.axvline(3.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.text(1.5, 1.06, 'Random Split', ha='center', fontsize=9, color='gray')
        ax.text(5.5, 1.06, 'Cluster Split', ha='center', fontsize=9, color='gray')

    legend_patches = [
        mpatches.Patch(color=GROUP_COLORS['Random'],  label='Individual (Random)',  alpha=0.85),
        mpatches.Patch(color=GROUP_COLORS['Cluster'], label='Individual (Cluster)', alpha=0.85),
        mpatches.Patch(color=GROUP_COLORS['Pan-HLA'], label='Pan-HLA (this work)',  alpha=0.85),
    ]

    fig.suptitle('Pan-HLA Universal Model vs. Individual HLA Models — All Metrics\n'
                 'Mean ± SD, n = 10 seeds',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])
    fig.legend(handles=legend_patches, loc='center left', ncol=1,
               bbox_to_anchor=(0.89, 0.5), fontsize=11, framealpha=0.9)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(OUT_DIR, f'fig5_pan_vs_individual.{ext}'),
                    dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('fig5 saved.')


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Saving figures to: {OUT_DIR}')
    fig1_auroc_overview()
    fig2_metrics_heatmap()
    fig3_random_vs_cluster()
    fig4_cold_start()
    fig5_pan_vs_individual()
    print('\nAll figures saved (PDF + PNG, 600 DPI).')
