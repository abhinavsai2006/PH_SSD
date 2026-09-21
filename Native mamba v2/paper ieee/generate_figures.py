import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style for academic publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'text.usetex': False,  # Robust across machines without full latex font setup
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

OUTPUT_DIR = Path(r"e:\DL Project\Native mamba v2\paper ieee\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data parsed exactly from values.txtx
CONFIGS = [
    'baseline',
    'mamba2_hedo',
    'mamba2_msa',
    'hedo_msa',
    'avsc_msa',
    'mamba2_avsc',
    'full',
    'hedo_avsc'
]

CONFIG_LABELS = {
    'baseline': 'Baseline (Linear Proj.)',
    'mamba2_hedo': 'Mamba-2 + HEDO',
    'mamba2_msa': 'Mamba-2 + MSA',
    'hedo_msa': 'HEDO + MSA',
    'avsc_msa': 'AVSC + MSA',
    'mamba2_avsc': 'Mamba-2 + AVSC',
    'full': 'Full Model (All)',
    'hedo_avsc': 'HEDO + AVSC'
}

# Ablation summary data: means and standard deviations
ABLATION_DATA = {
    'baseline': {
        'mean_recall': (37.1578, 0.5711),
        'i2t_r1': (16.5333, 0.7024),
        'i2t_r5': (43.2667, 0.7234),
        'i2t_r10': (57.4667, 1.1372),
        't2i_r1': (13.9200, 0.4468),
        't2i_r5': (38.3133, 0.3802),
        't2i_r10': (53.4467, 0.4917),
        'i2t_medr': (7.6667, 0.5774),
        't2i_medr': (9.0000, 0.0000)
    },
    'mamba2_hedo': {
        'mean_recall': (37.3111, 0.6835),
        'i2t_r1': (17.4000, 0.6245),
        'i2t_r5': (43.4667, 1.8502),
        'i2t_r10': (57.2000, 1.1000),
        't2i_r1': (14.2000, 0.5821),
        't2i_r5': (38.2467, 0.4688),
        't2i_r10': (53.3533, 0.5005),
        'i2t_medr': (7.6667, 0.5774),
        't2i_medr': (9.0000, 0.0000)
    },
    'mamba2_msa': {
        'mean_recall': (26.8867, 0.6829),
        'i2t_r1': (9.7667, 0.5132),
        'i2t_r5': (29.5333, 0.9452),
        'i2t_r10': (42.9000, 1.0440),
        't2i_r1': (8.9200, 0.4678),
        't2i_r5': (28.2200, 0.5923),
        't2i_r10': (41.9800, 0.7454),
        'i2t_medr': (13.8333, 0.7638),
        't2i_medr': (14.6667, 0.5774)
    },
    'hedo_msa': {
        'mean_recall': (26.4089, 0.4900),
        'i2t_r1': (9.4667, 0.4726),
        'i2t_r5': (29.5000, 1.4731),
        'i2t_r10': (42.8667, 1.5948),
        't2i_r1': (8.3333, 0.4168),
        't2i_r5': (27.4467, 0.0924),
        't2i_r10': (40.8400, 0.8400),
        'i2t_medr': (14.6667, 0.5774),
        't2i_medr': (15.3333, 0.5774)
    },
    'avsc_msa': {
        'mean_recall': (6.6422, 0.8208),
        'i2t_r1': (1.4667, 0.3215),
        'i2t_r5': (6.4000, 1.8682),
        'i2t_r10': (11.5000, 1.8358),
        't2i_r1': (1.5000, 0.2600),
        't2i_r5': (6.5067, 0.2831),
        't2i_r10': (12.4800, 0.7608),
        'i2t_medr': (81.6667, 5.8381),
        't2i_medr': (61.3333, 2.8868)
    },
    'mamba2_avsc': {
        'mean_recall': (6.1200, 1.1352),
        'i2t_r1': (1.8000, 0.7810),
        'i2t_r5': (6.2667, 1.1015),
        'i2t_r10': (11.8667, 0.9609),
        't2i_r1': (1.0467, 0.3215),
        't2i_r5': (5.5467, 1.3754),
        't2i_r10': (10.1933, 2.7353),
        'i2t_medr': (99.8333, 21.5019),
        't2i_medr': (72.3333, 14.7422)
    },
    'full': {
        'mean_recall': (5.9511, 0.4203),
        'i2t_r1': (1.4667, 0.3055),
        'i2t_r5': (5.9667, 0.4726),
        'i2t_r10': (10.7333, 0.6807),
        't2i_r1': (1.2467, 0.2859),
        't2i_r5': (5.6000, 0.6920),
        't2i_r10': (10.6933, 1.0874),
        'i2t_medr': (94.3333, 6.6583),
        't2i_medr': (69.6667, 3.7859)
    },
    'hedo_avsc': {
        'mean_recall': (5.0444, 0.8381),
        'i2t_r1': (1.0333, 0.1155),
        'i2t_r5': (4.9333, 0.8963),
        'i2t_r10': (9.2667, 1.0504),
        't2i_r1': (1.0067, 0.2318),
        't2i_r5': (4.8267, 1.1545),
        't2i_r10': (9.2000, 1.8880),
        'i2t_medr': (103.1667, 10.1283),
        't2i_medr': (79.0000, 13.2288)
    }
}

# Paired deltas relative to baseline
PAIRED_DELTAS = [
    {
        'comparison': 'Mamba-2 + HEDO',
        'mean_delta': 0.1533,
        'ci_low': -1.4246,
        'ci_high': 1.7313,
        'verdict': 'NO STAT. DIFF. (Not Supported)',
        'color': '#2b7bba'
    },
    {
        'comparison': 'Mamba-2 + MSA',
        'mean_delta': -10.2711,
        'ci_low': -13.1018,
        'ci_high': -7.4404,
        'verdict': 'CONTRADICTED (-10.27 pp)',
        'color': '#d95f02'
    },
    {
        'comparison': 'HEDO + MSA',
        'mean_delta': -10.7489,
        'ci_low': -12.3816,
        'ci_high': -9.1161,
        'verdict': 'CONTRADICTED (-10.75 pp)',
        'color': '#d95f02'
    },
    {
        'comparison': 'AVSC + MSA',
        'mean_delta': -30.5156,
        'ci_low': -32.4947,
        'ci_high': -28.5364,
        'verdict': 'CONTRADICTED (-30.52 pp)',
        'color': '#e41a1c'
    },
    {
        'comparison': 'Mamba-2 + AVSC',
        'mean_delta': -31.0378,
        'ci_low': -35.2627,
        'ci_high': -26.8129,
        'verdict': 'CONTRADICTED (-31.04 pp)',
        'color': '#e41a1c'
    },
    {
        'comparison': 'Full Model (All)',
        'mean_delta': -31.2067,
        'ci_low': -32.5987,
        'ci_high': -29.8146,
        'verdict': 'CONTRADICTED (-31.21 pp)',
        'color': '#e41a1c'
    },
    {
        'comparison': 'HEDO + AVSC',
        'mean_delta': -32.1133,
        'ci_low': -35.6092,
        'ci_high': -28.6174,
        'verdict': 'CONTRADICTED (-32.11 pp)',
        'color': '#e41a1c'
    }
]

# Efficiency data
EFFICIENCY_DATA = {
    'baseline': {
        'trainable_params': 465177,
        'peak_vram_mb': 1042.31,
        'train_seconds': 278.44,
        'head_throughput': 9549.93,
        'e2e_throughput': 5240.09,
        'mean_recall': 37.1578
    },
    'mamba2_hedo_v2': {
        'trainable_params': 597787,
        'peak_vram_mb': 1086.15,
        'train_seconds': 362.76,
        'head_throughput': 7054.12,
        'e2e_throughput': 4765.93,
        'mean_recall': 37.3111
    },
    'mamba2_msa_v2': {
        'trainable_params': 565273,
        'peak_vram_mb': 1045.36,
        'train_seconds': 306.34,
        'head_throughput': 8916.05,
        'e2e_throughput': 5089.16,
        'mean_recall': 26.8867
    },
    'hedo_msa_v2': {
        'trainable_params': 697883,
        'peak_vram_mb': 1088.07,
        'train_seconds': 399.01,
        'head_throughput': 6902.35,
        'e2e_throughput': 4418.65,
        'mean_recall': 26.4089
    },
    'avsc_msa_v2': {
        'trainable_params': 631321,
        'peak_vram_mb': 1046.63,
        'train_seconds': 367.30,
        'head_throughput': 7029.26,
        'e2e_throughput': 4354.75,
        'mean_recall': 6.6422
    },
    'mamba2_avsc_v2': {
        'trainable_params': 531225,
        'peak_vram_mb': 1043.18,
        'train_seconds': 329.82,
        'head_throughput': 7773.80,
        'e2e_throughput': 4718.49,
        'mean_recall': 6.1200
    },
    'full_v2': {
        'trainable_params': 763931,
        'peak_vram_mb': 1089.33,
        'train_seconds': 463.06,
        'head_throughput': 5671.25,
        'e2e_throughput': 3749.01,
        'mean_recall': 5.9511
    },
    'hedo_avsc_v2': {
        'trainable_params': 663835,
        'peak_vram_mb': 1087.42,
        'train_seconds': 427.66,
        'head_throughput': 6500.59,
        'e2e_throughput': 4112.24,
        'mean_recall': 5.0444
    }
}

# Per-seed exact values
PER_SEED_RUNS = {
    'baseline': {'mr': [37.7967, 36.6967, 36.9800], 'i2t_r1': [17.2, 16.6, 15.8], 't2i_r1': [14.42, 13.56, 13.78]},
    'mamba2_hedo_v2': {'mr': [37.5233, 36.5467, 37.8633], 'i2t_r1': [17.6, 16.7, 17.9], 't2i_r1': [13.76, 13.98, 14.86]},
    'mamba2_msa_v2': {'mr': [26.5400, 27.6733, 26.4467], 'i2t_r1': [9.9, 10.2, 9.2], 't2i_r1': [8.66, 9.46, 8.64]},
    'hedo_msa_v2': {'mr': [26.6767, 26.7067, 25.8433], 'i2t_r1': [9.1, 10.0, 9.3], 't2i_r1': [7.88, 8.70, 8.42]},
    'avsc_msa_v2': {'mr': [6.7833, 5.7600, 7.3833], 'i2t_r1': [1.1, 1.6, 1.7], 't2i_r1': [1.50, 1.24, 1.76]},
    'mamba2_avsc_v2': {'mr': [4.8133, 6.8633, 6.6833], 'i2t_r1': [0.9, 2.3, 2.2], 't2i_r1': [0.68, 1.28, 1.18]},
    'full_v2': {'mr': [6.2467, 6.1367, 5.4700], 'i2t_r1': [1.8, 1.2, 1.4], 't2i_r1': [1.18, 1.56, 1.00]},
    'hedo_avsc_v2': {'mr': [4.0867, 5.6433, 5.4033], 'i2t_r1': [0.9, 1.1, 1.1], 't2i_r1': [0.74, 1.16, 1.12]}
}


def plot_factorial_recall():
    """Figure 1: Grouped bar chart with error bars across all 8 configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)

    configs_ordered = [
        'baseline',
        'mamba2_hedo_v2',
        'mamba2_msa_v2',
        'hedo_msa_v2',
        'avsc_msa_v2',
        'mamba2_avsc_v2',
        'full_v2',
        'hedo_avsc_v2'
    ]
    labels = [CONFIG_LABELS[c] for c in configs_ordered]
    x = np.arange(len(configs_ordered))
    width = 0.22

    # Panel A: Image-to-Text (i2t) Recall
    i2t_r1 = [ABLATION_DATA[c]['i2t_r1'][0] for c in configs_ordered]
    i2t_r1_err = [ABLATION_DATA[c]['i2t_r1'][1] for c in configs_ordered]

    i2t_r5 = [ABLATION_DATA[c]['i2t_r5'][0] for c in configs_ordered]
    i2t_r5_err = [ABLATION_DATA[c]['i2t_r5'][1] for c in configs_ordered]

    i2t_r10 = [ABLATION_DATA[c]['i2t_r10'][0] for c in configs_ordered]
    i2t_r10_err = [ABLATION_DATA[c]['i2t_r10'][1] for c in configs_ordered]

    rects1 = ax1.bar(x - width, i2t_r1, width, yerr=i2t_r1_err, capsize=3, label='Recall@1', color='#1f77b4', edgecolor='black', linewidth=0.6)
    rects2 = ax1.bar(x, i2t_r5, width, yerr=i2t_r5_err, capsize=3, label='Recall@5', color='#aec7e8', edgecolor='black', linewidth=0.6)
    rects3 = ax1.bar(x + width, i2t_r10, width, yerr=i2t_r10_err, capsize=3, label='Recall@10', color='#c7c7c7', edgecolor='black', linewidth=0.6)

    ax1.set_title('(a) Image-to-Text (i2t) Retrieval Performance', fontweight='bold', pad=10)
    ax1.set_ylabel('Recall (%)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0, 65)

    # Panel B: Text-to-Image (t2i) Recall
    t2i_r1 = [ABLATION_DATA[c]['t2i_r1'][0] for c in configs_ordered]
    t2i_r1_err = [ABLATION_DATA[c]['t2i_r1'][1] for c in configs_ordered]

    t2i_r5 = [ABLATION_DATA[c]['t2i_r5'][0] for c in configs_ordered]
    t2i_r5_err = [ABLATION_DATA[c]['t2i_r5'][1] for c in configs_ordered]

    t2i_r10 = [ABLATION_DATA[c]['t2i_r10'][0] for c in configs_ordered]
    t2i_r10_err = [ABLATION_DATA[c]['t2i_r10'][1] for c in configs_ordered]

    ax2.bar(x - width, t2i_r1, width, yerr=t2i_r1_err, capsize=3, label='Recall@1', color='#ff7f0e', edgecolor='black', linewidth=0.6)
    ax2.bar(x, t2i_r5, width, yerr=t2i_r5_err, capsize=3, label='Recall@5', color='#ffbb78', edgecolor='black', linewidth=0.6)
    ax2.bar(x + width, t2i_r10, width, yerr=t2i_r10_err, capsize=3, label='Recall@10', color='#dbdb8d', edgecolor='black', linewidth=0.6)

    ax2.set_title('(b) Text-to-Image (t2i) Retrieval Performance', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Add dividing lines for visual grouping
    for ax in (ax1, ax2):
        ax.axvline(1.5, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(3.5, color='gray', linestyle=':', alpha=0.7)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_factorial_recall.pdf")
    fig.savefig(OUTPUT_DIR / "fig_factorial_recall.png")
    plt.close(fig)
    print("Saved fig_factorial_recall.pdf and .png")


def plot_paired_deltas():
    """Figure 2: Forest plot of paired deltas relative to baseline with 95% CIs."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    comparisons = [d['comparison'] for d in reversed(PAIRED_DELTAS)]
    means = [d['mean_delta'] for d in reversed(PAIRED_DELTAS)]
    ci_lows = [d['ci_low'] for d in reversed(PAIRED_DELTAS)]
    ci_highs = [d['ci_high'] for d in reversed(PAIRED_DELTAS)]
    colors = [d['color'] for d in reversed(PAIRED_DELTAS)]
    verdicts = [d['verdict'] for d in reversed(PAIRED_DELTAS)]

    y = np.arange(len(comparisons))

    # Plot zero reference line
    ax.axvline(0, color='black', linestyle='--', linewidth=1.2, zorder=1, label='Baseline ($\Delta = 0$)')

    # Plot error bars
    xerr_low = [m - cl for m, cl in zip(means, ci_lows)]
    xerr_high = [ch - m for m, ch in zip(means, ci_highs)]

    ax.errorbar(means, y, xerr=[xerr_low, xerr_high], fmt='o', color='navy', ecolor='navy',
                elinewidth=2.2, capsize=5, capthick=1.8, markersize=8, zorder=3)

    # Category dividing lines
    ax.axhline(3.5, color='gray', linestyle=':', alpha=0.45, linewidth=1.0)
    ax.axhline(5.5, color='gray', linestyle=':', alpha=0.45, linewidth=1.0)

    # Annotate points with verdict and numbers without overlapping
    for i, (m, y_pos, v, col, cl, ch) in enumerate(zip(means, y, verdicts, colors, ci_lows, ci_highs)):
        sign = "+" if m > 0 else ""
        if 'PARITY' in v or 'NO STAT' in v or 'DIFF' in v:
            short_v = "NO STAT. DIFF."
        else:
            short_v = "CONTRADICTED"

        text_str = f"{sign}{m:.2f} pp\n[{short_v}]"

        if m < -20:
            # AVSC configurations: place to the RIGHT of the error bar
            x_text = ch + 1.2
            ha = 'left'
        elif m < -5:
            # MSA configurations: place to the LEFT of the error bar
            x_text = cl - 1.2
            ha = 'right'
        else:
            # HEDO configuration: place to the RIGHT of the error bar
            x_text = ch + 1.2
            ha = 'left'

        ax.text(x_text, y_pos, text_str,
                va='center', ha=ha, fontsize=8.0, fontweight='bold', color=col,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.92, lw=0.9))

    ax.set_yticks(y)
    ax.set_yticklabels(comparisons, fontweight='bold', fontsize=9.5)
    ax.set_xlabel('Paired Delta in Mean Recall vs. Baseline (percentage points, 95% CI)', fontweight='bold')
    ax.set_title('Empirical Claim-Evidence Audit: Paired Deltas Across 3 Seeds\n(Flickr8k Cross-Modal Retrieval)', fontweight='bold', pad=14)

    ax.set_xlim(-40, 15)
    ax.set_ylim(-0.8, 7.4)

    # Category headers placed at y=6.8 with ample clearance
    ax.text(-31.0, 6.75, 'SEVERE DEGRADATION\n(AVSC Configurations)', color='#e41a1c', fontsize=8.2, fontweight='bold', ha='center', va='bottom')
    ax.text(-10.5, 6.75, 'MODERATE DEGRADATION\n(MSA Configurations)', color='#d95f02', fontsize=8.2, fontweight='bold', ha='center', va='bottom')
    ax.text(3.0, 6.75, 'SPANS ZERO\n(HEDO)', color='#2b7bba', fontsize=8.2, fontweight='bold', ha='center', va='bottom')

    ax.legend(loc='lower right', framealpha=0.95, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_paired_deltas.pdf")
    fig.savefig(OUTPUT_DIR / "fig_paired_deltas.png")
    plt.close(fig)
    print("Saved fig_paired_deltas.pdf and .png")


def plot_efficiency_tradeoff():
    """Figure 3: Efficiency vs. Accuracy Tradeoff Bubble Chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    configs = list(EFFICIENCY_DATA.keys())

    # Panel A: Head Throughput vs Mean Recall
    for c in configs:
        d = EFFICIENCY_DATA[c]
        mr = d['mean_recall']
        tp_head = d['head_throughput']
        tp_e2e = d['e2e_throughput']
        params = d['trainable_params'] / 1e3  # in thousands
        vram = d['peak_vram_mb']

        # Choose color based on category
        if c == 'baseline':
            color = '#333333'
            marker = 's'
        elif c == 'mamba2_hedo_v2':
            color = '#1f77b4'
            marker = '*'
        elif 'msa' in c and 'avsc' not in c:
            color = '#ff7f0e'
            marker = '^'
        else:
            color = '#d62728'
            marker = 'o'

        # Panel A: Head Throughput
        ax1.scatter(tp_head, mr, s=params * 0.4, color=color, alpha=0.75, edgecolors='black', linewidth=1.2, marker=marker)
        ax1.annotate(CONFIG_LABELS[c], (tp_head, mr), textcoords="offset points",
                     xytext=(6, 4) if c != 'baseline' else (-10, -15),
                     fontsize=8.5, fontweight='bold' if c in ['baseline', 'mamba2_hedo_v2'] else 'normal',
                     color=color)

        # Panel B: End-to-End Throughput vs Peak VRAM
        ax2.scatter(tp_e2e, vram, s=params * 0.4, color=color, alpha=0.75, edgecolors='black', linewidth=1.2, marker=marker)
        ax2.annotate(CONFIG_LABELS[c], (tp_e2e, vram), textcoords="offset points",
                     xytext=(6, 4) if c != 'mamba2_hedo_v2' else (-15, -14),
                     fontsize=8.5, fontweight='bold' if c in ['baseline', 'mamba2_hedo_v2'] else 'normal',
                     color=color)

    ax1.set_title('(a) Head Throughput vs. Retrieval Accuracy', fontweight='bold')
    ax1.set_xlabel('Head Throughput (pairs / sec)', fontweight='bold')
    ax1.set_ylabel('Mean Recall (%)', fontweight='bold')
    ax1.set_ylim(0, 42)
    ax1.set_xlim(5000, 10200)

    ax2.set_title('(b) End-to-End Throughput vs. Peak Memory Footprint', fontweight='bold')
    ax2.set_xlabel('End-to-End Throughput (pairs / sec)', fontweight='bold')
    ax2.set_ylabel('Peak Training VRAM (MB)', fontweight='bold')
    ax2.set_ylim(1030, 1105)
    ax2.set_xlim(3500, 5600)

    # Custom legend for bubble sizes and models
    patch_base = mpatches.Patch(color='#333333', label='Baseline')
    patch_hedo = mpatches.Patch(color='#1f77b4', label='Mamba-2 + HEDO')
    patch_msa = mpatches.Patch(color='#ff7f0e', label='MSA Configurations')
    patch_avsc = mpatches.Patch(color='#d62728', label='AVSC Configurations')

    ax1.legend(handles=[patch_base, patch_hedo, patch_msa, patch_avsc], loc='center left', framealpha=0.9, fontsize=8.5)

    fig.text(0.5, 0.01, 'Note: Bubble area represents trainable parameter count (465k to 764k parameters).', ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUTPUT_DIR / "fig_efficiency_tradeoff.pdf")
    fig.savefig(OUTPUT_DIR / "fig_efficiency_tradeoff.png")
    plt.close(fig)
    print("Saved fig_efficiency_tradeoff.pdf and .png")


def plot_seed_stability_metrics():
    """Figure 4: Seed stability across 3 seeds (42, 43, 44) and Median Rank distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    configs_ordered = [
        'baseline',
        'mamba2_hedo_v2',
        'mamba2_msa_v2',
        'hedo_msa_v2',
        'avsc_msa_v2',
        'mamba2_avsc_v2',
        'full_v2',
        'hedo_avsc_v2'
    ]
    labels = [CONFIG_LABELS[c] for c in configs_ordered]
    x = np.arange(len(configs_ordered))

    # Panel A: Per-seed Mean Recall
    seeds = [42, 43, 44]
    markers = ['o', 's', '^']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for s_idx, (seed, marker, col) in enumerate(zip(seeds, markers, colors)):
        mr_vals = [PER_SEED_RUNS[c]['mr'][s_idx] for c in configs_ordered]
        ax1.plot(x, mr_vals, marker=marker, linestyle='-', linewidth=1.2, markersize=7,
                 label=f'Seed {seed}', color=col, alpha=0.85)

    ax1.set_title('(a) Multi-Seed Run Consistency (Seeds 42, 43, 44)', fontweight='bold')
    ax1.set_ylabel('Mean Recall (%)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(0, 42)

    # Panel B: Median Rank (MedR - lower is better)
    i2t_medr = [ABLATION_DATA[c]['i2t_medr'][0] for c in configs_ordered]
    i2t_medr_err = [ABLATION_DATA[c]['i2t_medr'][1] for c in configs_ordered]
    t2i_medr = [ABLATION_DATA[c]['t2i_medr'][0] for c in configs_ordered]
    t2i_medr_err = [ABLATION_DATA[c]['t2i_medr'][1] for c in configs_ordered]

    width = 0.35
    ax2.bar(x - width/2, i2t_medr, width, yerr=i2t_medr_err, capsize=3, label='i2t MedR (Lower is better)', color='#4575b4', edgecolor='black', linewidth=0.6)
    ax2.bar(x + width/2, t2i_medr, width, yerr=t2i_medr_err, capsize=3, label='t2i MedR (Lower is better)', color='#d73027', edgecolor='black', linewidth=0.6)

    ax2.set_title('(b) Median Retrieval Rank (MedR)', fontweight='bold')
    ax2.set_ylabel('Median Rank (Tokens / Rank)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_yscale('log')
    ax2.set_ylim(1, 150)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_seed_stability_metrics.pdf")
    fig.savefig(OUTPUT_DIR / "fig_seed_stability_metrics.png")
    plt.close(fig)
    print("Saved fig_seed_stability_metrics.pdf and .png")


if __name__ == "__main__":
    print(f"Generating figures into {OUTPUT_DIR}...")
    plot_factorial_recall()
    plot_paired_deltas()
    plot_efficiency_tradeoff()
    plot_seed_stability_metrics()
    print("All 4 publication figures generated successfully!")
