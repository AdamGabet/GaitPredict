"""
Recreate Figure 4 as a single grid:
  Left column  (a, c): 4 stacked boxplots
                       a = Male  (Pearson r improvement Δ, then Pearson r)
                       c = Female(Pearson r improvement Δ, then Pearson r)
  Right column (b, d): 2 horizontal barplots (top biomarkers per system)
                       b = Male, d = Female

Reuses the (already-correct) data/filter functions from figure_4 and
figure_4b; only the drawing is composited onto a shared GridSpec.
"""
import os
import sys

_HERE = os.path.dirname(__file__)
# publication_colors lives in plotting/; the two component modules live in individual_plots/.
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(0, os.path.join(_HERE, 'individual_plots'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Data + drawing helpers (filters live here — do not reimplement).
# Import the figure_4b module FIRST: publication_figure_gm prepends
# plotting/downstream_results to sys.path on import, which would otherwise
# shadow figure_4b's combined_pearson_barplot with a same-named file there.
from combined_pearson_barplot import (
    get_unified_system_order, get_top_biomarkers_combined,
    load_seed_values, pretty_label, LABEL_DISPLAY_OVERRIDES,
)

# Shorter display names for this grid figure (don't touch the shared module).
LABEL_DISPLAY_OVERRIDES['totalpolyunsaturatedfattyacids_g'] = 'Total PUFA (g)'
from publication_figure_gm import load_gender_data, draw_boxplot_on_ax
from publication_colors import SYSTEM_COLOR_MAP


def draw_barplot_on_ax(ax, bio_df, model_type, gender, base_path,
                       label_fontsize=17, xlabel_fontsize=19, add_legend=False):
    """Port of create_combined_barplot's drawing body onto a provided ax."""
    bar_h = 0.26
    gap_within = 0.34      # spacing between the two biomarkers in a system pair
    system_gap = 0.60

    # y positions (top to bottom; inverted later)
    y_centers = []
    y = 0
    prev_system = None
    for _, row in bio_df.iterrows():
        if prev_system is not None and row['system'] != prev_system:
            y += system_gap
        y_centers.append(y)
        y += bar_h * 2 + gap_within
        prev_system = row['system']

    for i, (_, row) in enumerate(bio_df.iterrows()):
        system = row['system']
        yc = y_centers[i]

        base_color = np.array(SYSTEM_COLOR_MAP.get(system, (0.5, 0.5, 0.5, 1.0))[:3])
        light_color = tuple(base_color * 0.5 + 0.5) + (0.7,)
        dark_color = tuple(base_color) + (0.9,)

        baseline_seeds, gait_seeds = load_seed_values(
            base_path, system, row['label'], model_type, gender
        )

        y_base = yc
        y_gait = yc + bar_h + 0.03

        ax.barh(y_base, row['baseline_score'], height=bar_h,
                color=light_color, edgecolor='none', zorder=2)
        ax.barh(y_gait, row['score'], height=bar_h,
                color=dark_color, edgecolor='none', zorder=2)

        if baseline_seeds is not None:
            jitter = np.linspace(-bar_h * 0.2, bar_h * 0.2, len(baseline_seeds))
            ax.scatter(baseline_seeds, y_base + jitter,
                       color='black', s=2.0, alpha=0.4, zorder=5)
        if gait_seeds is not None:
            jitter = np.linspace(-bar_h * 0.2, bar_h * 0.2, len(gait_seeds))
            ax.scatter(gait_seeds, y_gait + jitter,
                       color='black', s=2.0, alpha=0.4, zorder=5)

    label_y = [yc + bar_h / 2 + 0.01 for yc in y_centers]
    labels = [pretty_label(row['label']) for _, row in bio_df.iterrows()]
    ax.set_yticks(label_y)
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', labelsize=xlabel_fontsize - 2)
    ax.set_xlim(left=0)
    ax.set_xlabel('Pearson r', fontsize=xlabel_fontsize, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if add_legend:
        legend_patches = [
            mpatches.Patch(facecolor='#b0b0b0', alpha=0.7, label='Age, BMI, VAT & Height'),
            mpatches.Patch(facecolor='#606060', alpha=0.9, label='Gait, Age, BMI, VAT & Height'),
        ]
        ax.legend(handles=legend_patches, loc='lower right',
                  bbox_to_anchor=(1.0, 1.005), fontsize=14,
                  framealpha=0.9, ncol=1, handlelength=1.2)


def main():
    pearson_path = os.path.join(_HERE, '..', '..', 'results', 'gait_vs_confounders_pearson.csv')
    base_path = None  # seed values now come from results/figure4_grid_seed_values.csv
    output_dir = os.path.join(_HERE, 'output')
    os.makedirs(output_dir, exist_ok=True)

    model_type = 'long_seq'

    # --- Left column data (boxplots) ---
    male = load_gender_data(pearson_path, model_type, 'male', fdr_threshold=0.1)
    female = load_gender_data(pearson_path, model_type, 'female', fdr_threshold=0.1)

    # --- Right column data (barplots) ---
    unified_order = get_unified_system_order(pearson_path, model_type)
    bio_male, _ = get_top_biomarkers_combined(
        pearson_path, model_type, 'male', top_n=2, candidate_top_k=5,
        use_manual_selection=True, fixed_system_order=unified_order)
    bio_female, _ = get_top_biomarkers_combined(
        pearson_path, model_type, 'female', top_n=2, candidate_top_k=5,
        use_manual_selection=True, fixed_system_order=unified_order)

    # --- Figure layout ---
    fig = plt.figure(figsize=(22, 26))
    gs = GridSpec(5, 2, figure=fig,
                  height_ratios=[1, 1, 0.18, 1, 1],
                  width_ratios=[1.65, 1],
                  left=0.06, right=0.99, bottom=0.115, top=0.965,
                  hspace=0.50, wspace=0.32)

    # Left: 4 boxplots
    ax_md = fig.add_subplot(gs[0, 0])
    ax_ms = fig.add_subplot(gs[1, 0])
    ax_fd = fig.add_subplot(gs[3, 0])
    ax_fs = fig.add_subplot(gs[4, 0])

    box_rows = [
        (ax_md, male, 'delta', 'Pearson r improvement\n(Δ from baseline)', (0, None), True),
        (ax_ms, male, 'score', 'Pearson r', (0, 1.0), True),
        (ax_fd, female, 'delta', 'Pearson r improvement\n(Δ from baseline)', (0, None), True),
        (ax_fs, female, 'score', 'Pearson r', (0, 1.0), True),
    ]
    for ax, gdata, vcol, ylabel, ylim, show_xticks in box_rows:
        draw_boxplot_on_ax(ax, gdata['filt'], gdata['systems'], gdata['colors'],
                           gdata['sig_counts'], gdata['total_counts'],
                           vcol, ylabel, ylim, show_xticklabels=show_xticks)
        # Enlarge fonts (draw_boxplot_on_ax hardcodes smaller sizes).
        ax.yaxis.label.set_size(18)
        ax.tick_params(axis='y', labelsize=15)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(16)

    ax_md.set_title('Male', fontsize=28, fontweight='bold', pad=12)
    ax_fd.set_title('Female', fontsize=28, fontweight='bold', pad=12)

    # Right: 2 barplots spanning each gender's pair of boxplots
    ax_bm = fig.add_subplot(gs[0:2, 1])
    ax_bf = fig.add_subplot(gs[3:5, 1])
    draw_barplot_on_ax(ax_bm, bio_male, model_type, 'male', base_path, add_legend=True)
    draw_barplot_on_ax(ax_bf, bio_female, model_type, 'female', base_path, add_legend=False)

    # Panel letters — anchored in figure coords so b/d sit at the same level
    # as a/c (the barplot axes span two rows, so transAxes would misalign them).
    fig.canvas.draw()
    y_top = ax_md.get_position().y1 + 0.004   # male-row letter level (a, b)
    y_bot = ax_fd.get_position().y1 + 0.004   # female-row letter level (c, d)
    letters = [
        (ax_md, 'a', y_top), (ax_bm, 'b', y_top),
        (ax_fd, 'c', y_bot), (ax_bf, 'd', y_bot),
    ]
    for ax, letter, y in letters:
        x = ax.get_position().x0 - 0.012
        fig.text(x, y, letter, fontsize=28, fontweight='bold',
                 va='bottom', ha='right')

    base_fname = f'{model_type}_figure4_grid'
    for ext in ['png', 'pdf']:
        out_path = os.path.join(output_dir, f'{base_fname}.{ext}')
        plt.savefig(out_path, dpi=200, facecolor='white', edgecolor='white', format=ext)
        print(f"Saved: {out_path}")
    plt.close()


if __name__ == '__main__':
    main()
