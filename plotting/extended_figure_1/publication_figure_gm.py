"""
Publication-quality boxplots for the 'gait' model (gait features baseline).
Generates Pearson r delta and score plots for male and female.
Uses FDR wilcox p-value threshold of 0.1.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from publication_colors import get_system_colors, SYSTEM_RENAME_DICT, merge_systems

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PEARSON_CSV = os.path.join(RESULTS_DIR, 'gait_vs_confounders_pearson.csv')
FDR_THRESHOLD = 0.1


def load_filtered_data(model_type, gender, fdr_threshold=FDR_THRESHOLD):
    df = pd.read_csv(PEARSON_CSV, low_memory=False)
    df = merge_systems(df)

    sub_model = 'flat' if model_type == 'gait' else 'ensemble'

    filtered = df[
        (df['sub_model'] == sub_model) &
        (df['model'] == model_type) &
        (df['gender'] == gender) &
        (df['score_type'] == 'pearson_r')
    ].copy()

    exclude_systems = ['wearable_weekly', 'wearable_monthly', 'subject', 'proteomics']
    exclude_labels = ['frailty_height', 'weight', 'height']
    filtered = filtered[~filtered['system'].isin(exclude_systems)]
    filtered = filtered[~filtered['label'].isin(exclude_labels)]

    original = filtered.copy()

    filtered = filtered[
        (filtered['wilcox_pvalue_fdr'] < fdr_threshold) &
        (filtered['score_pvalue'] < 0.05) &
        (filtered['score'] > 0) &
        (filtered['delta'] > 0)
    ].copy()

    return filtered, original


def get_system_order_and_counts(filtered_df, original_df):
    system_stats = filtered_df.groupby('system')['delta'].median().sort_values(ascending=False)
    systems_ordered = system_stats.index.tolist()
    sig_counts = filtered_df.groupby('system').size()
    total_counts = original_df.groupby('system').size()
    return systems_ordered, sig_counts, total_counts


def draw_boxplot_on_ax(ax, data, systems_ordered, color_dict, sig_counts, total_counts,
                       value_col, ylabel, ylim=None, show_xticklabels=True):
    boxplot_data = [data[data['system'] == sys][value_col].values for sys in systems_ordered]
    system_labels = [SYSTEM_RENAME_DICT.get(sys, sys.replace('_', ' ').title()) for sys in systems_ordered]

    ax.boxplot(boxplot_data, patch_artist=True, showfliers=False, widths=0.5,
               boxprops=dict(linewidth=1.0, edgecolor='black', facecolor='white'),
               whiskerprops=dict(linewidth=1.0, color='black'),
               capprops=dict(linewidth=1.0, color='black'),
               medianprops=dict(linewidth=1.5, color='black'))

    for i, system in enumerate(systems_ordered):
        y_vals = data[data['system'] == system][value_col].values
        x_vals = np.random.normal(i + 1, 0.03, size=len(y_vals))
        ax.scatter(x_vals, y_vals, alpha=0.7, s=20, color=color_dict.get(system, 'gray'),
                   zorder=3, edgecolors='none')

    tick_labels = []
    for sys, name in zip(systems_ordered, system_labels):
        n_sig = int(sig_counts.get(sys, 0))
        n_total = int(total_counts.get(sys, 0))
        tick_labels.append(f"{name}\n({n_sig}/{n_total})")

    ax.set_xticks(range(1, len(systems_ordered) + 1))
    if show_xticklabels:
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=15, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_boxplot(data, systems_ordered, color_dict, sig_counts, total_counts,
                   value_col, ylabel, output_path, ylim=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.28, top=0.95)
    draw_boxplot_on_ax(ax, data, systems_ordered, color_dict, sig_counts, total_counts,
                       value_col, ylabel, ylim)
    for ext in ['png', 'pdf']:
        plt.savefig(output_path.replace('.png', f'.{ext}'), dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def load_gender_data(model_type, gender):
    filt, orig = load_filtered_data(model_type, gender)
    systems, sig_counts, total_counts = get_system_order_and_counts(filt, orig)
    colors = get_system_colors(systems)
    print(f"  {gender}: {len(filt)} significant / {len(orig)} total")
    return dict(filt=filt, orig=orig, systems=systems,
                sig_counts=sig_counts, total_counts=total_counts, colors=colors)


def main():
    model_type = 'gait'

    for gender in ['male', 'female']:
        gdata = load_gender_data(model_type, gender)
        prefix = f"gait_{gender}"

        create_boxplot(
            gdata['filt'], gdata['systems'], gdata['colors'],
            gdata['sig_counts'], gdata['total_counts'],
            value_col='delta', ylabel='Pearson r improvement\n(Δ from baseline)',
            output_path=os.path.join(OUTPUT_DIR, f"{prefix}_pearson_delta.png"),
            ylim=(0, None)
        )
        create_boxplot(
            gdata['filt'], gdata['systems'], gdata['colors'],
            gdata['sig_counts'], gdata['total_counts'],
            value_col='score', ylabel='Pearson r',
            output_path=os.path.join(OUTPUT_DIR, f"{prefix}_pearson_score.png"),
            ylim=(0, 1.0)
        )

    print("Done.")


if __name__ == '__main__':
    main()
