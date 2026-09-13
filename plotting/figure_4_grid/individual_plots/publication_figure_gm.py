"""
Publication-quality boxplots for gait model comparison.
Generates 4 separate figures per model-gender:
  1. Pearson r improvement (delta)
  2. Actual Pearson r
  3. R² improvement (contribution)
  4. Actual R²
Uses FDR wilcox p-value threshold of 0.1.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
from publication_colors import get_system_colors, SYSTEM_RENAME_DICT, SYSTEM_COLOR_MAP, merge_systems


def load_filtered_data(csv_path, model_type, gender, score_type, fdr_threshold=0.1):
    """
    Load and filter data with FDR wilcox threshold.
    
    Args:
        csv_path: Path to comparisons CSV
        model_type: 'long_seq' or 'single_cycle'
        gender: 'male' or 'female'
        score_type: 'pearson_r' or 'r2'
        fdr_threshold: FDR-corrected wilcox p-value threshold (default 0.1)
    
    Returns:
        filtered_df, original_df (before significance filter)
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df = merge_systems(df)
    
    # Basic filters
    filtered = df[
        (df['sub_model'] == 'ensemble') &
        (df['model'] == model_type) &
        (df['gender'] == gender) &
        (df['score_type'] == score_type)
    ].copy()
    
    # Exclude systems and labels
    exclude_systems = ['wearable_weekly', 'wearable_monthly', 'subject', 'proteomics']
    exclude_labels = ['frailty_height', 'weight', 'height']
    filtered = filtered[~filtered['system'].isin(exclude_systems)]
    filtered = filtered[~filtered['label'].isin(exclude_labels)]
    
    original = filtered.copy()
    
    # Apply FDR wilcox significance filter
    # For R², skip score_pvalue filter (wilcox + delta > 0 is sufficient)
    if score_type == 'r2':
        filtered = filtered[
            (filtered['wilcox_pvalue_fdr'] < fdr_threshold) &
            (filtered['score'] > 0) &
            (filtered['delta'] > 0)
        ].copy()
    else:
        filtered = filtered[
            (filtered['wilcox_pvalue_fdr'] < fdr_threshold) &
            (filtered['score_pvalue'] < 0.05) &
            (filtered['score'] > 0) &
            (filtered['delta'] > 0)
        ].copy()
    
    # Calculate contribution (min of delta and score)
    filtered['contribution'] = filtered[['delta', 'score']].min(axis=1)
    
    return filtered, original


def get_system_order_and_counts(filtered_df, original_df, sort_col='contribution'):
    """Get systems ordered by median of sort_col, with sig/total counts."""
    system_stats = filtered_df.groupby('system')[sort_col].median().sort_values(ascending=False)
    systems_ordered = system_stats.index.tolist()
    
    sig_counts = filtered_df.groupby('system').size()
    total_counts = original_df.groupby('system').size()
    
    return systems_ordered, sig_counts, total_counts


def draw_boxplot_on_ax(ax, data, systems_ordered, color_dict, sig_counts, total_counts,
                       value_col, ylabel, ylim=None, show_xticklabels=True):
    """Draw a single boxplot on a given axis."""
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
        label_shift_points = 7
        label_shift = ScaledTranslation(
            label_shift_points / 72, 0, ax.figure.dpi_scale_trans
        )
        for label in ax.get_xticklabels():
            label.set_transform(label.get_transform() + label_shift)
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
    """Create and save a single boxplot figure as PNG and PDF."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.28, top=0.95)
    
    draw_boxplot_on_ax(ax, data, systems_ordered, color_dict, sig_counts, total_counts,
                       value_col, ylabel, ylim)
    
    plt.savefig(output_path, dpi=300)
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def load_gender_data(pearson_path, model_type, gender, fdr_threshold=0.1):
    """Load and prepare all Pearson data for one gender. Returns dict with all needed objects."""
    pearson_filt, pearson_orig = load_filtered_data(pearson_path, model_type, gender, 'pearson_r', fdr_threshold)
    systems, sig_counts, total_counts = get_system_order_and_counts(pearson_filt, pearson_orig, 'delta')
    colors = get_system_colors(systems)
    print(f"  {gender}: {len(pearson_filt)} significant / {len(pearson_orig)} total")
    return dict(filt=pearson_filt, orig=pearson_orig, systems=systems,
                sig_counts=sig_counts, total_counts=total_counts, colors=colors)


def generate_figures(pearson_path, r2_path, model_type, output_dir, fdr_threshold=0.1):
    """Generate individual figures + combined 4-row Pearson figure for one model type."""
    print(f"\n{model_type}:")
    
    # Load both genders
    male = load_gender_data(pearson_path, model_type, 'male', fdr_threshold)
    female = load_gender_data(pearson_path, model_type, 'female', fdr_threshold)

    # --- Individual figures (kept for convenience) ---
    for gender, gdata in [('male', male), ('female', female)]:
        prefix = f"{model_type}_{gender}"
        create_boxplot(
            gdata['filt'], gdata['systems'], gdata['colors'],
            gdata['sig_counts'], gdata['total_counts'],
            value_col='delta', ylabel='Pearson r improvement\n(Δ from baseline)',
            output_path=os.path.join(output_dir, f"{prefix}_pearson_delta.png"),
            ylim=(0, None)
        )
        create_boxplot(
            gdata['filt'], gdata['systems'], gdata['colors'],
            gdata['sig_counts'], gdata['total_counts'],
            value_col='score', ylabel='Pearson r',
            output_path=os.path.join(output_dir, f"{prefix}_pearson_score.png"),
            ylim=(0, 1.0)
        )

    # --- R² figures ---
    for gender in ['male', 'female']:
        r2_filt, r2_orig = load_filtered_data(r2_path, model_type, gender, 'r2', fdr_threshold)
        r2_systems, r2_sig, r2_total = get_system_order_and_counts(r2_filt, r2_orig, 'contribution')
        r2_colors = get_system_colors(r2_systems)
        prefix = f"{model_type}_{gender}"
        create_boxplot(
            r2_filt, r2_systems, r2_colors, r2_sig, r2_total,
            value_col='contribution', ylabel='R² contribution\n(beyond Age, BMI, Height & VAT)',
            output_path=os.path.join(output_dir, f"{prefix}_r2_contribution.png"),
            ylim=(0, 0.15)
        )
        create_boxplot(
            r2_filt, r2_systems, r2_colors, r2_sig, r2_total,
            value_col='score', ylabel='R²',
            output_path=os.path.join(output_dir, f"{prefix}_r2_score.png"),
            ylim=(0, 1.0)
        )

    # --- Combined 4-row Pearson figure ---
    # Uses GridSpec with 5 rows: 4 plots + 1 empty spacer between male/female pairs
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 26))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 0.35, 1, 1],
                  left=0.08, right=0.97, bottom=0.08, top=0.96, hspace=0.4)

    axes = [fig.add_subplot(gs[i]) for i in [0, 1, 3, 4]]

    rows = [
        (male, 'delta', 'Pearson r improvement\n(Δ from baseline)', (0, None), False),
        (male, 'score', 'Pearson r', (0, 1.0), True),
        (female, 'delta', 'Pearson r improvement\n(Δ from baseline)', (0, None), False),
        (female, 'score', 'Pearson r', (0, 1.0), True),
    ]

    for ax, (gdata, vcol, ylabel, ylim, show_xticks) in zip(axes, rows):
        draw_boxplot_on_ax(ax, gdata['filt'], gdata['systems'], gdata['colors'],
                           gdata['sig_counts'], gdata['total_counts'],
                           vcol, ylabel, ylim, show_xticklabels=show_xticks)

    axes[0].set_title('Male', fontsize=22, fontweight='bold', pad=14)
    axes[2].set_title('Female', fontsize=22, fontweight='bold', pad=14)

    combined_path = os.path.join(output_dir, f"{model_type}_pearson_combined.png")
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"Saved: {combined_path}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results')
    pearson_path = os.path.join(results_dir, 'gait_vs_confounders_pearson.csv')
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating long_seq Pearson figures (male + female)")
    print("=" * 60)

    for gender in ['male', 'female']:
        gdata = load_gender_data(pearson_path, 'long_seq', gender, fdr_threshold=0.1)
        prefix = f"long_seq_{gender}"
        create_boxplot(
            gdata['filt'], gdata['systems'], gdata['colors'],
            gdata['sig_counts'], gdata['total_counts'],
            value_col='delta', ylabel='Pearson r improvement\n(Δ from baseline)',
            output_path=os.path.join(output_dir, f"{prefix}_pearson_delta.png"),
            ylim=(0, None)
        )
        create_boxplot(
            gdata['filt'], gdata['systems'], gdata['colors'],
            gdata['sig_counts'], gdata['total_counts'],
            value_col='score', ylabel='Pearson r',
            output_path=os.path.join(output_dir, f"{prefix}_pearson_score.png"),
            ylim=(0, 1.0)
        )

    print("\n" + "=" * 60)
    print("All figures generated!")
    print("=" * 60)


if __name__ == '__main__':
    main()
