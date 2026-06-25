"""
Figure 2b: UMAP visualization of gait embeddings colored by activity type.
Loads pre-computed UMAP coordinates from results/umap_embeddings.csv (deterministic;
UMAP itself is recomputed only by the source BUILD script because n_jobs=-1 is stochastic).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', '..', '..', 'results')
OUTPUT_DIR = HERE  # panels live alongside this script (read by the composite)

# AUC from one-vs-all logistic regression (synced to source umap_activity_embeddings.py)
CLASSIFICATION_AUC = {
    'romberg_closed': 1.0000,
    'self_selected_gait_speed': 0.9633,
    'sit_to_stand': 1.0000,
    'stationary_walk': 0.9997,
    'tm_3kmh': 0.9651,
}

ACTIVITY_COLORS = {
    'tm_3kmh': '#27ae60',
    'self_selected_gait_speed': '#f1c40f',
    'romberg_closed': '#e74c3c',
    'sit_to_stand': '#9b59b6',
    'stationary_walk': '#3498db',
}

ACTIVITY_NAMES = {
    'tm_3kmh': 'Treadmill 3km/h',
    'self_selected_gait_speed': 'Self-Selected Speed',
    'romberg_closed': 'Romberg Closed',
    'sit_to_stand': 'Sit to Stand',
    'stationary_walk': 'Stationary Walking',
}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'umap_embeddings.csv'))
    print(f"Loaded {len(df)} samples")
    print(df['activity'].value_counts())

    unique_activities = sorted(
        df['activity'].unique(),
        key=lambda x: CLASSIFICATION_AUC.get(x, 1.0)
    )

    # --- Main plot (no legend) ---
    fig, ax = plt.subplots(figsize=(12, 10))
    for activity in unique_activities:
        mask = df['activity'] == activity
        color = ACTIVITY_COLORS.get(activity, '#7f8c8d')
        ax.scatter(df.loc[mask, 'umap_1'], df.loc[mask, 'umap_2'],
                   c=color, alpha=0.6, s=15, edgecolors='none')

    ax.set_xlabel('UMAP 1', fontsize=20)
    ax.set_ylabel('UMAP 2', fontsize=20)
    ax.set_title('Gait Embeddings by Activity (UMAP)', fontsize=28, fontweight='bold', pad=25)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'umap_by_activity.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # --- Standalone legend (emitted here; composite reads umap_by_activity_legend.png) ---
    fig_legend, ax_legend = plt.subplots(figsize=(6, 4))
    ax_legend.axis('off')
    handles = []
    for activity in unique_activities:
        color = ACTIVITY_COLORS.get(activity, '#7f8c8d')
        auc = CLASSIFICATION_AUC.get(activity)
        name = ACTIVITY_NAMES.get(activity, activity.replace('_', ' ').title())
        label = f"{name} (AUC={auc:.3f})" if auc else name
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                                   markersize=20, label=label, linestyle='None'))
    legend = ax_legend.legend(handles=handles, loc='center', framealpha=0.9, fontsize=22,
                               title='Activity (one-vs-all AUC)')
    legend.get_title().set_fontsize(24)
    legend.get_title().set_fontweight('bold')
    legend_path = os.path.join(OUTPUT_DIR, 'umap_by_activity_legend.png')
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.savefig(legend_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {legend_path}")


if __name__ == '__main__':
    main()
