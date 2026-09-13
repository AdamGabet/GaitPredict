"""
Figure 2 c-h: Gait Fusion vs individual activities (and gait-features bars) for
Age, BMI, VAT in a 3x2 grid (rows = targets, cols = male/female).
Data loaded from pre-computed results/asbv_metrics.csv and results/movement_features_asbv.csv.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', '..', '..', 'results')
OUTPUT_DIR = HERE  # panels live alongside this script (read by the composite)

FIXED_YLIM = True

ACTIVITY_NAMES = {
    'activity_tm_3kmh': 'Treadmill Walk\n(3 km/h)',
    'activity_self_selected_gait_speed': 'Treadmill Walk\n(Self-Paced)',
    'activity_stationary_walk': 'Stationary\nWalking',
    'activity_sit_to_stand': 'Sit-to-Stand',
    'activity_romberg_closed': "Romberg's\nTest",
}

ACTIVITY_ORDER = [
    'activity_tm_3kmh',
    'activity_self_selected_gait_speed',
    'activity_stationary_walk',
    'activity_sit_to_stand',
    'activity_romberg_closed',
]

ACTIVITY_TO_MOVEMENT_MODEL = {
    'activity_tm_3kmh': 'tm_3kmh',
    'activity_self_selected_gait_speed': 'self_selected_gait_speed',
    'activity_stationary_walk': 'stationary',
    'activity_sit_to_stand': 'sit_to_stand',
    'activity_romberg_closed': 'rombergs_closed',
}

# short name -> y-axis label; long name -> panel title (matches source)
TARGETS = {
    'age': {'short': 'Age', 'long': 'Age'},
    'bmi': {'short': 'BMI', 'long': 'Body Mass Index'},
    'total_scan_vat_area': {'short': 'VAT Area', 'long': 'Visceral Adipose Tissue Area'},
}


def load_data():
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'asbv_metrics.csv'))
    movement = pd.read_csv(os.path.join(RESULTS_DIR, 'movement_features_asbv.csv'))
    seeds = pd.read_csv(os.path.join(RESULTS_DIR, 'asbv_seed_values.csv'))
    return metrics, movement, seeds


def get_seed_vals(seeds, target, activity, gender, source):
    """Per-seed values behind one bar (Nature data-distribution policy).

    NOTE the vocabulary difference vs asbv_metrics.csv: there the movement-features
    ensemble is stored as activity='gait', whereas here it is
    source='features', activity='ensemble'.
    """
    sel = seeds[(seeds['target'] == target) & (seeds['activity'] == activity) &
                (seeds['gender'] == gender) & (seeds['source'] == source)]
    if sel.empty:
        return None
    return sel['value'].to_numpy(dtype=float)


def get_val(metrics, target, activity, gender):
    row = metrics[(metrics['target'] == target) & (metrics['activity'] == activity) &
                  (metrics['gender'] == gender)]
    if row.empty:
        return None, None
    return float(row['mean'].iloc[0]), float(row['std'].iloc[0])


def get_movement_val(movement, target, activity_key, gender):
    model_name = ACTIVITY_TO_MOVEMENT_MODEL.get(activity_key)
    if model_name is None:
        return None
    row = movement[(movement['label'] == target) & (movement['model'] == model_name) &
                   (movement['gender'] == gender)]
    if row.empty:
        return None
    return float(row['score'].iloc[0])


def plot_grid_2x3(metrics, movement, seeds):
    # Font comes from the repo-root matplotlibrc (Times New Roman).
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    targets = ['age', 'bmi', 'total_scan_vat_area']
    subsets = ['male', 'female']

    for row_idx, target in enumerate(targets):
        for col_idx, subset in enumerate(subsets):
            ax = axes[row_idx, col_idx]

            labels, emb_means, emb_stds, mov_means = [], [], [], []
            emb_seed_lists, mov_seed_lists, mov_stds = [], [], []

            # Gait Fusion column: embedding ensemble + movement-features ensemble ('gait')
            ens_mean, ens_std = get_val(metrics, target, 'ensemble', subset)
            gait_mean, _ = get_val(metrics, target, 'gait', subset)
            labels.append('Gait Fusion')
            emb_means.append(ens_mean)
            emb_stds.append(ens_std)
            mov_means.append(gait_mean)
            emb_seed_lists.append(get_seed_vals(seeds, target, 'ensemble', subset, 'embeddings'))
            fusion_mov_seeds = get_seed_vals(seeds, target, 'ensemble', subset, 'features')
            mov_seed_lists.append(fusion_mov_seeds)
            mov_stds.append(np.std(fusion_mov_seeds, ddof=1) if fusion_mov_seeds is not None else 0.0)

            # Per-activity columns
            for act in ACTIVITY_ORDER:
                m, s = get_val(metrics, target, act, subset)
                if m is None:
                    continue
                mv = get_movement_val(movement, target, act, subset)
                labels.append(ACTIVITY_NAMES[act])
                emb_means.append(m)
                emb_stds.append(s)
                mov_means.append(mv if mv is not None else np.nan)
                emb_seed_lists.append(get_seed_vals(seeds, target, act, subset, 'embeddings'))
                act_mov_seeds = get_seed_vals(seeds, target, act, subset, 'features')
                mov_seed_lists.append(act_mov_seeds)
                mov_stds.append(np.std(act_mov_seeds, ddof=1) if act_mov_seeds is not None else 0.0)

            x = np.arange(len(labels))
            width = 0.35

            emb_color = '#D35400' if subset == 'female' else '#1F618D'
            mov_color = '#F5B041' if subset == 'female' else '#5DADE2'
            err_kw = dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor='#333333', zorder=5)

            ax.bar(x - width/2, emb_means, width, label='Gait Embeddings',
                   color=emb_color, zorder=3, yerr=emb_stds, error_kw=err_kw)
            ax.bar(x + width/2, mov_means, width, label='Gait Features',
                   color=mov_color, zorder=3, yerr=mov_stds, error_kw=err_kw)

            # Nature policy: show the underlying data distribution, not just
            # bar + error bar. One point per cross-validation seed.
            for xc, vals in zip(x - width/2, emb_seed_lists):
                if vals is None or len(vals) == 0:
                    continue
                jitter = np.linspace(-width * 0.17, width * 0.17, len(vals))
                ax.scatter(xc + jitter, vals, s=16, color='black', alpha=0.55,
                           linewidths=0.4, edgecolors='white', zorder=6)
            for xc, vals in zip(x + width/2, mov_seed_lists):
                if vals is None or len(vals) == 0:
                    continue
                jitter = np.linspace(-width * 0.17, width * 0.17, len(vals))
                ax.scatter(xc + jitter, vals, s=16, color='black', alpha=0.55,
                           linewidths=0.4, edgecolors='white', zorder=6)

            ax.set_ylabel(f'{TARGETS[target]["short"]} Prediction Power ($R$)',
                          fontsize=16, labelpad=10)
            if FIXED_YLIM:
                ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='-', alpha=0.3, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=20, rotation=30, ha='right')
            ax.tick_params(axis='y', labelsize=14)
            ax.tick_params(axis='x', pad=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.0, 1.03))
            # source uses subset_str=' (Male)' appended after a space -> double space
            ax.set_title(f'{TARGETS[target]["long"]}  ({subset.capitalize()})',
                         fontsize=24, fontweight='bold', pad=20)

    plt.tight_layout(pad=3.0)
    stem = 'grid_2x3_long_seq_with_movement_data'
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'{stem}.{ext}'), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {stem}")


def main():
    metrics, movement, seeds = load_data()
    plot_grid_2x3(metrics, movement, seeds)


if __name__ == '__main__':
    main()
