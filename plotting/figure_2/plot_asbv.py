"""
Plot Multi-Task DL Fusion vs Individual Activities for Age, BMI, VAT
with side-by-side comparison to movement_data_features results.
Data loaded from pre-computed CSVs in results/.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

TARGETS = {
    'age': {'display_name': 'Age'},
    'bmi': {'display_name': 'Body Mass Index'},
    'total_scan_vat_area': {'display_name': 'Visceral Adipose Tissue Area'},
}


def load_data():
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, 'asbv_metrics.csv'))
    movement = pd.read_csv(os.path.join(RESULTS_DIR, 'movement_features_asbv.csv'))
    return metrics, movement


def get_val(metrics, target, activity, gender):
    row = metrics[(metrics['target'] == target) & (metrics['activity'] == activity) & (metrics['gender'] == gender)]
    if row.empty:
        return None, None
    return float(row['mean'].iloc[0]), float(row['std'].iloc[0])


def get_movement_val(movement, target, activity_key, gender):
    model_name = ACTIVITY_TO_MOVEMENT_MODEL.get(activity_key)
    if model_name is None:
        return None
    row = movement[(movement['label'] == target) & (movement['model'] == model_name) & (movement['gender'] == gender)]
    if row.empty:
        return None
    return float(row['score'].iloc[0])


def plot_grid_2x3(metrics, movement):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    fig, axes = plt.subplots(3, 2, figsize=(20, 20))
    targets = ['age', 'bmi', 'total_scan_vat_area']
    subsets = ['male', 'female']

    for row_idx, target in enumerate(targets):
        for col_idx, subset in enumerate(subsets):
            ax = axes[row_idx, col_idx]

            labels = ['Gait Fusion']
            emb_means, emb_stds, mov_means = [], [], []

            ens_mean, ens_std = get_val(metrics, target, 'ensemble', subset)
            gait_mean, _ = get_val(metrics, target, 'gait', subset)
            labels.append('Gait Fusion')
            emb_means.append(ens_mean)
            emb_stds.append(ens_std)
            mov_means.append(gait_mean)

            # Remove the duplicate first entry
            labels, emb_means, emb_stds, mov_means = [], [], [], []
            ens_mean, ens_std = get_val(metrics, target, 'ensemble', subset)
            gait_mean, _ = get_val(metrics, target, 'gait', subset)
            labels.append('Gait Fusion')
            emb_means.append(ens_mean)
            emb_stds.append(ens_std)
            mov_means.append(gait_mean)

            for act in ACTIVITY_ORDER:
                m, s = get_val(metrics, target, act, subset)
                if m is None:
                    continue
                mv = get_movement_val(movement, target, act, subset)
                labels.append(ACTIVITY_NAMES[act])
                emb_means.append(m)
                emb_stds.append(s)
                mov_means.append(mv if mv is not None else np.nan)

            x = np.arange(len(labels))
            width = 0.35

            emb_color = '#D35400' if subset == 'female' else '#1F618D'
            mov_color = '#F5B041' if subset == 'female' else '#5DADE2'
            err_kw = dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor='#333333', zorder=5)

            ax.bar(x - width/2, emb_means, width, label='Gait Embeddings',
                   color=emb_color, zorder=3, yerr=emb_stds, error_kw=err_kw)
            ax.bar(x + width/2, mov_means, width, label='Gait Features',
                   color=mov_color, zorder=3)

            ax.set_ylabel(f'{TARGETS[target]["display_name"]} Prediction Power ($R$)', fontsize=16, labelpad=10)
            if FIXED_YLIM:
                ax.set_ylim(0, 1)
            ax.grid(axis='y', linestyle='-', alpha=0.3, zorder=0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=20, rotation=30, ha='right')
            ax.tick_params(axis='y', labelsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right', fontsize=14, bbox_to_anchor=(1.0, 1.03))
            ax.set_title(f'{TARGETS[target]["display_name"]} ({subset.capitalize()})',
                         fontsize=24, fontweight='bold', pad=20)

    plt.tight_layout(pad=3.0)
    stem = 'grid_2x3_long_seq_with_movement_data'
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'{stem}.{ext}'), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {stem}")


def main():
    metrics, movement = load_data()
    plot_grid_2x3(metrics, movement)


if __name__ == '__main__':
    main()
