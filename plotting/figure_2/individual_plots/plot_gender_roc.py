"""
Figure 2a: ROC curves for gender prediction across all activities.
Data loaded from pre-computed results/roc_gender.csv (curve points + AUC, no subject data).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', '..', '..', 'results')
OUTPUT_DIR = HERE  # panels live alongside this script (read by the composite)

ACTIVITY_NAMES = {
    'activity_tm_3kmh': 'Treadmill Walk (3 km/h)',
    'activity_self_selected_gait_speed': 'Treadmill Walk (Self-Paced)',
    'activity_stationary_walk': 'Stationary Walking',
    'activity_sit_to_stand': 'Sit-to-Stand',
    'activity_romberg_closed': "Romberg's Test",
    'ensemble': 'Gait Fusion',
    'gait': 'Gait Features Baseline',
}

ACTIVITY_COLORS = {
    'activity_tm_3kmh': '#27ae60',
    'activity_self_selected_gait_speed': '#f1c40f',
    'activity_romberg_closed': '#e74c3c',
    'activity_sit_to_stand': '#9b59b6',
    'activity_stationary_walk': '#3498db',
    'ensemble': '#ed80e9',
    'gait': '#7f8c8d',
}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'roc_gender.csv'))

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    fig, ax = plt.subplots(figsize=(9, 8))

    # Sort legend by AUC descending (matches source)
    auc_vals = df.groupby('activity')['auc'].first().sort_values(ascending=False)

    for activity in auc_vals.index:
        # Stable monotone curve order: sort by fpr then tpr (handles vertical segments)
        act_df = df[df['activity'] == activity].sort_values(['fpr', 'tpr'])
        roc_auc = auc_vals[activity]
        name = ACTIVITY_NAMES.get(activity, activity)
        label = f'{name} ({roc_auc:.3f})'
        color = ACTIVITY_COLORS.get(activity, 'black')

        if activity == 'gait':
            ax.plot(act_df['fpr'], act_df['tpr'], color=color, linewidth=2.5,
                    linestyle='--', label=label, zorder=9)
        elif activity == 'ensemble':
            ax.plot(act_df['fpr'], act_df['tpr'], color=color, linewidth=3,
                    label=label, zorder=10)
        else:
            ax.plot(act_df['fpr'], act_df['tpr'], color=color, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=18, labelpad=10)
    ax.set_ylabel('True Positive Rate', fontsize=18, labelpad=10)
    ax.set_title('Gender Prediction', fontsize=22, fontweight='bold', pad=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(loc='lower right', fontsize=16, framealpha=0.95)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle='-', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    stem = 'gender_roc_curves_long_seq'
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'{stem}.{ext}'), dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {stem}")


if __name__ == '__main__':
    main()
