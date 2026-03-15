"""
Combined horizontal bar plot: top biomarkers per system in a single figure.
Lighter bar = baseline (Age, BMI, VAT, Height), darker bar = full model (+ Gait).
Std error bars overlaid on each bar (from pre-computed figure4b_seed_std.csv).
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from publication_colors import (
    LABEL_RENAME_DICT, SYSTEM_COLOR_MAP, LIFESTYLE_DUPLICATE_MAP, merge_systems
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PEARSON_CSV = os.path.join(RESULTS_DIR, 'gait_vs_confounders_pearson.csv')
STD_CSV = os.path.join(RESULTS_DIR, 'figure4b_seed_std.csv')

MANUAL_LABEL_SELECTION = {
    'male': {
        'blood_tests_lipids': ['bt__triglycerides', 'bt__hdl_cholesterol'],
        'bone_density': ['spine_l1_l4_area', 'spine_l1_l3_bmc'],
        'cardiovascular_system': ['hr_bpm', 'automorph_vein_fractal_dimension'],
        'frailty': ['hand_grip_left', 'hand_grip_right'],
        'glycemic_status': [],
        'hematopoietic': ['bt__hct'],
        'immune_system': [],
        'lifestyle': ['vigorous_activity_days', 'sleep_hours'],
        'liver': ['liver_sound_speed', 'liver_viscosity'],
        'mental': ['happiness_level', 'depressed_longest_period_weeks'],
        'metabolites': ['LysoPE(P-16_0_0_0)', 'C17 Oxylipin (291.16 m_z)'],
        'microbiome': ['Rep_3257', 'Rep_2163'],
        'nightingale': ['XL_HDL_PL', 'GlycA'],
        'proteomics': [],
        'renal_function': ['bt__creatinine'],
        'sleep_group': ['ahi', 'hypoxic_burden'],
    },
    'female': {
        'blood_tests_lipids': ['bt__total_cholesterol', 'bt__non_hdl_cholesterol'],
        'bone_density': ['spine_l1_l4_area', 'spine_l1_l3_bmc'],
        'cardiovascular_system': ['automorph_artery_fractal_dimension', 'intima_media_th_2_fit'],
        'frailty': ['hand_grip_left', 'hand_grip_right'],
        'glycemic_status': [],
        'hematopoietic': ['bt__ferritin', 'bt__mcv'],
        'immune_system': [],
        'lifestyle': ['vigorous_activity_minutes', 'walking_minutes_day'],
        'liver': ['liver_sound_speed', 'bt__alt_gpt'],
        'mental': ['happiness_level', 'health_satisfaction'],
        'metabolites': ['LysoPE(P-16_0_0_0)', 'C17 Oxylipin (291.16 m_z)'],
        'microbiome': ['Rep_2101', 'Rep_1819'],
        'nightingale': ['L_VLDL_FC', 'XL_HDL_PL'],
        'proteomics': [],
        'sleep_group': ['sleep_efficiency', 'saturation_mean'],
    },
}

LABEL_DISPLAY_OVERRIDES = {
    'automorph_vein_fractal_dimension': 'Vein Fractal Dimension',
    'spine_l1_l3_bmc': 'Spine L1-L3 BMC',
    'hand_grip_left': 'L Hand Grip',
    'hand_grip_right': 'R Hand Grip',
    'intima_media_th_2_fit': 'Carotid IMT',
    'bt__mcv': 'MCV',
    'depressed_longest_period_weeks': 'Depressive Episode Length',
    'LysoPE(P-16_0_0_0)': 'LysoPE (P-16:0)',
    'C17 Oxylipin (291.16 m_z)': 'C17 Oxylipin',
    'Rep_3257': 'Microbiome Feature 3257',
    'Rep_2163': 'Microbiome Feature 2163',
    'Rep_2101': 'Microbiome Feature 2101',
    'Rep_1819': 'Microbiome Feature 1819',
    'GlycA': 'GlycA Inflammation Signal',
    'XL_HDL_PL': 'XL HDL Phospholipids',
    'L_VLDL_FC': 'L VLDL Free Cholesterol',
    'spine_l1_l4_area': 'Spine L1-L4 Area',
    'sleep_hours': 'Sleep Hours in 24H',
}


def _build_lifestyle_label_candidates(label):
    candidates = [label]
    reverse_map = {}
    for src_label, canonical_label in LIFESTYLE_DUPLICATE_MAP.items():
        reverse_map.setdefault(canonical_label, []).append(src_label)
    if label in reverse_map:
        candidates.extend(reverse_map[label])
    if label in LIFESTYLE_DUPLICATE_MAP:
        candidates.append(LIFESTYLE_DUPLICATE_MAP[label])
    return list(dict.fromkeys(candidates))


def get_unified_system_order(df, model_type, fdr_threshold=0.1):
    df = merge_systems(df)
    exclude_systems = ['wearable_weekly', 'wearable_monthly', 'subject',
                       'anthropometric_group', 'body_composition', 'proteomics', 'microbiome']
    exclude_labels = ['frailty_height', 'weight', 'height', 'chronotype_ordinal', 'bt__mchc']
    wave_pattern = r'[rspqtj]_m[vs]_'

    ranks = {}
    for gender in ['male', 'female']:
        filtered = df[
            (df['sub_model'] == 'ensemble') &
            (df['model'] == model_type) &
            (df['gender'] == gender) &
            (df['score_type'] == 'pearson_r') &
            (df['wilcox_pvalue_fdr'] < fdr_threshold) &
            (df['score_pvalue'] < 0.05) &
            (df['score'] > 0) &
            (df['score'] <= 0.6) &
            (df['delta'] > 0)
        ].copy()
        filtered = filtered[~filtered['system'].isin(exclude_systems)]
        filtered = filtered[~filtered['label'].isin(exclude_labels)]
        filtered = filtered[~filtered['label'].str.match(wave_pattern, na=False)]
        order = filtered.groupby('system')['delta'].median().sort_values(ascending=False)
        for rank, system in enumerate(order.index):
            ranks.setdefault(system, []).append(rank)

    all_systems = list(ranks.keys())
    penalty = len(all_systems)
    avg_rank = {s: sum(ranks[s]) / len(ranks[s]) if len(ranks[s]) == 2
                else ranks[s][0] + penalty / 2 for s in all_systems}
    return sorted(all_systems, key=lambda s: avg_rank[s])


def get_top_biomarkers(df, model_type, gender, top_n=2, candidate_top_k=5,
                       fdr_threshold=0.1, fixed_system_order=None):
    df = merge_systems(df)
    filtered = df[
        (df['sub_model'] == 'ensemble') &
        (df['model'] == model_type) &
        (df['gender'] == gender) &
        (df['score_type'] == 'pearson_r') &
        (df['wilcox_pvalue_fdr'] < fdr_threshold) &
        (df['score_pvalue'] < 0.05) &
        (df['score'] > 0) &
        (df['score'] <= 0.6) &
        (df['delta'] > 0)
    ].copy()

    exclude_systems = ['wearable_weekly', 'wearable_monthly', 'subject',
                       'anthropometric_group', 'body_composition', 'proteomics', 'microbiome']
    exclude_labels = ['frailty_height', 'weight', 'height', 'chronotype_ordinal', 'bt__mchc']
    wave_pattern = r'[rspqtj]_m[vs]_'
    filtered = filtered[~filtered['system'].isin(exclude_systems)]
    filtered = filtered[~filtered['label'].isin(exclude_labels)]
    filtered = filtered[~filtered['label'].str.match(wave_pattern, na=False)]

    if fixed_system_order is not None:
        present = set(filtered['system'].unique())
        systems_ordered = [s for s in fixed_system_order if s in present]
    else:
        systems_ordered = filtered.groupby('system')['delta'].median().sort_values(ascending=False).index.tolist()

    rows = []
    manual_for_gender = MANUAL_LABEL_SELECTION.get(gender, {})

    for system in systems_ordered:
        sys_data = filtered[filtered['system'] == system].copy()
        selected = pd.DataFrame()

        if system in manual_for_gender:
            picked_labels = manual_for_gender[system]
            if len(picked_labels) == 0:
                continue
            manual_selected = sys_data[sys_data['label'].isin(picked_labels)].copy()
            manual_selected['manual_rank'] = manual_selected['label'].map(
                lambda x: picked_labels.index(x) if x in picked_labels else 999)
            manual_selected = manual_selected.sort_values('manual_rank').drop(columns=['manual_rank'])
            selected = manual_selected

        if len(selected) < top_n:
            already = selected['label'].tolist() if 'label' in selected.columns else []
            remaining = sys_data[~sys_data['label'].isin(already)].copy()
            top_corr = remaining.sort_values('score', ascending=False).head(candidate_top_k)
            auto_selected = top_corr.sort_values(['delta', 'score'], ascending=False).head(top_n - len(selected))
            selected = pd.concat([selected, auto_selected], ignore_index=True)

        for _, row in selected.iterrows():
            rows.append({'system': row['system'], 'label': row['label'],
                         'score': row['score'], 'baseline_score': row['baseline_score'],
                         'delta': row['delta']})

    return pd.DataFrame(rows), systems_ordered


def pretty_label(label, max_len=32):
    if label in LABEL_DISPLAY_OVERRIDES:
        name = LABEL_DISPLAY_OVERRIDES[label]
    elif label in LABEL_RENAME_DICT:
        name = LABEL_RENAME_DICT[label]
    else:
        name = label.replace('_', ' ').replace('bt  ', '').title()
    if len(name) > max_len:
        name = name[:max_len - 1] + '…'
    return name


def create_combined_barplot(bio_df, systems_ordered, model_type, gender, std_df):
    n_bars = len(bio_df)
    bar_h = 0.26
    gap_within = 0.18
    system_gap = 0.55

    y_centers = []
    y = 0
    prev_system = None
    for _, row in bio_df.iterrows():
        if prev_system is not None and row['system'] != prev_system:
            y += system_gap
        y_centers.append(y)
        y += bar_h * 2 + gap_within
        prev_system = row['system']

    total_height = y
    fig_width = 3.54
    fig_height = max(3.5, total_height * 0.22 + 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    for i, (_, row) in enumerate(bio_df.iterrows()):
        system = row['system']
        yc = y_centers[i]

        base_color = np.array(SYSTEM_COLOR_MAP.get(system, (0.5, 0.5, 0.5, 1.0))[:3])
        light_color = tuple(base_color * 0.5 + 0.5) + (0.7,)
        dark_color = tuple(base_color) + (0.9,)

        y_base = yc
        y_gait = yc + bar_h + 0.03

        # Look up std for error bars
        std_row = std_df[(std_df['system'] == system) & (std_df['label'] == row['label']) & (std_df['gender'] == gender)]
        gait_std = float(std_row['gait_std'].iloc[0]) if not std_row.empty else None
        base_std = float(std_row['baseline_std'].iloc[0]) if not std_row.empty else None

        ax.barh(y_base, row['baseline_score'], height=bar_h, color=light_color, edgecolor='none', zorder=2,
                xerr=base_std, error_kw=dict(elinewidth=0.8, capsize=2, ecolor='#555555'))
        ax.barh(y_gait, row['score'], height=bar_h, color=dark_color, edgecolor='none', zorder=2,
                xerr=gait_std, error_kw=dict(elinewidth=0.8, capsize=2, ecolor='#555555'))

    label_y = [yc + bar_h / 2 + 0.01 for yc in y_centers]
    labels = [pretty_label(row['label']) for _, row in bio_df.iterrows()]
    ax.set_yticks(label_y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel('Pearson r', fontsize=7.5, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.46, right=0.98, top=0.98, bottom=0.11)

    base_fname = f'{model_type}_{gender}_combined_pearson_absolute'
    for ext in ['png', 'pdf']:
        out_path = os.path.join(OUTPUT_DIR, f'{base_fname}.{ext}')
        plt.savefig(out_path, dpi=200, facecolor='white', edgecolor='white', format=ext)
        print(f"Saved: {out_path}")
    plt.close()


def save_legend():
    fig, ax = plt.subplots(figsize=(3.54, 0.5))
    ax.axis('off')
    legend_patches = [
        mpatches.Patch(facecolor='#b0b0b0', alpha=0.7, label='Age, BMI, VAT & Height'),
        mpatches.Patch(facecolor='#606060', alpha=0.9, label='Gait, Age, BMI, VAT & Height'),
    ]
    ax.legend(handles=legend_patches, loc='center', fontsize=7.5, framealpha=0.9,
              ncol=2, handlelength=1.2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(OUTPUT_DIR, f'legend.{ext}'), dpi=200,
                    facecolor='white', edgecolor='white', format=ext)
    plt.close()
    print("Saved: legend")


def main():
    df = pd.read_csv(PEARSON_CSV, low_memory=False)
    std_df = pd.read_csv(STD_CSV)
    model_type = 'long_seq'

    unified_order = get_unified_system_order(df, model_type)
    print(f"System order: {unified_order}")

    for gender in ['male', 'female']:
        bio_df, systems_ordered = get_top_biomarkers(df, model_type, gender,
                                                      fixed_system_order=unified_order)
        print(f"\n{gender}: {len(bio_df)} biomarkers across {len(systems_ordered)} systems")
        create_combined_barplot(bio_df, systems_ordered, model_type, gender, std_df)

    save_legend()


if __name__ == '__main__':
    main()
