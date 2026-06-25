"""
Combined horizontal bar plot: top biomarkers per system in a single figure.
Lighter bar = baseline (Age, BMI, VAT, Height), darker bar = full model (+ Gait).
Seed-level scatter points overlaid on each bar.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'results')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from publication_colors import (
    LABEL_RENAME_DICT, SYSTEM_COLOR_MAP, LIFESTYLE_DUPLICATE_MAP, merge_systems,
)

# Manually curated medically interpretable labels per system.
# If a listed label is not present after filtering, fallback selection fills remaining slots.
MANUAL_LABEL_SELECTION = {
    'male': {
        'blood_tests_lipids': ['bt__triglycerides', 'bt__hdl_cholesterol'],
        'bone_density': ['spine_l1_l4_area', 'spine_l1_l3_bmc'],
        'cardiovascular_system': ['hr_bpm', 'automorph_vein_fractal_dimension'],
        'frailty': ['hand_grip_left', 'hand_grip_right'],
        'glycemic_status': [],
        'hematopoietic': ['bt__hct'],  # top delta auto-fills second slot
        'immune_system': [],
        'lifestyle': ['vigorous_activity_days', 'sleep_hours'],
        'liver': ['liver_sound_speed', 'liver_viscosity'],
        'mental': ['happiness_level', 'depressed_longest_period_weeks'],
        'metabolites': ['Fibrinopeptide B (1-12)', 'C17 Oxylipin (291.16 m_z)'],
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
        'metabolites': ['Fibrinopeptide B (1-12)', 'C17 Oxylipin (291.16 m_z)'],
        'microbiome': ['Rep_2101', 'Rep_1819'],
        'nightingale': ['L_VLDL_FC', 'XL_HDL_PL'],
        'proteomics': [],
        'sleep_group': ['sleep_efficiency', 'saturation_mean'],
    },
}

# Clearer display names for manually selected labels.
LABEL_DISPLAY_OVERRIDES = {
    'automorph_vein_fractal_dimension': 'Vein Fractal Dimension',
    'spine_l1_l3_bmc': 'Spine L1-L3 BMC',
    'hand_grip_left': 'L Hand Grip',
    'hand_grip_right': 'R Hand Grip',
    'intima_media_th_2_fit': 'Carotid IMT',
    'bt__mcv': 'MCV',
    'frailty_body_comp_arm_right_lean_mass': 'R Arm Lean Mass',
    'frailty_body_comp_leg_right_lean_mass': 'R Leg Lean Mass',
    'depressed_longest_period_weeks': 'Depressive Episode Length',
    'xanthurenate _4 6-Dihydroxy-2-quinolinecarboxylic acid_Zeanic acid': 'Xanthurenate / Quinoline',
    'LysoPE(P-16_0_0_0)': 'LysoPE (P-16:0)',
    'Sphingosine 1-phosphate (d16_1-P)': 'S1P (d16:1)',
    'Pyrogallol-O-sulphate': 'Pyrogallol-O-sulphate',
    'C17 Oxylipin (291.16 m_z)': 'C17 Oxylipin',
    'Rep_3257': 'Microbiome Feature 3257',
    'Rep_2163': 'Microbiome Feature 2163',
    'Rep_2101': 'Microbiome Feature 2101',
    'Rep_1819': 'Microbiome Feature 1819',
    'GlycA': 'GlycA Inflammation Signal',
    'XL_HDL_PL': 'XL HDL Phospholipids',
    'L_VLDL_FC': 'L VLDL Free Cholesterol',
    'body_legs_bmc': 'Leg BMC',
    'body_arms_bmc': 'Arms BMC',
    'spine_l1_l4_area': 'Spine L1-L4 Area',
    'manual_work_ordinal': 'Manual Work Hours',
    'sleep_hours': 'Sleep Hours in 24H',
    'Perfluorohexanesulfonic acid\xa0(PFHxS)\xa0': 'PFHxS (PFAS)',
}


def _build_lifestyle_label_candidates(label):
    """
    Build possible original lifestyle labels that can map to the merged label.
    Needed because merge_systems() canonicalizes label names for ranking only.
    """
    candidates = [label]
    reverse_map = {}
    for src_label, canonical_label in LIFESTYLE_DUPLICATE_MAP.items():
        reverse_map.setdefault(canonical_label, []).append(src_label)

    if label in reverse_map:
        candidates.extend(reverse_map[label])
    if label in LIFESTYLE_DUPLICATE_MAP:
        candidates.append(LIFESTYLE_DUPLICATE_MAP[label])

    # Preserve order while deduplicating
    return list(dict.fromkeys(candidates))


_SEED_VALUES_CSV = os.path.join(_RESULTS_DIR, 'figure4_grid_seed_values.csv')
_SEED_VALUES_DF = None


def load_seed_values(base_path, system, label, model_type, gender,
                     baseline_model='Age_Gender_BMI_height_VAT'):
    """Load per-seed Pearson r values from the precomputed, de-identified CSV.

    (Originally globbed per-target metrics.csv across the cluster tree; the public
    repo reads the same 15-seed aggregate values from results/figure4_grid_seed_values.csv.)
    """
    global _SEED_VALUES_DF
    if _SEED_VALUES_DF is None:
        _SEED_VALUES_DF = pd.read_csv(_SEED_VALUES_CSV)

    sub = _SEED_VALUES_DF[
        (_SEED_VALUES_DF['system'] == system) &
        (_SEED_VALUES_DF['label'] == label) &
        (_SEED_VALUES_DF['gender'] == gender)
    ].sort_values('seed_idx')

    if sub.empty:
        return None, None
    return sub['baseline_r'].values, sub['gait_r'].values


def get_unified_system_order(pearson_path, model_type, fdr_threshold=0.1):
    """
    Compute a single system order shared by both genders.
    Systems are ranked by their average median-delta rank across male and female.
    """
    df = pd.read_csv(pearson_path, low_memory=False)
    df = merge_systems(df)

    exclude_systems = [
        'wearable_weekly', 'wearable_monthly', 'subject',
        'anthropometric_group', 'body_composition', 'proteomics', 'microbiome',
    ]
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

    # Average rank across genders; systems missing from one gender get a penalty rank
    all_systems = list(ranks.keys())
    penalty = len(all_systems)
    avg_rank = {s: sum(ranks[s]) / len(ranks[s]) if len(ranks[s]) == 2
                else ranks[s][0] + penalty / 2
                for s in all_systems}
    return sorted(all_systems, key=lambda s: avg_rank[s])


def get_top_biomarkers_combined(
    pearson_path,
    model_type,
    gender,
    top_n=2,
    candidate_top_k=5,
    fdr_threshold=0.1,
    use_manual_selection=True,
    fixed_system_order=None
):
    """
    Get top N biomarkers per merged system, sorted by delta.
    Uses the same merge flow as publication_figure_gm.py.
    """
    df = pd.read_csv(pearson_path, low_memory=False)
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

    exclude_systems = [
        'wearable_weekly',
        'wearable_monthly',
        'subject',
        'anthropometric_group',
        'body_composition',
        'proteomics',
        'microbiome',
    ]
    # bt__mchc excluded by request; ECG wave labels excluded by request
    exclude_labels = ['frailty_height', 'weight', 'height', 'chronotype_ordinal', 'bt__mchc']
    wave_pattern = r'[rspqtj]_m[vs]_'  # ECG wave amplitude/duration labels
    filtered = filtered[~filtered['system'].isin(exclude_systems)]
    filtered = filtered[~filtered['label'].isin(exclude_labels)]
    filtered = filtered[~filtered['label'].str.match(wave_pattern, na=False)]

    # Use a fixed shared order if provided, otherwise fall back to per-gender delta ranking.
    if fixed_system_order is not None:
        present = set(filtered['system'].unique())
        systems_ordered = [s for s in fixed_system_order if s in present]
    else:
        system_stats = filtered.groupby('system')['delta'].median().sort_values(ascending=False)
        systems_ordered = system_stats.index.tolist()

    rows = []
    manual_for_gender = MANUAL_LABEL_SELECTION.get(gender, {})

    for system in systems_ordered:
        sys_data = filtered[filtered['system'] == system].copy()
        selected = pd.DataFrame()
        manual_defined = False

        if use_manual_selection and system in manual_for_gender:
            manual_defined = True
            picked_labels = manual_for_gender[system]
            if len(picked_labels) == 0:
                # Explicitly skip this system when user requests hiding it.
                continue
            manual_selected = sys_data[sys_data['label'].isin(picked_labels)].copy()
            # Keep the manual order from the config (not alphabetical)
            manual_selected['manual_rank'] = manual_selected['label'].map(
                lambda x: picked_labels.index(x) if x in picked_labels else 999
            )
            manual_selected = manual_selected.sort_values('manual_rank').drop(columns=['manual_rank'])
            selected = manual_selected

        # Fill missing slots with strong automatic picks:
        # best remaining labels from top Pearson-correlation candidates,
        # ranked by delta then score.
        if len(selected) < top_n:
            already = selected['label'].tolist() if 'label' in selected.columns else []
            remaining = sys_data[~sys_data['label'].isin(already)].copy()
            top_corr = remaining.sort_values('score', ascending=False).head(candidate_top_k)
            auto_selected = top_corr.sort_values(['delta', 'score'], ascending=False).head(top_n - len(selected))
            selected = pd.concat([selected, auto_selected], ignore_index=True)

        for _, row in selected.iterrows():
            rows.append({
                'system': row['system'],
                'label': row['label'],
                'score': row['score'],
                'baseline_score': row['baseline_score'],
                'delta': row['delta'],
            })

    return pd.DataFrame(rows), systems_ordered


def pretty_label(label, max_len=32):
    """Convert raw label to publication-quality name, truncating if needed."""
    if label in LABEL_DISPLAY_OVERRIDES:
        name = LABEL_DISPLAY_OVERRIDES[label]
    elif label in LABEL_RENAME_DICT:
        name = LABEL_RENAME_DICT[label]
    else:
        name = label.replace('_', ' ').replace('bt  ', '').title()
    if len(name) > max_len:
        name = name[:max_len - 1] + '…'
    return name


def create_combined_barplot(bio_df, systems_ordered, model_type, gender,
                            output_dir, base_path, scale_mode='absolute'):
    """
    Create a single combined horizontal bar plot with seed scatter.

    scale_mode:
        'absolute'  - x-axis is raw Pearson r (0 to max)
        'zoomed'    - x-axis starts near the minimum baseline value
        'delta_only'- only show the delta (improvement) per biomarker
    """
    n_bars = len(bio_df)

    # Tight spacing
    bar_h = 0.26
    gap_within = 0.18       # gap between baseline/gait pair within same biomarker
    system_gap = 0.55       # extra gap between body systems

    # Pre-compute y positions (top to bottom)
    y_centers = []
    y = 0
    prev_system = None
    for _, row in bio_df.iterrows():
        if prev_system is not None and row['system'] != prev_system:
            y += system_gap
        y_centers.append(y)
        y += (bar_h * 2 + gap_within) if scale_mode != 'delta_only' else (bar_h + gap_within)
        prev_system = row['system']

    total_height = y
    fig_width = 3.54  # 9 cm for publication
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

        # Load seed-level data
        baseline_seeds, gait_seeds = load_seed_values(
            base_path, system, row['label'], model_type, gender
        )

        if scale_mode == 'delta_only':
            ax.barh(yc, row['delta'], height=bar_h, color=dark_color, edgecolor='none', zorder=2)
            if gait_seeds is not None and baseline_seeds is not None:
                deltas = gait_seeds - baseline_seeds
                jitter = np.linspace(-bar_h * 0.3, bar_h * 0.3, len(deltas))
                ax.scatter(deltas, yc + jitter, color='black', s=4, alpha=0.35, zorder=5)
        else:
            y_base = yc
            y_gait = yc + bar_h + 0.03

            ax.barh(y_base, row['baseline_score'], height=bar_h,
                    color=light_color, edgecolor='none', zorder=2)
            ax.barh(y_gait, row['score'], height=bar_h,
                    color=dark_color, edgecolor='none', zorder=2)

            if baseline_seeds is not None:
                jitter = np.linspace(-bar_h * 0.2, bar_h * 0.2, len(baseline_seeds))
                ax.scatter(baseline_seeds, y_base + jitter,
                           color='black', s=1.2, alpha=0.4, zorder=5)
            if gait_seeds is not None:
                jitter = np.linspace(-bar_h * 0.2, bar_h * 0.2, len(gait_seeds))
                ax.scatter(gait_seeds, y_gait + jitter,
                           color='black', s=1.2, alpha=0.4, zorder=5)

    # Y-axis labels (biomarker names)
    if scale_mode == 'delta_only':
        label_y = y_centers
    else:
        label_y = [yc + bar_h / 2 + 0.01 for yc in y_centers]

    labels = [pretty_label(row['label']) for _, row in bio_df.iterrows()]
    ax.set_yticks(label_y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', labelsize=6.5)
    ax.invert_yaxis()

    # X-axis
    if scale_mode == 'delta_only':
        ax.set_xlabel('Δ Pearson r (improvement over baseline)', fontsize=7.5, fontweight='bold')
        ax.set_xlim(left=0)
    else:
        ax.set_xlim(left=0)
        ax.set_xlabel('Pearson r', fontsize=7.5, fontweight='bold')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Fixed margins for 9 cm publication width.
    plt.subplots_adjust(left=0.46, right=0.98, top=0.98, bottom=0.11)

    os.makedirs(output_dir, exist_ok=True)
    base_fname = f'{model_type}_{gender}_combined_pearson_{scale_mode}'
    for ext in ['png', 'pdf']:
        out_path = os.path.join(output_dir, f'{base_fname}.{ext}')
        plt.savefig(out_path, dpi=200, facecolor='white', edgecolor='white',
                    format=ext)
        print(f"Saved: {out_path}")
    plt.close()


def save_legend(output_dir):
    """Save the legend as a standalone figure."""
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
        out_path = os.path.join(output_dir, f'legend.{ext}')
        plt.savefig(out_path, dpi=200, facecolor='white', edgecolor='white', format=ext)
        print(f"Saved: {out_path}")
    plt.close()


def main():
    pearson_path = os.path.join(_RESULTS_DIR, 'gait_vs_confounders_pearson.csv')
    base_path = None  # per-seed values come from results/figure4_grid_seed_values.csv
    output_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(output_dir, exist_ok=True)

    for model_type in ['long_seq']:
        unified_order = get_unified_system_order(pearson_path, model_type)
        print(f"Unified system order: {unified_order}")

        for gender in ['male', 'female']:
            bio_df, systems_ordered = get_top_biomarkers_combined(
                pearson_path,
                model_type,
                gender,
                top_n=2,
                candidate_top_k=5,
                use_manual_selection=True,
                fixed_system_order=unified_order,
            )
            print(f"\n{model_type} {gender}: {len(bio_df)} biomarkers across {len(systems_ordered)} systems")

            create_combined_barplot(
                bio_df, systems_ordered, model_type, gender,
                output_dir, base_path, scale_mode='absolute'
            )

    save_legend(output_dir)


if __name__ == '__main__':
    main()
