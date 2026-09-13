"""
Final Figure 3 radar plots:
  1. Ensemble all-all (single line, all genders combined)
  2. Gender overlay (male=blue, female=orange on same chart)

Both saved as PNG + PDF in this directory.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from publication_colors import (
    SYSTEM_COLOR_MAP, SYSTEM_RENAME_DICT, merge_systems
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

BODY_SYSTEMS = ['body_composition', 'bone_density', 'anthropometric_group', 'frailty']

EXCLUDE_LABELS = ['frailty_height', 'height', 'weight', 'total_scan_vat_area', 'gender', 'bmi', 'age', 'Creatinine']
EXCLUDE_LABEL_KEYWORDS = ['vat']
EXCLUDE_SYSTEMS = ['proteomics']

# Benjamini-Hochberg FDR threshold for endpoint selection. Matches the threshold
# already used by Figures 4 and 5 and Extended Data Figure 1
# (wilcox_pvalue_fdr < 0.1), so one correction regime applies across the article.
FDR_Q = 0.10

# Local overrides on top of SYSTEM_RENAME_DICT
SYSTEM_RENAME_OVERRIDES = {
    'body_composition': 'Body Comp',
}

LABEL_RENAME = {
    'bt__hdl_cholesterol': 'HDL Cholesterol',
    'bt__triglycerides': 'Triglycerides',
    'bt__total_cholesterol': 'Total Cholesterol',
    'bt__non_hdl_cholesterol': 'Non-HDL Cholesterol',
    'bt__hemoglobin': 'Hemoglobin',
    'bt__hct': 'Hematocrit',
    'bt__rbc': 'RBC Count',
    'bt__ferritin': 'Ferritin',
    'bt__alt_gpt': 'ALT (GPT)',
    'bt__ast_got': 'AST (GOT)',
    'bt__albumin': 'Albumin',
    'bt__creatinine': 'Creatinine',
    'bt__urea': 'Urea',
    'bt__glucose': 'Glucose',
    'bt__hba1c': 'HbA1c',
    'bt__wbc': 'WBC Count',
    'bt__monocytes_abs': 'Monocytes',
    'bt__monocytes_%': 'Monocytes (%)',
    'bt__lymphocytes_abs': 'Lymphocytes',
    'bt__neutrophils_abs': 'Neutrophils',
    'bt__mchc': 'MCHC',
    'liver_sound_speed': 'Liver Sound Speed',
    'liver_elasticity': 'Liver Elasticity',
    'sitting_blood_pressure_systolic': 'Sitting BP Systolic',
    'standing_three_min_blood_pressure_systolic': 'Standing BP Systolic',
    'lying_blood_pressure_systolic': 'Lying BP Systolic',
    'l_ankle_pressure': 'Ankle Pressure',
    't_mv_V4': 'T-wave Amplitude (V4)',
    'body_comp_total_lean_mass': 'Total Lean Mass',
    'body_comp_total_fat_mass': 'Total Fat Mass',
    'body_comp_android_tissue_mass': 'Android Tissue Mass',
    'body_comp_gynoid_tissue_mass': 'Gynoid Tissue Mass',
    'body_comp_total_fat_free_mass': 'Fat-Free Mass',
    'waist': 'Waist Circumference',
    'body_total_bmc': 'Total BMC',
    'body_total_area': 'Total Bone Area',
    'body_arms_bmc': 'Arms BMC',
    'body_legs_bmc': 'Legs BMC',
    'hand_grip_right': 'R Hand Grip Strength',
    'hand_grip_left': 'L Hand Grip Strength',
    'frailty_body_comp_arm_left_lean_mass': 'L Arm Lean Mass',
    'frailty_body_comp_arm_right_lean_mass': 'R Arm Lean Mass',
    'frailty_body_comp_leg_left_lean_mass': 'L Leg Lean Mass',
    'frailty_body_comp_leg_right_lean_mass': 'R Leg Lean Mass',
    'vigorous_activity_minutes': 'Vigorous Activity',
    'work_hours_day': 'Work Hours/Day',
    'work_hours': 'Work Hours',
    'physical_activity_maderate_days_a_week': 'Moderate Activity Days',
    'ahi': 'AHI',
    'hypoxic_burden': 'Hypoxic Burden',
    'rdi': 'RDI',
    'saturation_mean': 'Mean O2 Saturation',
    'odi': 'ODI',
    'Creatinine': 'Creatinine (NMR)',
    'HDL_size': 'HDL Size',
    'L_HDL_C': 'Large HDL-C',
    'L_HDL_CE': 'Large HDL-CE',
    'VLDL_size': 'VLDL Size',
    'rds_score': 'RDS Score',
    'depressed_longest_period_weeks': 'Longest Depression (weeks)',
    'walking_pace_ordinal': 'Walking Pace',
    'health_satisfaction': 'Health Satisfaction',
    'friendships_satisfaction': 'Friendship Satisfaction',
    'financial_situation_satisfaction': 'Financial Satisfaction',
    # Metabolites — annotated names cleaned up for display
    '12a-Hydroxy-3-oxocholadienic acid_Bufalin': '12a-OH-Oxocholadienic Acid',
    'indolelactate_3-Indolehydracrylic acid': 'Indolelactate',
    '11beta-hydroxyandrosterone glucuronide': '11β-OH-Androsterone Glucuronide',
    'Cholic acid glucuronide': 'Cholic Acid Glucuronide',
    '16alpha-Hydroxy DHEA 3-sulfate_16beta-Hydroxydehydroepiandrosterone sulfate_16a-Hydroxy dhea 3-sulphate': '16α-OH-DHEA Sulfate',
    'Sphingosine 1-phosphate (d16_1-P)': 'S1P (d16:1)',
    'Fibrinopeptide A (2-15)': 'Fibrinopeptide A (2-15)',
    'Pregnenolone sulfate_3beta-Hydroxypregn-5-en-20-one sulfate': 'Pregnenolone Sulfate',
    'O-Cresol sulfate_p-Cresol sulfate': 'p-Cresol Sulfate',
    'gamma-glutamylleucine': 'γ-Glutamylleucine',
    'xanthurenate': 'Xanthurenate',
    'urate': 'Urate',
    'biliverdin': 'Biliverdin',
    'Thyroxine': 'Thyroxine',
    'Phenylacetylglutamine': 'Phenylacetylglutamine',
    'Indoleacrylic acid': 'Indoleacrylic Acid',
    'Valsartan': 'Valsartan',
    'Dodecanedioic acid': 'Dodecanedioic Acid',
    'Dodecadienoate': 'Dodecadienoate',
    '15 16-DiHODE': '15,16-DiHODE',
    '7-HOC acid': '7-HOC Acid',
    '7-Hoca': '7-HOCA',
    '4-Vinylphenol sulfate': '4-Vinylphenol Sulfate',
    '4-Hydroxyphenylacetic acid sulfate': '4-OH-Phenylacetic Acid Sulfate',
    'Cortolone-3-glucuronide': 'Cortolone-3-Glucuronide',
    'Tetradecanedioate(C14-dicarboxylic acid )': 'Tetradecanedioate',
    'Hexadecanedioic acid': 'Hexadecanedioic Acid',
    'Estrone sulfate_Estrone sulphate': 'Estrone Sulfate',
    'p-Cresol glucuronide_P-Tolyl-β-D-glucuronide': 'p-Cresol Glucuronide',
    'Enterolactone 3-glucuronide_Enterolactone 3-glucuronide': 'Enterolactone Glucuronide',
    'Enterolactone 3-sulfate': 'Enterolactone Sulfate',
    'Phenylalanylphenylalanine': 'Phe-Phe Dipeptide',
    'Phenylalanylthreonine': 'Phe-Thr Dipeptide',
    'Fibrinopeptide B (1-11)': 'Fibrinopeptide B (1-11)',
    'Fibrinopeptide B (1-12)': 'Fibrinopeptide B (1-12)',
    'Fibrinopeptide B (1-13)': 'Fibrinopeptide B (1-13)',
    'Fibrinopeptide A human': 'Fibrinopeptide A',
    'Fibrinopeptide A (3-15)': 'Fibrinopeptide A (3-15)',
    'Fibrinopeptide A (3-16)': 'Fibrinopeptide A (3-16)',
    'Fibrinopeptide A (5-16)*': 'Fibrinopeptide A (5-16)',
    'Fibrinopeptide A (8-16)': 'Fibrinopeptide A (8-16)',
}

ACTIVITY_PATTERNS_TO_REMOVE = [
    ' - rombergs_closed', ' - rombergs_open', ' - sit_to_stand', ' - stationary',
    ' - TM3', ' - TMS',
]

PREFERRED_METABOLITE_LABELS = [
    '12a-Hydroxy-3-oxocholadienic acid_Bufalin',
    'indolelactate_3-Indolehydracrylic acid',
    '11beta-hydroxyandrosterone glucuronide',
    'Cholic acid glucuronide',
    '16alpha-Hydroxy DHEA 3-sulfate_16beta-Hydroxydehydroepiandrosterone sulfate_16a-Hydroxy dhea 3-sulphate',
    'Sphingosine 1-phosphate (d16_1-P)',
    'Fibrinopeptide A (2-15)',
    'Pregnenolone sulfate_3beta-Hydroxypregn-5-en-20-one sulfate',
    'O-Cresol sulfate_p-Cresol sulfate',
    'gamma-glutamylleucine',
]

PREFERRED_LABELS = {
    'blood_tests_lipids': ['bt__hdl_cholesterol', 'bt__triglycerides',
                           'bt__total_cholesterol', 'bt__non_hdl_cholesterol'],
    'hematopoietic': ['bt__hemoglobin', 'bt__hct', 'bt__rbc', 'bt__ferritin'],
    'liver': ['liver_sound_speed', 'liver_elasticity', 'bt__alt_gpt', 'bt__ast_got', 'bt__albumin'],
    'cardiovascular_system': ['sitting_blood_pressure_systolic',
                              'standing_three_min_blood_pressure_systolic',
                              'lying_blood_pressure_systolic', 'l_ankle_pressure'],
    'renal_function': ['bt__creatinine', 'bt__urea', 'Creatinine'],
    'glycemic_status': ['bt__glucose', 'bt__hba1c'],
    'body_composition': ['body_comp_total_lean_mass', 'body_comp_total_fat_mass',
                         'body_comp_android_tissue_mass', 'body_comp_gynoid_tissue_mass',
                         'body_comp_total_fat_free_mass', 'waist'],
    'bone_density': ['body_total_bmc', 'body_total_area', 'body_arms_bmc', 'body_legs_bmc'],
    'frailty': ['hand_grip_right', 'hand_grip_left'],
    'immune_system': ['bt__wbc', 'bt__monocytes_abs', 'bt__lymphocytes_abs', 'bt__neutrophils_abs'],
    'lifestyle': ['vigorous_activity_minutes', 'work_hours_day', 'work_hours',
                  'physical_activity_maderate_days_a_week'],
    'sleep_group': ['ahi', 'hypoxic_burden', 'rdi', 'saturation_mean', 'odi'],
    'nightingale': ['Creatinine', 'HDL_size', 'L_HDL_C', 'L_HDL_CE'],
    # 'metabolites': PREFERRED_METABOLITE_LABELS,
}

KEYWORD_LIMITS = {
    'cardiovascular_system': {'mv': 3, 'blood_pressure': 3},
    'body_composition': {'arm_': 2, 'leg_': 2, 'android': 1, 'gynoid': 1},
    'bone_density': {'arm_': 2, 'leg_': 2},
    'frailty': {'frailty_': 2},
    'glycemic_status': {'iglu_': 2},
    'nightingale': {'HDL': 3, 'L_HDL': 2},
    'sleep_group': {'heart_rate': 2},
    'metabolites': {'Fibrinopeptide': 2},
}


def rename_label(label):
    """Rename label using LABEL_RENAME dict, fallback to title-case formatting."""
    if label in LABEL_RENAME:
        renamed = LABEL_RENAME[label]
        if label.startswith('bt__'):
            return f'BT {renamed}'
        return renamed

    if label.startswith('iglu_'):
        token_renames = {
            'adrr': 'ADRR', 'auc': 'AUC', 'cogi': 'COGI', 'conga': 'CONGA',
            'cv': 'CV', 'ea1c': 'eA1c', 'gmi': 'GMI', 'hbgi': 'HBGI',
            'iqr': 'IQR', 'lbgi': 'LBGI', 'mad': 'MAD', 'mag': 'MAG',
            'mage': 'MAGE', 'modd': 'MODD', 'sd': 'SD', 'sdb': 'SDB',
            'sdbdm': 'SDBDM', 'sdhhmm': 'SDHHMM', 'sdw': 'SDW',
        }
        suffix_tokens = label[len('iglu_'):].split('_')
        pretty_tokens = [token_renames.get(tok, tok.capitalize()) for tok in suffix_tokens]
        return 'IGLU ' + ' '.join(pretty_tokens)

    cleaned = label
    for pattern in ACTIVITY_PATTERNS_TO_REMOVE:
        cleaned = cleaned.replace(pattern, '')

    cleaned = cleaned.replace('_m:', ' (m):').replace('_s:', ' (s):').replace('_pct:', ' (%):')
    cleaned = cleaned.replace('_kmh:', ' (km/h):').replace('_ms:', ' (ms):')
    cleaned = cleaned.replace('LLeg_', 'L Leg ').replace('RLeg_', 'R Leg ')
    cleaned = cleaned.replace('LArm_', 'L Arm ').replace('RArm_', 'R Arm ')
    cleaned = cleaned.replace('LStep_', 'L Step ').replace('RStep_', 'R Step ')
    cleaned = cleaned.replace('Leg_', 'Leg ').replace('Step_', 'Step ').replace('Stride_', 'Stride ')
    cleaned = cleaned.replace('Arm_Swing', 'Arm Swing').replace('Hedo_vrt_sway', 'Head Vert Sway')
    cleaned = cleaned.replace('Walking_speed', 'Walking Speed')
    cleaned = cleaned.replace('Non_dir_Diff_Wrist_Sway', 'Wrist Sway Diff')
    cleaned = cleaned.replace('Non_dir_Ratio_Wrist_Sway', 'Wrist Sway Ratio')
    cleaned = cleaned.replace('_1D_in_z_dir', '').replace('_m', ' (m)').replace('_s', ' (s)')
    cleaned = cleaned.replace('_pct', ' (%)').replace('_kmh', ' (km/h)')
    cleaned = cleaned.replace('double_support_time', 'Double Support').replace('single_support_time', 'Single Support')
    cleaned = cleaned.replace('stance_time', 'Stance Time').replace('swing_time', 'Swing Time')
    cleaned = cleaned.replace('length', 'Length').replace('time', 'Time').replace('width', 'Width')
    cleaned = cleaned.replace('asymmetry', 'Asymmetry')
    cleaned = cleaned.replace('_', ' ')

    if cleaned == label:
        cleaned = label.replace('_', ' ').replace('bt  ', '').title()

    words = cleaned.split()
    cleaned = ' '.join(word[0].upper() + word[1:] if len(word) > 0 else word for word in words)

    if label.startswith('bt__'):
        cleaned = cleaned.replace('Bt ', '').replace('Bt__', '').strip()
        return f'BT {cleaned}'

    return cleaned


def get_top_labels_by_system(df, n_per_system=4, exclude_systems=None):
    """Get top N labels per system, ordered by median score (descending)."""
    df_filtered = df[~df['label'].isin(EXCLUDE_LABELS)].copy()
    if exclude_systems is None:
        exclude_systems = EXCLUDE_SYSTEMS
    if exclude_systems:
        df_filtered = df_filtered[~df_filtered['system'].isin(exclude_systems)].copy()
    for keyword in EXCLUDE_LABEL_KEYWORDS:
        df_filtered = df_filtered[~df_filtered['label'].str.contains(keyword, case=False, na=False)]

    all_systems = df_filtered['system'].unique()

    system_labels = {}
    for system in all_systems:
        system_data = df_filtered[df_filtered['system'] == system].copy()

        preferred = PREFERRED_LABELS.get(system, [])
        if preferred:
            preferred_data = system_data[system_data['label'].isin(preferred)].sort_values('score', ascending=False)
            other_data = system_data[~system_data['label'].isin(preferred)].sort_values('score', ascending=False)
            system_data = pd.concat([preferred_data, other_data])
        else:
            system_data = system_data.sort_values('score', ascending=False)

        keyword_counts = {kw: 0 for kw in KEYWORD_LIMITS.get(system, {}).keys()}
        labels_for_system = []
        seen_display_labels = set()

        for _, row in system_data.iterrows():
            if len(labels_for_system) >= n_per_system:
                break

            label = row['label']
            skip = False
            if system in KEYWORD_LIMITS:
                for kw, limit in KEYWORD_LIMITS[system].items():
                    if kw.lower() in label.lower():
                        if keyword_counts[kw] >= limit:
                            skip = True
                            break
                        keyword_counts[kw] += 1
            if skip:
                continue

            display_name = rename_label(label).strip().lower()
            if display_name in seen_display_labels:
                continue
            seen_display_labels.add(display_name)

            labels_for_system.append((system, row['label'], row['score']))

        if len(labels_for_system) >= 2:
            system_labels[system] = labels_for_system

    system_median_scores = {
        sys: float(np.median([sc for _, _, sc in labels]))
        for sys, labels in system_labels.items()
    }
    sorted_systems = sorted(system_median_scores.keys(), key=lambda s: system_median_scores[s], reverse=True)

    top_labels = []
    for system in sorted_systems:
        sorted_labels = sorted(system_labels[system], key=lambda x: x[2], reverse=True)
        top_labels.extend(sorted_labels)

    return top_labels


def _draw_radar_base(ax, angles, all_systems, max_val, angle_step, system_color_map):
    """Draw arcs, separator lines, system name labels on the polar axis."""
    arc_radius = max_val * 1.12
    label_radius = max_val * 1.18

    for angle in angles:
        ax.plot([angle, angle], [0, arc_radius], color='lightgray', linewidth=0.5, alpha=0.5)

    # System segments
    system_segments = []
    current_sys = all_systems[0]
    start_idx = 0
    for i, sys in enumerate(all_systems):
        if sys != current_sys:
            system_segments.append((current_sys, start_idx, i - 1))
            current_sys = sys
            start_idx = i
    system_segments.append((current_sys, start_idx, len(all_systems) - 1))

    arc_inner_offset = max_val * 0.01
    for angle, system in zip(angles, all_systems):
        sys_color = system_color_map[system]
        arc_angles = np.linspace(angle + angle_step / 2, angle - angle_step / 2, 30)
        ax.plot(arc_angles, [arc_radius - arc_inner_offset] * len(arc_angles),
                linewidth=22, color=sys_color, solid_capstyle='butt')

    for sys, start_idx, end_idx in system_segments:
        start_angle = angles[start_idx] + angle_step / 2
        ax.plot([start_angle, start_angle], [arc_radius * 0.95, arc_radius * 1.05],
                color='black', linewidth=1.5, zorder=10)

    ROTATION_RULES = {
        (0, 90): 'outside', (90, 180): 'outside',
        (180, 270): 'inside', (270, 360): 'inside',
    }
    for sys, start_idx, end_idx in system_segments:
        start_angle = angles[start_idx] + angle_step / 2
        end_angle = angles[end_idx] - angle_step / 2
        mid_angle = (start_angle + end_angle) / 2
        angle_norm = mid_angle % (2 * np.pi)
        angle_deg = np.degrees(angle_norm)
        rotation = next(
            (angle_deg + 90 if v == 'inside' else angle_deg - 90
             for (lo, hi), v in ROTATION_RULES.items() if lo <= angle_deg < hi),
            angle_deg - 90
        )
        sys_name = SYSTEM_RENAME_OVERRIDES.get(sys, SYSTEM_RENAME_DICT.get(sys, sys.replace('_', ' ').title()))
        ax.text(mid_angle, arc_radius * 0.99, sys_name, fontsize=12, fontweight='bold',
                ha='center', va='center', rotation=rotation, rotation_mode='anchor',
                color='black', bbox=dict(boxstyle='round,pad=0.1', facecolor='none', edgecolor='none'))

    return arc_radius, label_radius, system_segments


def _add_outer_labels(ax, angles, labels, label_radius):
    """Add label text outside the colored arcs."""
    for angle, label in zip(angles, labels):
        angle_norm = angle % (2 * np.pi)
        angle_deg = np.degrees(angle_norm)
        if angle_deg <= 90 or angle_deg >= 270:
            rotation = angle_deg
            ha = 'left'
        else:
            rotation = angle_deg + 180
            ha = 'right'
        display_label = rename_label(label)
        ax.text(angle, label_radius, display_label, fontsize=12,
                ha=ha, va='center', rotation=rotation, rotation_mode='anchor')


def _save_fig(fig, base_path):
    """Save figure as both PNG and PDF."""
    fig.savefig(base_path + '.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {base_path}.png')
    fig.savefig(base_path + '.pdf', bbox_inches='tight')
    print(f'Saved: {base_path}.pdf')


def create_radar_ensemble_all(df_data, save_base, figsize=(14, 14), n_per_system=5, exclude_systems=None):
    """Single-line radar: ensemble all genders."""
    top_labels = get_top_labels_by_system(df_data, n_per_system=n_per_system, exclude_systems=exclude_systems)
    if not top_labels:
        print("No labels found")
        return None

    labels = [l for _, l, _ in top_labels]
    scores = [s for _, _, s in top_labels]
    systems = [sys for sys, _, _ in top_labels]
    n_labels = len(labels)

    unique_systems = list(dict.fromkeys(systems))
    system_color_map = {sys: (*SYSTEM_COLOR_MAP.get(sys, (0.5, 0.5, 0.5, 1.0))[:3], 0.6) for sys in unique_systems}

    angles = (np.pi / 2 - np.linspace(0, 2 * np.pi, n_labels, endpoint=False)).tolist()
    angle_step = 2 * np.pi / n_labels
    angles_closed = angles + [angles[0]]
    scores_closed = scores + [scores[0]]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    ax.plot(angles_closed, scores_closed, 'o-', linewidth=2,
            color='#393b79', markersize=5, alpha=0.8)
    ax.fill(angles_closed, scores_closed, color='#393b79', alpha=0.2)

    max_val = max(scores)
    min_val = min(scores)
    arc_radius, label_radius, _ = _draw_radar_base(ax, angles, systems, max_val, angle_step, system_color_map)

    ax.set_xticks([])
    _add_outer_labels(ax, angles, labels, label_radius)

    y_min = min(0, min_val - 0.02) if min_val < 0 else -0.02
    ax.set_ylim(y_min, arc_radius * 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_fig(fig, save_base)
    plt.close(fig)
    return fig


def create_radar_gender_overlay(df_male, df_female, save_base, figsize=(14, 14), n_per_system=5, exclude_systems=None):
    """Two-line radar: male (blue) + female (orange) on the same chart."""
    common_labels = set(df_male['label'].unique()) & set(df_female['label'].unique())
    df_male_c = df_male[df_male['label'].isin(common_labels)].copy()
    df_female_c = df_female[df_female['label'].isin(common_labels)].copy()

    # Use male data as reference for label selection order
    top_labels = get_top_labels_by_system(df_male_c, n_per_system=n_per_system, exclude_systems=exclude_systems)
    if not top_labels:
        print("No labels found")
        return None

    all_labels = [l for _, l, _ in top_labels]
    all_systems = [sys for sys, _, _ in top_labels]
    n_labels = len(all_labels)

    unique_systems = list(dict.fromkeys(all_systems))
    system_color_map = {sys: (*SYSTEM_COLOR_MAP.get(sys, (0.5, 0.5, 0.5, 1.0))[:3], 0.6) for sys in unique_systems}

    angles = (np.pi / 2 - np.linspace(0, 2 * np.pi, n_labels, endpoint=False)).tolist()
    angle_step = 2 * np.pi / n_labels
    angles_closed = angles + [angles[0]]

    male_scores = [df_male_c[df_male_c['label'] == l]['score'].iloc[0] if len(df_male_c[df_male_c['label'] == l]) > 0 else 0 for l in all_labels]
    female_scores = [df_female_c[df_female_c['label'] == l]['score'].iloc[0] if len(df_female_c[df_female_c['label'] == l]) > 0 else 0 for l in all_labels]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))

    male_color = '#1f77b4'
    female_color = '#ff7f0e'

    ax.plot(angles_closed, male_scores + [male_scores[0]], 'o-', linewidth=2,
            label='Male', color=male_color, markersize=5, alpha=0.8)
    ax.fill(angles_closed, male_scores + [male_scores[0]], alpha=0.15, color=male_color)

    ax.plot(angles_closed, female_scores + [female_scores[0]], 'o-', linewidth=2,
            label='Female', color=female_color, markersize=5, alpha=0.8)
    ax.fill(angles_closed, female_scores + [female_scores[0]], alpha=0.15, color=female_color)

    all_scores = male_scores + female_scores
    max_val = max(all_scores)
    min_val = min(all_scores)
    arc_radius, label_radius, _ = _draw_radar_base(ax, angles, all_systems, max_val, angle_step, system_color_map)

    ax.set_xticks([])
    _add_outer_labels(ax, angles, all_labels, label_radius)

    y_min = min(0, min_val - 0.02) if min_val < 0 else -0.02
    ax.set_ylim(y_min, arc_radius * 1.02)
    ax.grid(True, alpha=0.3)

    # Embedded Male/Female legend at bottom-right of the plot
    ax.legend(loc='lower right', bbox_to_anchor=(1.12, 0.0), fontsize=13,
              framealpha=0.9, handlelength=1.5)

    plt.tight_layout()

    _save_fig(fig, save_base)
    plt.close(fig)
    return fig


def save_gender_legend(save_base):
    """Save a standalone Male/Female legend as PNG and PDF."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(2.2, 0.8))
    ax.axis('off')
    handles = [
        Line2D([0], [0], color='#1f77b4', linewidth=2.5, marker='o', markersize=6, label='Male'),
        Line2D([0], [0], color='#ff7f0e', linewidth=2.5, marker='o', markersize=6, label='Female'),
    ]
    ax.legend(handles=handles, loc='center', fontsize=13, framealpha=0.9,
              ncol=2, handlelength=1.5, columnspacing=1.2)
    plt.tight_layout(pad=0.2)
    _save_fig(fig, save_base)
    plt.close(fig)


def significant_by_fdr(df, gender, q=FDR_Q):
    """Endpoints passing BH-FDR, corrected across every endpoint tested in the panel.

    Replaces the previous uncorrected `score_pvalue < 0.05` gate: Nature requires
    the figure legend to state whether an adjustment for multiple comparisons was
    made, and leaving this panel uncorrected made it the only figure in the
    article without one.
    """
    panel = df[
        (df['model'] == 'long_seq') &
        (df['sub_model'] == 'ensemble') &
        (df['gender'] == gender)
    ].copy()

    valid = panel['score_pvalue'].notna()
    panel['score_pvalue_fdr'] = np.nan
    _, corrected, _, _ = multipletests(panel.loc[valid, 'score_pvalue'].values,
                                       method='fdr_bh')
    panel.loc[valid, 'score_pvalue_fdr'] = corrected
    return panel[panel['score_pvalue_fdr'] < q].copy()


def main():
    pearson_csv = os.path.join(RESULTS_DIR, 'gait_only_pearson.csv')

    df = pd.read_csv(pearson_csv)
    df = merge_systems(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Figure 3: Ensemble all-all ────────────────────────────────────────────
    df_ensemble_all = significant_by_fdr(df, 'all')

    print(f"Ensemble all: {len(df_ensemble_all)} labels")
    create_radar_ensemble_all(
        df_ensemble_all[['system', 'label', 'score']],
        os.path.join(OUTPUT_DIR, 'figure3_ensemble_all_pearson_r'),
        n_per_system=5,
        exclude_systems=None,  # uses EXCLUDE_SYSTEMS = ['proteomics']
    )

    # ── Extended: Gender overlay ──────────────────────────────────────────────
    df_male = significant_by_fdr(df, 'male')

    df_female = significant_by_fdr(df, 'female')

    print(f"Male: {len(df_male)}, Female: {len(df_female)}")
    create_radar_gender_overlay(
        df_male[['system', 'label', 'score']],
        df_female[['system', 'label', 'score']],
        os.path.join(OUTPUT_DIR, 'figure3_extended_gender_overlay_pearson_r'),
        n_per_system=5,
        exclude_systems=None,  # uses EXCLUDE_SYSTEMS = ['proteomics']
    )

    save_gender_legend(os.path.join(OUTPUT_DIR, 'figure3_extended_gender_legend'))

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
