"""
Plot showing 4 body groups with lists of labels where each group is most significant.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import argparse
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # plotting/
from publication_colors import LABEL_RENAME_DICT

# ========== VIEWING ANGLES ==========
VIEW_ELEV = -112.5
VIEW_AZIM = -89.4

# Joint groups mapping (26-joint reduced skeleton)
JOINT_GROUPS = {
    'legs': [12, 13, 14, 15, 16, 17, 18, 19],
    'arms': [4, 5, 6, 7, 8, 9, 10, 11],
    'torso': [0, 1, 2, 3],
    'head': [20, 21, 22, 23, 24, 25],
}

# Bones for 26-joint skeleton
SKELETON_BONES = [
    (2, 1), (1, 0), (2, 3), (3, 20), (20, 21),
    (2, 4), (4, 5), (5, 6), (6, 7),
    (2, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
    (21, 22), (22, 23),
    (21, 24), (24, 25),
]

# Mapping from 26-joint noise format to k4abt 32-joint format
NOISE_TO_K4ABT = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 11, 9: 12, 10: 13, 11: 14,
    12: 18, 13: 19, 14: 20, 15: 21,
    16: 22, 17: 23, 18: 24, 19: 25,
    20: 26, 21: 27, 22: 28, 23: 29, 24: 30, 25: 31,
}

# Labels to exclude (AUC/classification tasks + redundant + noisy)
EXCLUDE_LABELS = {
    'high_exercise_duration_Between 2 and 3 hours', 'usual_walking_pace_Slow', 'label',
    'frailty_height',
    'frailty_body_comp_arm_left_lean_mass', 'frailty_body_comp_arm_right_lean_mass',
    'white_wine_glasses_week', 'red_wine_glasses_week', 'champagne_plus_white_wine_glasses_week',
    'fortified_wine_glasses_week', 'beer_plus_cider_pints_week', 'spirits_measures_week',
    'other_alcoholic_drinks_glasses_week',
    'COMT', 'ODAM', 'SAV1', 'DPP10', 'CD79B', 'CPE', 'PNLIPRP1', 'GlycA', 'CTSH',
    'iglu_1st_quartile', 'iglu_adrr', 'iglu_mean', 'iglu_median', 'iglu_std',
}

# Labels to group together (keep only first one as representative)
LABEL_GROUPS = {
    'spine_l1_l4_bmc': ['spine_l1_l2_bmd', 'spine_l1_l3_area', 'spine_l1_l3_bmc',
                        'spine_l1_l4_area', 'spine_l2_l4_area', 'spine_l2_l4_bmc', 'body_spine_area'],
    'body_comp_arms_lean_mass': ['body_comp_arm_left_fat_free_mass', 'body_comp_arm_left_lean_mass',
                                  'body_comp_arm_right_bone_mass', 'body_comp_arm_right_lean_mass',
                                  'body_comp_arms_fat_free_mass', 'body_comp_arm_right_bone_mass'],
    'frailty_body_comp_arm_left_lean_mass': ['frailty_body_comp_arm_right_lean_mass'],
    'frailty_body_comp_leg_left_lean_mass': ['frailty_body_comp_leg_right_lean_mass'],
    'bt__hdl_cholesterol': ['XL_HDL_C', 'XL_HDL_CE', 'XL_HDL_L', 'XL_HDL_PL', 'L_HDL_C', 'L_HDL_CE', 'bt__non_hdl_cholesterol'],
    'r_mv_V3': ['r_mv_aVR', 's_mv_V2', 's_mv_V3', 'st_mv_V1'],
    'from_l_thigh_to_l_ankle_duration': ['from_r_thigh_to_r_ankle_duration'],
    'heart_rate_mean_during_sleep': ['heart_rate_min_during_sleep'],
    'walking_10min_days_a_week': ['walking_minutes_day'],
}

SKIP_LABELS = set()
for grouped in LABEL_GROUPS.values():
    SKIP_LABELS.update(grouped)


def get_label_display(label):
    """Convert raw label to display name using publication naming first."""
    if label in LABEL_RENAME_DICT:
        display = LABEL_RENAME_DICT[label]
    else:
        display = label.replace('bt__', '').replace('_', ' ').strip()
        display = re.sub(r'\s+', ' ', display)
        display = display.title()

    display = re.sub(r'\bBody Comp\s+', '', display, flags=re.IGNORECASE)
    display = re.sub(
        r'^\s*Body\s+(Head|Arms|Legs|Trunk|Spine)\b',
        r'\1', display, flags=re.IGNORECASE,
    )
    display = re.sub(r'^\s*Wearable\s+', '', display, flags=re.IGNORECASE)
    if label.lower().startswith('bt__'):
        display = re.sub(r'^\s*BT\s*', '', display, flags=re.IGNORECASE)
        display = f'BT {display}'
    if display.strip().lower() == 'r hand grip':
        display = 'R Hand Grip Strength'
    if display.strip().lower() == 'qt ms':
        display = 'ECG QT Interval'
    if label == 'climb_stairs_ordinal':
        display = 'Stair Climbing'
    if label == 'intima_media_th_2_fit':
        display = 'Carotid IMT'
    if label == 'LysoPE(P-16_0_0_0)':
        display = 'LysoPE 16'
    if display.strip().lower() in {'sleep hours', 'sleep hours in 24h'}:
        display = 'Sleep Hours in 24H'
    if label == 'heart_rate_mean_during_sleep':
        display = 'HR During Sleep'
    if label == 'lying_blood_pressure_pulse_rate':
        display = 'Heart Rate'
    if label == 'walking_10min_days_a_week':
        display = 'Days A Week Walking'
    display = re.sub(r'\bBmc\b', 'BMC', display)
    display = re.sub(r'\bBmd\b', 'BMD', display)
    display = re.sub(r'\s+', ' ', display)
    return display.strip()


def should_include_label(label):
    if label in EXCLUDE_LABELS:
        return False
    if label in SKIP_LABELS:
        return False
    return True


# Walking activities to average
WALKING_ACTIVITIES = ['activity_self_selected_gait_speed', 'activity_tm_3kmh']

# Colors for each group
GROUP_COLORS = {
    'head': '#E74C3C',
    'arms': '#3498DB',
    'torso': '#2ECC71',
    'legs': '#F39C12',
}

GRAY_COLOR = '#808080'
FONT_SCALE = 2.15

TOP_CANDIDATES_PER_GROUP = 12
TOP_DIVERSE_TO_SHOW = 7
FORCE_INCLUDE_LABELS = ('age', 'gender')
TOKEN_STOPWORDS = {
    'and', 'or', 'of', 'in', 'to', 'with', 'without', 'mean', 'total',
    'left', 'right', 'during', 'from', 'for', 'the'
}

# Hand-picked from body_group_top_labels.csv (the `add` column). Rendered
# verbatim in the listed group (no data regrouping). Age is added manually to
# legs (its data-top group for males; ranks #46 so it was beyond the CSV top-30).
MANUAL_LABELS_PEARSON_MALE = {
    'head': [
        'L_HDL_P',
        'bt__hdl_cholesterol',
        'body_head_bmc',
        'lying_blood_pressure_pulse_rate',
        'body_comp_android_fat_mass',
    ],
    'arms': [
        'body_comp_arms_tissue_mass',
        'bmi',
        'body_arms_bmc',
        'automorph_artery_fractal_dimension',
        'bt__hct',
    ],
    'torso': [
        'vigorous_activity_days',
        'liver_elasticity',
        'wearable_weightlifting_monthly_hours',
        'rdi',
        'total_scan_vat_area',
    ],
    'legs': [
        'body_legs_bmc',
        'automorph_vein_fractal_dimension',
        'liver_sound_speed',
        'liver_viscosity',
        'age',
    ],
}

# Hand-picked from body_group_top_labels.csv (the `add` column), rendered verbatim.
MANUAL_LABELS_PEARSON_FEMALE = {
    'head': [
        'automorph_artery_fractal_dimension',
        'snoring',
        'L_HDL_P',
        'bt__ferritin',
        'bt__hct',
    ],
    'arms': [
        'body_comp_arms_fat_mass',
        'bmi',
        'total_scan_vat_area',
        'liver_sound_speed',
        'age',
    ],
    'torso': [
        'heart_rate_mean_during_sleep',
        'wearable_weightlifting_monthly_hours',
        'sleep_hours',
        'rdi',
        'liver_viscosity',
    ],
    'legs': [
        'body_comp_legs_lean_mass',
        'walking_10min_days_a_week',
        'body_legs_bmc',
        'hand_grip_right',
        'bt__triglycerides',
    ],
}


def load_skeleton(csv_path, frame_idx=0):
    """Load skeleton XYZ coordinates from CSV."""
    df = pd.read_csv(csv_path)
    coords = np.zeros((26, 3))
    for joint_idx in range(26):
        k4abt_idx = NOISE_TO_K4ABT[joint_idx]
        for col in df.columns:
            if col.startswith(f'{k4abt_idx}_') and col.endswith('_x'):
                coords[joint_idx] = [
                    df.iloc[frame_idx][col],
                    df.iloc[frame_idx][col.replace('_x', '_y')],
                    df.iloc[frame_idx][col.replace('_x', '_z')]
                ]
                break
    return coords


def compute_group_importance_averaged(df, label, gender, model_prefix, activities):
    """Compute importance scores averaged across multiple activities."""
    groups = ['arms', 'legs', 'torso', 'head']
    all_drops = {g: [] for g in groups}
    all_all_buts = {g: [] for g in groups}

    for activity in activities:
        df_act = df[df['sub_model'] == activity]
        baseline_model = f'{model_prefix}_seq'
        baseline_rows = df_act[
            (df_act['model'] == baseline_model)
            & (df_act['label'] == label)
            & (df_act['gender'] == gender)
        ]
        if len(baseline_rows) == 0:
            baseline_rows = df_act[
                (df_act['model'] == baseline_model)
                & (df_act['label'] == label)
                & (df_act['gender'] == 'all')
            ]
        if len(baseline_rows) == 0:
            continue

        baseline_score = baseline_rows['score'].values[0]

        for group in groups:
            masked_rows = df_act[
                (df_act['model'] == f'{model_prefix}_masked_{group}')
                & (df_act['label'] == label)
                & (df_act['gender'] == gender)
            ]
            all_but_rows = df_act[
                (df_act['model'] == f'{model_prefix}_all_but_{group}')
                & (df_act['label'] == label)
                & (df_act['gender'] == gender)
            ]
            if len(masked_rows) == 0:
                masked_rows = df_act[
                    (df_act['model'] == f'{model_prefix}_masked_{group}')
                    & (df_act['label'] == label)
                    & (df_act['gender'] == 'all')
                ]
            if len(all_but_rows) == 0:
                all_but_rows = df_act[
                    (df_act['model'] == f'{model_prefix}_all_but_{group}')
                    & (df_act['label'] == label)
                    & (df_act['gender'] == 'all')
                ]
            if len(masked_rows) > 0 and len(all_but_rows) > 0:
                all_drops[group].append(baseline_score - masked_rows['score'].values[0])
                all_all_buts[group].append(all_but_rows['score'].values[0])

    drops = {g: np.mean(v) if v else 0 for g, v in all_drops.items()}
    all_buts = {g: np.mean(v) if v else 0 for g, v in all_all_buts.items()}

    if not any(drops.values()):
        return None

    def minmax_normalize(d):
        vals = list(d.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return {k: 0.5 for k in d}
        return {k: (v - mn) / (mx - mn) for k, v in d.items()}

    drops_norm = minmax_normalize(drops)
    all_buts_norm = minmax_normalize(all_buts)
    importance = {g: drops_norm[g] + all_buts_norm[g] for g in groups}
    return importance


def get_top_group(importance):
    if not importance:
        return None
    return max(importance.items(), key=lambda x: x[1])[0]


def plot_skeleton_single_group(skeleton_coords, highlight_group, ax, group_color, fixed_max_range=None):
    """Plot skeleton with single group highlighted."""
    coords = skeleton_coords - skeleton_coords.mean(axis=0)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    highlight_joints = set(JOINT_GROUPS.get(highlight_group, []))

    for j1, j2 in SKELETON_BONES:
        if j1 < 26 and j2 < 26:
            if j1 in highlight_joints and j2 in highlight_joints:
                color, linewidth, alpha = group_color, 5, 1.0
            else:
                color, linewidth, alpha = GRAY_COLOR, 2, 0.4
            ax.plot([x[j1], x[j2]], [y[j1], y[j2]], [z[j1], z[j2]],
                   color=color, linewidth=linewidth, alpha=alpha, solid_capstyle='round')

    for j in range(26):
        if j in highlight_joints:
            color, size, alpha = group_color, 100, 1.0
        else:
            color, size, alpha = GRAY_COLOR, 30, 0.4
        ax.scatter(x[j], y[j], z[j], c=color, s=size, alpha=alpha)

    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    max_range = fixed_max_range if fixed_max_range is not None else np.abs(coords).max() * 0.65
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_axis_off()


def create_grouped_figure(
    df,
    skeleton,
    model_prefix='long',
    gender='male',
    output_path=None,
    manual_labels_by_group=None,
    fixed_max_range=None,
    show_all_labels=False,
    regroup_manual_labels=False,
    head_data_top_only=False,
    data_driven_top_k=None,
    enforce_in_list_top_n=10,
):
    """Create figure with 4 body groups and their top-ranked labels."""
    groups = ['head', 'arms', 'torso', 'legs']

    all_labels = [l for l in df['label'].unique() if should_include_label(l)]
    label_to_importance = {}
    for label in all_labels:
        importance = compute_group_importance_averaged(df, label, gender, model_prefix, WALKING_ACTIVITIES)
        if importance:
            label_to_importance[label] = importance

    group_to_labels_with_scores = {g: [] for g in groups}
    for label, importance in label_to_importance.items():
        top_group = get_top_group(importance)
        if top_group in group_to_labels_with_scores:
            top_score = importance[top_group]
            display_name = get_label_display(label)
            group_to_labels_with_scores[top_group].append((display_name, top_score))

    original_label_set = set()
    if manual_labels_by_group:
        for g_labels in manual_labels_by_group.values():
            original_label_set.update(g_labels)

    group_to_labels = {}
    if data_driven_top_k is not None:
        # Fully data-driven placement: every label sits in the body group the
        # data ranks as its top. Within each group, rank by importance. A label
        # from the original curated list is ENFORCED (guaranteed in) if it ranks
        # within the top `enforce_in_list_top_n` of that group; the remaining
        # slots are filled by pure data rank. Result: data_driven_top_k per group.
        raw_by_group = {g: [] for g in groups}
        for label, importance in label_to_importance.items():
            tg = get_top_group(importance)
            if tg not in raw_by_group:
                continue
            vals = sorted(importance.values(), reverse=True)
            gap = vals[0] - (vals[1] if len(vals) > 1 else 0.0)
            raw_by_group[tg].append((label, importance[tg], gap))

        for g in groups:
            ranked = sorted(raw_by_group[g], key=lambda x: (x[1], x[2]), reverse=True)
            rank_index = {lab: i for i, (lab, _, _) in enumerate(ranked)}
            top_n_labels = {lab for lab, _, _ in ranked[:enforce_in_list_top_n]}
            enforced = [lab for lab, _, _ in ranked if lab in original_label_set and lab in top_n_labels]
            ordered = enforced + [lab for lab, _, _ in ranked if lab not in enforced]

            picked, seen = [], set()
            for lab in ordered:
                disp = get_label_display(lab)
                if disp in seen:
                    continue
                seen.add(disp)
                picked.append(lab)
                if len(picked) >= data_driven_top_k:
                    break
            picked.sort(key=lambda lab: rank_index.get(lab, 10 ** 9))  # display in data-rank order
            group_to_labels[g] = [get_label_display(lab) for lab in picked]
            n_enf = sum(1 for lab in picked if lab in original_label_set)
            print(f"  [data-driven:{g}] {len(picked)} labels, {n_enf} enforced from original list: "
                  f"{group_to_labels[g]}")

    elif show_all_labels or not manual_labels_by_group:
        for g in groups:
            sorted_labels = sorted(group_to_labels_with_scores[g], key=lambda x: x[1], reverse=True)
            if show_all_labels:
                group_to_labels[g] = [label for label, _ in sorted_labels]
            else:
                top_candidates = sorted_labels[:TOP_CANDIDATES_PER_GROUP]
                group_to_labels[g] = select_diverse_labels(top_candidates, top_n=TOP_DIVERSE_TO_SHOW)
        if not show_all_labels:
            enforce_forced_labels(group_to_labels, label_to_importance, max_labels_per_group=TOP_DIVERSE_TO_SHOW)

    elif manual_labels_by_group and regroup_manual_labels:
        curated_group_by_label = {}
        curated_labels_ordered = []
        for src_group in groups:
            for raw_label in manual_labels_by_group.get(src_group, []):
                if raw_label not in curated_group_by_label:
                    curated_group_by_label[raw_label] = src_group
                    curated_labels_ordered.append(raw_label)

        regrouped = {g: [] for g in groups}
        for raw_label in curated_labels_ordered:
            importance = label_to_importance.get(raw_label)
            display_name = get_label_display(raw_label)
            if importance:
                target_group = get_top_group(importance)
                score = float(importance[target_group])
                manual_group = curated_group_by_label[raw_label]
                if target_group != manual_group:
                    print(f"  [regroup] {raw_label}: manual={manual_group} → model={target_group}")
            else:
                target_group = curated_group_by_label[raw_label]
                score = -np.inf
                print(f"  [missing] {raw_label}: no importance data, kept in manual group={target_group}")
            regrouped[target_group].append((display_name, score))

        for g in groups:
            regrouped[g] = sorted(regrouped[g], key=lambda x: x[1], reverse=True)
            group_to_labels[g] = [label for label, _ in regrouped[g]]
    elif manual_labels_by_group:
        for g in groups:
            if g in manual_labels_by_group:
                labels_g = manual_labels_by_group[g]
                if head_data_top_only and g == 'head':
                    # Head is the weakest region: show only labels the data
                    # actually ranks top-in-head, dropping hand-placed ones.
                    kept = []
                    for raw_label in labels_g:
                        imp = label_to_importance.get(raw_label)
                        if imp and get_top_group(imp) == 'head':
                            kept.append(raw_label)
                        else:
                            print(f"  [head-filter] dropped {raw_label} (data top != head)")
                    labels_g = kept
                group_to_labels[g] = [get_label_display(label) for label in labels_g]
    elif not show_all_labels:
        enforce_forced_labels(group_to_labels, label_to_importance, max_labels_per_group=TOP_DIVERSE_TO_SHOW)

    max_labels = max((len(v) for v in group_to_labels.values()), default=0)
    fig_height = max(8.5, 4.2 + 0.22 * max_labels) if show_all_labels else 8.5
    fig = plt.figure(figsize=(17, fig_height))

    # Top row: skeletons
    for i, group in enumerate(groups):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        plot_skeleton_single_group(
            skeleton, group, ax, GROUP_COLORS[group],
            fixed_max_range=fixed_max_range,
        )

    # Bottom row: ranked label lists
    for i, group in enumerate(groups):
        ax = fig.add_subplot(2, 4, i + 5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        labels = group_to_labels[group]
        if len(labels) == 0:
            ax.text(0.5, 0.5, 'No labels', ha='center', va='center',
                   fontsize=12, style='italic', color='gray')
        else:
            ranked_text = '\n'.join(labels)
            label_fontsize = (8.0 if show_all_labels else 11.0) * FONT_SCALE
            linespacing = 1.15 if show_all_labels else 1.2
            ax.text(0.5, 0.95, ranked_text, ha='center', va='top',
                   fontsize=label_fontsize, linespacing=linespacing,
                   bbox=dict(boxstyle='round,pad=0.15', facecolor=GROUP_COLORS[group],
                            alpha=0.15, edgecolor=GROUP_COLORS[group], linewidth=1.5))

    plt.suptitle(gender.title(), fontsize=20 * FONT_SCALE, fontweight='bold', y=0.96)
    plt.subplots_adjust(
        left=0.02, right=0.98, top=0.90, bottom=0.08,
        wspace=0.32, hspace=-0.10
    )

    if output_path:
        paths = output_path if isinstance(output_path, list) else [output_path]
        for path in paths:
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved: {path}")

    return fig


def tokenize_for_diversity(label):
    tokens = re.findall(r'[A-Za-z0-9]+', label.lower())
    return {t for t in tokens if len(t) > 2 and t not in TOKEN_STOPWORDS}


def jaccard_overlap(a, b):
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def select_diverse_labels(candidates, top_n=7):
    """Select diverse labels from score-sorted candidates."""
    if len(candidates) <= top_n:
        return [label for label, _ in candidates]

    scores = np.array([score for _, score in candidates], dtype=float)
    score_min, score_max = float(scores.min()), float(scores.max())
    if score_max == score_min:
        norm_scores = np.ones_like(scores) * 0.5
    else:
        norm_scores = (scores - score_min) / (score_max - score_min)

    labels = [label for label, _ in candidates]
    token_sets = [tokenize_for_diversity(label) for label in labels]

    selected_idx = [0]
    remaining = set(range(1, len(labels)))

    while len(selected_idx) < top_n and remaining:
        best_i, best_value = None, -1e9
        for i in remaining:
            max_overlap = max(jaccard_overlap(token_sets[i], token_sets[j]) for j in selected_idx)
            combined = 0.65 * float(norm_scores[i]) + 0.35 * (1.0 - max_overlap)
            if combined > best_value:
                best_value = combined
                best_i = i
        selected_idx.append(best_i)
        remaining.remove(best_i)

    selected_idx = sorted(selected_idx)
    return [labels[i] for i in selected_idx]


def enforce_forced_labels(group_to_labels, label_to_importance, max_labels_per_group):
    """Ensure required covariates are present in their strongest body-group lists."""
    existing_labels = {label for labels in group_to_labels.values() for label in labels}
    for pinned_label in FORCE_INCLUDE_LABELS:
        importance = label_to_importance.get(pinned_label)
        if not importance:
            continue
        pinned_group = get_top_group(importance)
        if pinned_group not in group_to_labels:
            continue
        pinned_display = get_label_display(pinned_label)
        if pinned_display in existing_labels:
            continue
        labels_for_group = group_to_labels[pinned_group]
        if pinned_display in labels_for_group:
            continue
        if len(labels_for_group) >= max_labels_per_group:
            labels_for_group[-1] = pinned_display
        else:
            labels_for_group.append(pinned_display)
        existing_labels.add(pinned_display)


def main():
    parser = argparse.ArgumentParser(description='Plot body-part grouped joint importance figures.')
    parser.add_argument('--metric', choices=['r2', 'pearson'], default='pearson')
    args = parser.parse_args()

    _here = os.path.dirname(__file__)
    _results = os.path.join(_here, '..', '..', '..', 'results')
    _samples = os.path.join(_here, '..', '..', '..', 'sample_data', 'skeleton_body_groups')
    csv_map = {
        # r2 variant not shipped (no listed figure uses it); pearson is the published metric.
        'r2': os.path.join(_results, 'masking_ablation_r2.csv'),
        'pearson': os.path.join(_results, 'masking_ablation_pearson.csv'),
    }
    suffix_map = {'r2': '', 'pearson': '_pearson'}

    csv_path = csv_map[args.metric]
    output_suffix = suffix_map[args.metric]
    male_path = os.path.join(_samples, 'skeleton_male.csv')
    female_path = os.path.join(_samples, 'skeleton_female.csv')
    output_dir = os.path.join(_here, 'figure6b')
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    skeleton_male = load_skeleton(male_path)
    skeleton_female = load_skeleton(female_path)
    male_range = np.abs(skeleton_male - skeleton_male.mean(axis=0)).max() * 0.65
    female_range = np.abs(skeleton_female - skeleton_female.mean(axis=0)).max() * 0.65
    shared_skeleton_range = max(male_range, female_range)

    print(f"Creating grouped body part figures (metric={args.metric})...")

    def _paths(stem):
        return [os.path.join(output_dir, f'{stem}.png'), os.path.join(output_dir, f'{stem}.pdf')]

    print("\nGenerating male figure...")
    male_manual = MANUAL_LABELS_PEARSON_MALE if args.metric == 'pearson' else None
    fig_male = create_grouped_figure(
        df, skeleton_male, model_prefix='long', gender='male',
        output_path=_paths(f'grouped_by_body_part_male{output_suffix}'),
        manual_labels_by_group=male_manual,
        fixed_max_range=shared_skeleton_range,
    )
    plt.close(fig_male)

    print("Generating female figure...")
    female_manual = MANUAL_LABELS_PEARSON_FEMALE if args.metric == 'pearson' else None
    fig_female = create_grouped_figure(
        df, skeleton_female, model_prefix='long', gender='female',
        output_path=_paths(f'grouped_by_body_part_female{output_suffix}'),
        manual_labels_by_group=female_manual,
        fixed_max_range=shared_skeleton_range,
    )
    plt.close(fig_female)

    if args.metric == 'pearson':
        print("\nGenerating all-label Pearson figures...")
        fig_male_all = create_grouped_figure(
            df, skeleton_male, model_prefix='long', gender='male',
            output_path=_paths('grouped_by_body_part_male_all_pearson'),
            manual_labels_by_group=None,
            fixed_max_range=shared_skeleton_range,
            show_all_labels=True,
        )
        plt.close(fig_male_all)

        fig_female_all = create_grouped_figure(
            df, skeleton_female, model_prefix='long', gender='female',
            output_path=_paths('grouped_by_body_part_female_all_pearson'),
            manual_labels_by_group=None,
            fixed_max_range=shared_skeleton_range,
            show_all_labels=True,
        )
        plt.close(fig_female_all)

    print("\nDone!")


if __name__ == '__main__':
    main()
