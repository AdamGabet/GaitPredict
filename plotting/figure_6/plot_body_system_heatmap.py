"""
Panel A: Body system x body group heatmap.

Each heatmap cell is the mean of top-K phenotype-level importance scores
within a body system for a given body group.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from publication_colors import SYSTEM_RENAME_DICT, merge_systems
sys.path.insert(0, os.path.dirname(__file__))
from plot_grouped_by_body_part import (
    WALKING_ACTIVITIES,
    compute_group_importance_averaged,
    should_include_label,
)

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')


BODY_GROUPS = ["head", "arms", "torso", "legs"]
FONT_SCALE = 1.6

SYSTEM_NAME_MAP = {
    "age_gender_bmi_VAT": "age_gender_bmi_VAT",
    "anthropometric_group": "anthropometric_group",
    "blood_tests_lipids_group": "blood_tests_lipids",
    "body_composition_group": "body_composition",
    "bone_density_group": "bone_density",
    "cardiovascular_system_group": "cardiovascular_system",
    "exercise_label": "exercise_label",
    "frailty_group": "frailty",
    "hematopoietic_group": "hematopoietic",
    "lifestyle_group": "lifestyle_group",
    "lifestyle_ordinal": "lifestyle_ordinal",
    "liver_group": "liver",
    "mental_group": "mental",
    "metabolites_annotated_group": "metabolites_annotated",
    "nightingale_group": "nightingale",
    "sleep_group_group": "sleep_group",
    "subject_group": "subject",
    "wearable_monthly_group": "wearable_monthly",
}

EXCLUDE_SYSTEMS = {"subject", "age_gender_bmi_VAT", "wearable_monthly", "wearable_weekly"}


def _canonical_system_name(system_name):
    return SYSTEM_NAME_MAP.get(system_name, system_name)


def _system_display_name(system_name):
    if system_name in SYSTEM_RENAME_DICT:
        return SYSTEM_RENAME_DICT[system_name]
    return system_name.replace("_", " ").title()


def _infer_score_type(csv_path):
    lower_name = os.path.basename(csv_path).lower()
    if "pearson" in lower_name:
        return "pearson_r"
    return "r2"


def _resolve_cmap(cmap_name):
    if cmap_name in {"radar_dark_blue", "radar_v2_blue"}:
        return LinearSegmentedColormap.from_list("radar_v2_blue", ["#ffffff", "#393b79"])
    return cmap_name


def _save(fig, paths):
    for path in paths if isinstance(paths, list) else [paths]:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")


def load_and_prepare(csv_path, score_type):
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["score_type"] == score_type].copy()
    df["system"] = df["system"].map(_canonical_system_name)
    df = merge_systems(df)
    return df


def build_label_importance_table(df, gender, model_prefix):
    records = []
    df_gender = df[df["gender"] == gender].copy()

    system_label_pairs = (
        df_gender[["system", "label"]]
        .drop_duplicates()
        .sort_values(["system", "label"])
        .itertuples(index=False, name=None)
    )

    for system, label in system_label_pairs:
        if system in EXCLUDE_SYSTEMS:
            continue
        if not should_include_label(label):
            continue

        df_pair = df_gender[(df_gender["system"] == system) & (df_gender["label"] == label)]
        importance = compute_group_importance_averaged(
            df_pair, label=label, gender=gender,
            model_prefix=model_prefix, activities=WALKING_ACTIVITIES,
        )
        if not importance:
            continue

        rec = {"system": system, "label": label}
        for group in BODY_GROUPS:
            rec[group] = importance.get(group, np.nan)
        records.append(rec)

    if not records:
        return pd.DataFrame(columns=["system", "label"] + BODY_GROUPS)
    return pd.DataFrame(records)


def aggregate_topk_by_system(label_importance_df, top_k):
    systems = sorted(label_importance_df["system"].unique().tolist())
    if not systems:
        return pd.DataFrame(columns=BODY_GROUPS, dtype=float)

    rows = []
    for system in systems:
        row = {"system": system}
        df_sys = label_importance_df[label_importance_df["system"] == system]
        for group in BODY_GROUPS:
            vals = df_sys[group].dropna().sort_values(ascending=False).head(top_k)
            row[group] = float(vals.mean()) if len(vals) > 0 else np.nan
        rows.append(row)

    heatmap_df = pd.DataFrame(rows).set_index("system")
    heatmap_df[BODY_GROUPS] = heatmap_df[BODY_GROUPS].apply(pd.to_numeric, errors="coerce")
    heatmap_df["row_max"] = heatmap_df.max(axis=1)
    heatmap_df = heatmap_df.sort_values("row_max", ascending=False).drop(columns=["row_max"])
    return heatmap_df


def normalize_rows_to_unit_interval(heatmap_df):
    out = heatmap_df.copy()
    for idx in out.index:
        row = out.loc[idx, BODY_GROUPS].astype(float)
        mn = row.min(skipna=True)
        mx = row.max(skipna=True)
        if pd.isna(mn) or pd.isna(mx):
            out.loc[idx, BODY_GROUPS] = np.nan
            continue
        if mx == mn:
            out.loc[idx, BODY_GROUPS] = 0.5
        else:
            out.loc[idx, BODY_GROUPS] = (row - mn) / (mx - mn)
    return out


def compute_row_dominance_scores(heatmap_df):
    dominance = {}
    for idx in heatmap_df.index:
        row_vals = heatmap_df.loc[idx, BODY_GROUPS].dropna().astype(float).sort_values(ascending=False).values
        if len(row_vals) == 0:
            dominance[idx] = -np.inf
        elif len(row_vals) == 1:
            dominance[idx] = row_vals[0]
        else:
            dominance[idx] = row_vals[0] - row_vals[1]
    return pd.Series(dominance)


def sort_by_dominance(heatmap_df):
    if heatmap_df.empty:
        return heatmap_df
    dominance = compute_row_dominance_scores(heatmap_df)
    order = dominance.sort_values(ascending=False).index.tolist()
    return heatmap_df.reindex(order)


def _highlight_row_max_cells(ax, values):
    for i in range(values.shape[0]):
        row = values[i, :]
        if np.all(np.isnan(row)):
            continue
        max_j = int(np.nanargmax(row))
        rect = patches.Rectangle(
            (max_j - 0.5, i - 0.5), 1, 1,
            linewidth=2.2, edgecolor="black", facecolor="none",
        )
        ax.add_patch(rect)


def _annotate_cells(ax, values, small_font=False):
    fontsize = (8.0 if small_font else 9.0) * FONT_SCALE
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            txt = "NA" if np.isnan(val) else f"{val:.2f}"
            text_color = "white" if (not np.isnan(val) and val >= 0.72) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fontsize, color=text_color)


def plot_single_heatmap(heatmap_df, gender, top_k, output_path, annotate, cmap_name):
    if heatmap_df.empty:
        fig, ax = plt.subplots(figsize=(8.8, 4.0))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for {gender} (top-{top_k})",
                ha="center", va="center", fontsize=13 * FONT_SCALE, fontweight="bold")
        plt.tight_layout()
        _save(fig, output_path)
        plt.close(fig)
        return

    fig_h = max(5.0, 0.42 * len(heatmap_df) + 2.8)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))

    values = heatmap_df[BODY_GROUPS].to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap_name, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(BODY_GROUPS)))
    ax.set_xticklabels([g.title() for g in BODY_GROUPS], fontsize=12 * FONT_SCALE, fontweight="bold")

    display_rows = [_system_display_name(s) for s in heatmap_df.index.tolist()]
    ax.set_yticks(np.arange(len(display_rows)))
    ax.set_yticklabels(display_rows, fontsize=11 * FONT_SCALE)

    ax.set_title(gender.title(), fontsize=14 * FONT_SCALE, fontweight="bold", pad=14)

    if annotate:
        _annotate_cells(ax, values, small_font=False)
    _highlight_row_max_cells(ax, values)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Importance within System (0-1)", fontsize=11 * FONT_SCALE)

    plt.tight_layout()
    _save(fig, output_path)
    plt.close(fig)


def plot_combined_heatmap(male_df, female_df, top_k, output_path, annotate, male_cmap_name, female_cmap_name):
    if male_df.empty and female_df.empty:
        fig, ax = plt.subplots(figsize=(14.5, 4.5))
        ax.axis("off")
        ax.text(0.5, 0.5, f"No data for combined heatmap (top-{top_k})",
                ha="center", va="center", fontsize=13 * FONT_SCALE, fontweight="bold")
        plt.tight_layout()
        _save(fig, output_path)
        plt.close(fig)
        return

    combined_index = []
    for idx in male_df.index.tolist() + female_df.index.tolist():
        if idx not in combined_index:
            combined_index.append(idx)

    if combined_index:
        dom_order_df = pd.DataFrame(index=combined_index, columns=BODY_GROUPS, dtype=float)
        for idx in combined_index:
            if idx in male_df.index:
                dom_order_df.loc[idx, BODY_GROUPS] = male_df.loc[idx, BODY_GROUPS].astype(float).values
            if idx in female_df.index:
                female_row = female_df.loc[idx, BODY_GROUPS].astype(float).values
                if dom_order_df.loc[idx, BODY_GROUPS].isna().all():
                    dom_order_df.loc[idx, BODY_GROUPS] = female_row
                else:
                    current = dom_order_df.loc[idx, BODY_GROUPS].astype(float).values
                    dom_order_df.loc[idx, BODY_GROUPS] = np.nanmean(np.vstack([current, female_row]), axis=0)
        dom_order_df = sort_by_dominance(dom_order_df)
        combined_index = dom_order_df.index.tolist()

    male_aligned = male_df.reindex(combined_index)
    female_aligned = female_df.reindex(combined_index)

    values_m = male_aligned[BODY_GROUPS].to_numpy(dtype=float)
    values_f = female_aligned[BODY_GROUPS].to_numpy(dtype=float)

    fig_h = max(6.0, 0.42 * len(combined_index) + 2.8)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, fig_h), sharey=True)

    im_m = axes[0].imshow(values_m, cmap=male_cmap_name, aspect="auto", vmin=0, vmax=1)
    im_f = axes[1].imshow(values_f, cmap=female_cmap_name, aspect="auto", vmin=0, vmax=1)

    for ax, vals, gender_name in zip(axes, [values_m, values_f], ["Male", "Female"]):
        ax.set_xticks(np.arange(len(BODY_GROUPS)))
        ax.set_xticklabels([g.title() for g in BODY_GROUPS], fontsize=11 * FONT_SCALE, fontweight="bold")
        ax.set_title(gender_name, fontsize=13 * FONT_SCALE, fontweight="bold", pad=8)
        if annotate:
            _annotate_cells(ax, vals, small_font=True)
        _highlight_row_max_cells(ax, vals)

    display_rows = [_system_display_name(s) for s in combined_index]
    axes[0].set_yticks(np.arange(len(display_rows)))
    axes[0].set_yticklabels(display_rows, fontsize=10 * FONT_SCALE)

    cbar_m = fig.colorbar(im_m, ax=axes[0], fraction=0.046, pad=0.02)
    cbar_m.set_label("Male (0-1)", fontsize=9 * FONT_SCALE)
    cbar_f = fig.colorbar(im_f, ax=axes[1], fraction=0.046, pad=0.02)
    cbar_f.set_label("Female (0-1)", fontsize=9 * FONT_SCALE)

    fig.subplots_adjust(left=0.20, right=0.92, top=0.90, bottom=0.06, wspace=0.06)
    _save(fig, output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create body-system heatmaps from masking ablation results.")
    parser.add_argument(
        "--csv-path", type=str,
        default=os.path.join(_RESULTS_DIR, 'masking_ablation_pearson.csv'),
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(os.path.dirname(__file__), 'output', 'figure6a'),
    )
    parser.add_argument("--model-prefix", type=str, default="long")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--cmap-male", type=str, default="Blues")
    parser.add_argument("--cmap-female", type=str, default="Oranges")
    parser.add_argument("--cmap-both", type=str, default="radar_dark_blue")
    parser.add_argument("--score-type", type=str, choices=["auto", "r2", "pearson_r"], default="auto")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    score_type = args.score_type
    if score_type == "auto":
        score_type = _infer_score_type(args.csv_path)
    print(f"Using score_type={score_type}")

    df = load_and_prepare(args.csv_path, score_type=score_type)

    male_label_importance = build_label_importance_table(df, gender="male", model_prefix=args.model_prefix)
    female_label_importance = build_label_importance_table(df, gender="female", model_prefix=args.model_prefix)
    both_label_importance = pd.concat([male_label_importance, female_label_importance], ignore_index=True)

    male_heatmap = normalize_rows_to_unit_interval(aggregate_topk_by_system(male_label_importance, top_k=args.top_k))
    female_heatmap = normalize_rows_to_unit_interval(aggregate_topk_by_system(female_label_importance, top_k=args.top_k))
    both_heatmap = normalize_rows_to_unit_interval(aggregate_topk_by_system(both_label_importance, top_k=args.top_k))

    both_heatmap = sort_by_dominance(both_heatmap)
    both_order = both_heatmap.index.tolist()
    male_heatmap = male_heatmap.reindex(both_order)
    female_heatmap = female_heatmap.reindex(both_order)

    male_cmap = _resolve_cmap(args.cmap_male)
    female_cmap = _resolve_cmap(args.cmap_female)
    both_cmap = _resolve_cmap(args.cmap_both)

    def _paths(stem):
        return [
            os.path.join(args.output_dir, f"{stem}.png"),
            os.path.join(args.output_dir, f"{stem}.pdf"),
        ]

    stem = f"body_system_heatmap_top{args.top_k}"

    plot_single_heatmap(male_heatmap, gender="male", top_k=args.top_k,
                        output_path=_paths(f"{stem}_male"), annotate=args.annotate, cmap_name=male_cmap)
    plot_single_heatmap(female_heatmap, gender="female", top_k=args.top_k,
                        output_path=_paths(f"{stem}_female"), annotate=args.annotate, cmap_name=female_cmap)
    plot_combined_heatmap(male_heatmap, female_heatmap, top_k=args.top_k,
                          output_path=_paths(f"{stem}_male_female"), annotate=args.annotate,
                          male_cmap_name=male_cmap, female_cmap_name=female_cmap)
    plot_single_heatmap(both_heatmap, gender="both sexes", top_k=args.top_k,
                        output_path=_paths(f"{stem}_both"), annotate=args.annotate, cmap_name=both_cmap)


if __name__ == "__main__":
    main()
