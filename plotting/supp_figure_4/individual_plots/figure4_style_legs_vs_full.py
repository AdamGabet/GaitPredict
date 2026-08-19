"""Figure-4-style grouped boxplot: legs-only (lower_body9) vs full body (long_seq).

Data sources (de-identified aggregates in the repo results/ dir):
  Legs only: lower_body9_full_pearson.csv      (lower_body9_full)
  Full body:  gait_vs_confounders_pearson.csv   (all_but_room_a_full_results)
Both use Age_Gender_BMI_height_VAT baseline — deltas are directly comparable.
Significance filter: full-body FDR q<0.1, is_better, score>0, score_pvalue<0.05.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from publication_colors import merge_systems  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR   = Path(__file__).parent
RESULTS   = Path(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")).resolve()
LEGS_CSV  = RESULTS / "lower_body9_full_pearson.csv"
FULL_CSV  = RESULTS / "gait_vs_confounders_pearson.csv"

FDR_THRESHOLD = 0.1
EXCLUDE_SYSTEMS = {"wearable_weekly", "wearable_monthly", "subject", "proteomics", "metabolites"}
EXCLUDE_LABELS  = {"height", "frailty_height", "weight"}

GROUPS = [
    ("Legs only",  "legs_delta",  "#2166ac"),
    ("Full body",  "full_delta",  "#555555"),
]
GROUP_LABELS = [g[0] for g in GROUPS]
GROUP_COLS   = [g[1] for g in GROUPS]
GROUP_COLORS = [g[2] for g in GROUPS]

SYSTEM_PRETTY = {
    "blood_tests_lipids":    "Blood Lipids",
    "body_composition":      "Body Composition",
    "bone_density":          "Bone Density",
    "cardiovascular_system": "Cardiovascular",
    "frailty":               "Frailty",
    "glycemic_status":       "Glycemic",
    "hematopoietic":         "Hematopoietic",
    "high_level_diet_with_stage": "Diet",
    "immune_system":         "Immune",
    "lifestyle_group":       "Lifestyle",
    "liver":                 "Liver",
    "medical_conditions":    "Medical",
    "mental":                "Mental",
    "nightingale":           "Nightingale",
    "renal_function":        "Renal",
    "sleep_group":           "Sleep",
}


def _load_ensemble(path: Path, model_col_name: str, keep_model: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = merge_systems(df)
    df = df[
        (df["model"] == keep_model)
        & (df["sub_model"] == "ensemble")
        & (df["score_type"] == "pearson_r")
        & (df["gender"].isin(["male", "female"]))
        & (~df["system"].isin(EXCLUDE_SYSTEMS))
        & (~df["label"].isin(EXCLUDE_LABELS))
    ].copy()
    df = df.rename(columns={"delta": model_col_name, "score": f"{model_col_name}_score"})
    return df[["system", "label", "gender", model_col_name, f"{model_col_name}_score",
               "wilcox_pvalue_fdr", "score_pvalue", "description"]]


def _load_data() -> pd.DataFrame:
    full = _load_ensemble(FULL_CSV, "full_delta", keep_model="long_seq")
    legs = _load_ensemble(LEGS_CSV, "legs_delta", keep_model="lower_body9")

    # significance filter on full body — match figure_4 exactly
    sig = full[
        (full["wilcox_pvalue_fdr"] < FDR_THRESHOLD)
        & (full["full_delta"] > 0)
        & (full["full_delta_score"] > 0)
        & (full["score_pvalue"] < 0.05)
    ][["system", "label", "gender"]].copy()

    merged = sig.merge(
        full[["system", "label", "gender", "full_delta"]],
        on=["system", "label", "gender"], how="left",
    ).merge(
        legs[["system", "label", "gender", "legs_delta"]],
        on=["system", "label", "gender"], how="left",
    )
    systems_with_legs = merged.groupby("system")["legs_delta"].apply(lambda x: x.notna().any())
    merged = merged[merged["system"].isin(systems_with_legs[systems_with_legs].index)]
    return merged


def _sys_order(df: pd.DataFrame) -> list[str]:
    return (
        df.groupby("system")["full_delta"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _draw_panel(ax, sub: pd.DataFrame, sys_order: list[str],
                rng, panel_letter: str, title: str) -> None:
    n_boxes = len(GROUPS)
    box_w   = 0.16
    group_w = n_boxes * box_w + 0.22

    for si, sys in enumerate(sys_order):
        feat_rows = sub[sub["system"] == sys]
        cx = si * group_w
        for bi, (lbl, col, color) in enumerate(GROUPS):
            bx = cx + (bi - (n_boxes - 1) / 2.0) * box_w
            vals = feat_rows[col].dropna().values
            if len(vals) == 0:
                continue
            if len(vals) == 1:
                ax.scatter([bx], vals, color=color, s=10, alpha=0.6, lw=0, zorder=5)
                ax.plot([bx - box_w / 2, bx + box_w / 2], [vals[0], vals[0]],
                        color=color, lw=1.4, zorder=4)
                continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            iqr = q3 - q1
            wlo = max(vals.min(), q1 - 1.5 * iqr)
            whi = min(vals.max(), q3 + 1.5 * iqr)
            ax.add_patch(plt.Rectangle(
                (bx - box_w / 2, q1), box_w, q3 - q1,
                facecolor=(*mcolors.to_rgb(color), 0.18),
                edgecolor=color, lw=0.9, zorder=3,
            ))
            ax.plot([bx - box_w / 2, bx + box_w / 2], [med, med],
                    color=color, lw=1.4, zorder=4)
            ax.plot([bx, bx], [wlo, q1], color=color, lw=0.7, zorder=2)
            ax.plot([bx, bx], [q3, whi], color=color, lw=0.7, zorder=2)
            jitter = rng.uniform(-box_w * 0.3, box_w * 0.3, size=len(vals))
            ax.scatter(bx + jitter, vals, color=color, s=4, alpha=0.35,
                       lw=0, zorder=5)

    n_sys = len(sys_order)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xticks([si * group_w for si in range(n_sys)])
    ax.set_xticklabels(
        [SYSTEM_PRETTY.get(s, s.replace("_", " ").title()) for s in sys_order],
        rotation=40, ha="right", fontsize=7.5,
    )
    ax.set_ylabel("Pearson r improvement\n(Δ from baseline)", fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-group_w * 0.6, (n_sys - 1) * group_w + group_w * 0.6)
    ax.text(-0.06, 1.02, panel_letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")


def plot(df: pd.DataFrame) -> None:
    sys_order = _sys_order(df)
    rng = np.random.default_rng(42)
    n_sys = len(sys_order)

    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_sys * 0.75), 6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.22,
                        wspace=0.28)

    for ax, (gender_key, letter, title) in zip(
        axes, [("male", "a", "Male"), ("female", "b", "Female")]
    ):
        sub = df[df["gender"] == gender_key].copy()
        panel_sys = [s for s in sys_order if s in sub["system"].values]
        _draw_panel(ax, sub, panel_sys, rng, letter, title)

    legend_handles = [
        Patch(facecolor=(*mcolors.to_rgb(c), 0.25), edgecolor=c, lw=1.0, label=lbl)
        for lbl, _, c in GROUPS
    ]
    fig.legend(handles=legend_handles, ncol=2, fontsize=8, framealpha=0.85,
               loc="upper center", bbox_to_anchor=(0.5, 0.99),
               bbox_transform=fig.transFigure)
    fig.suptitle("Lower-body model vs full-body — Δr vs confounders by system",
                 fontsize=11, fontweight="bold", y=1.06)

    stem = OUT_DIR / "figure4_style_legs_vs_full"
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}.{{pdf,png}}")


def main() -> None:
    df = _load_data()
    print(f"Labels after filter: {len(df)} rows, {df['system'].nunique()} systems")
    plot(df)


if __name__ == "__main__":
    main()
