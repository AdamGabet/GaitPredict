"""
Supplementary Figure 3 — FDR-significant longitudinal next-visit improvements
(ensemble + gait vs previous-visit + age/BMI baseline), Pearson r.

Reads one de-identified, plot-ready CSV (results/longitudinal_supp_fig3_pearson.csv)
carrying per (system, label, gender): bar scores + 10 baseline seeds + 10 gait seeds.
Writes PDF + PNG to plotting/supp_figure_3/output/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIRNAME = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(DIRNAME, "..", "..", "results")
OUTPUT_DIR = os.path.join(DIRNAME, "output")
DATA_CSV = os.path.join(RESULTS_DIR, "longitudinal_supp_fig3_pearson.csv")

GENDER_COLORS = {
    "male": "#2F66B3",
    "female": "#C23A64",
}
BASELINE_BAR = "#C7CDD3"
FIG_WIDTH_CM = 9.0
FONT_SIZE = 7.5

LABEL_DISPLAY = {
    "liver_sound_speed": "Liver sound speed",
    "bt__basophils_abs": "Basophils (abs)",
    "sitting_blood_pressure_systolic": "Sitting systolic BP",
    "bt__wbc": "WBC",
    "r_abi": "ABI (ankle-brachial index)",
    "depressed_whole_week": "Depressed whole week",
    "nervous_person": "Nervous person",
    "nerves_anxiety_tension_depression_doctor": "Anxiety/tension (doctor)",
}


def pretty_label(label: str) -> str:
    if label in LABEL_DISPLAY:
        return LABEL_DISPLAY[label]
    return label.replace("_", " ").replace("bt  ", "").title()


def row_display_name(row: pd.Series) -> str:
    return f"{pretty_label(row['label'])} ({row['gender']})"


def seed_arrays(row: pd.Series, prefix: str) -> np.ndarray:
    cols = sorted(
        [c for c in row.index if c.startswith(prefix)],
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    vals = row[cols].to_numpy(dtype=float)
    return vals[~np.isnan(vals)]


def create_paired_barplot(plot_df: pd.DataFrame, out_png: str, out_pdf: str, xlabel: str) -> None:
    plot_df = plot_df.sort_values("delta", ascending=True).reset_index(drop=True)
    n = len(plot_df)
    bar_h = 0.22
    pair_gap = 0.06
    row_gap = 0.35

    y_positions = []
    y = 0.0
    for _ in range(n):
        y_positions.append(y + bar_h)
        y += 2 * bar_h + pair_gap + row_gap

    fig_height = max(2.8, y * 0.28 + 0.8)
    fig_width = FIG_WIDTH_CM / 2.54
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    matplotlib.rcParams.update({"font.size": FONT_SIZE})

    x_max = 0.0
    for i, row in plot_df.iterrows():
        yc = y_positions[i]
        gait_color = GENDER_COLORS.get(row["gender"], "#4A4A4A")
        baseline_val = float(row["baseline_score"])
        gait_val = float(row["score"])
        x_max = max(x_max, baseline_val, gait_val)

        ax.barh(yc + bar_h / 2 + pair_gap / 2, baseline_val, height=bar_h,
                color=BASELINE_BAR, edgecolor="none", zorder=2)
        ax.barh(yc - bar_h / 2 - pair_gap / 2, gait_val, height=bar_h,
                color=gait_color, edgecolor="none", zorder=2)

        baseline_seeds = seed_arrays(row, "baseline_seed_")
        gait_seeds = seed_arrays(row, "gait_seed_")
        if baseline_seeds.size and gait_seeds.size:
            ax.scatter(
                baseline_seeds,
                np.full(len(baseline_seeds), yc + bar_h / 2 + pair_gap / 2),
                s=8, color="#888888", alpha=0.55, zorder=3, linewidths=0,
            )
            ax.scatter(
                gait_seeds,
                np.full(len(gait_seeds), yc - bar_h / 2 - pair_gap / 2),
                s=8, color=gait_color, alpha=0.55, zorder=3, linewidths=0,
            )

        row_x = max(baseline_val, gait_val)
        ax.text(row_x + 0.018, yc, f"+{row['delta']:.2f}",
                va="center", ha="left", fontsize=FONT_SIZE, color="#333333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row_display_name(r) for _, r in plot_df.iterrows()])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE)
    ax.set_xlim(0, min(1.0, x_max + 0.14))
    ax.axvline(0, color="#DDDDDD", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.5)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=BASELINE_BAR, label="Prev visit + age/BMI")]
    for g in ["male", "female"]:
        if g in plot_df["gender"].values:
            legend_handles.append(Patch(facecolor=GENDER_COLORS[g], label=f"+ gait ({g})"))
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 0.02),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=FONT_SIZE - 0.5,
    )

    left = min(0.55, 0.32 + 0.012 * max(len(plot_df), 1))
    fig.subplots_adjust(left=left, right=0.84, top=0.96, bottom=0.20)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def main() -> None:
    plot_df = pd.read_csv(DATA_CSV)
    create_paired_barplot(
        plot_df,
        os.path.join(OUTPUT_DIR, "longitudinal_significant_improvements_pearson.png"),
        os.path.join(OUTPUT_DIR, "longitudinal_significant_improvements_pearson.pdf"),
        "Pearson r (next visit)",
    )


if __name__ == "__main__":
    main()
