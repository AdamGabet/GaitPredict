from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent.parent.parent / "results"
OUT_DIR = HERE / "panels"
SENS_CSV = RESULTS_DIR / "clinical_sensitivity_top10_table.csv"
UTILITY_CSV = RESULTS_DIR / "clinical_utility_by_gender_table.csv"
SEED_AUC_CSV = RESULTS_DIR / "clinical_scenario_seed_auc.csv"
MODEL = "long_seq"
BASELINE_MODEL = "Age_Gender_BMI_height"

MALE = "#2F66B3"
FEMALE = "#C23A64"
BASELINE = "#8A8A8A"
GRID = "#DADADA"
TEXT = "#333333"

SEX_COLORS = {"male": MALE, "female": FEMALE}
SEX_TITLES = {"male": "Male", "female": "Female"}
SEX_N = {"male": 1607, "female": 1721}

LABEL_RENAME = {
    "Gastrointestinal_any": "Gastrointestinal Disorders",
    "Sleep_disorder_any": "Sleep Disorders",
}


def _pretty_label(label: str) -> str:
    return LABEL_RENAME.get(str(label), str(label))


def _system_title(system: str) -> str:
    return "MEDICATIONS" if system == "medications" else "MEDICAL CONDITIONS"


def _ordered_sections(df: pd.DataFrame, sort_col: str) -> list[tuple[str, pd.DataFrame]]:
    sections = []
    for system in ["medical_conditions", "medical_conditions_grouped", "medications"]:
        part = df[df["system"].eq(system)].copy()
        if part.empty:
            continue
        title = _system_title(system)
        if sections and sections[-1][0] == title:
            prev_title, prev = sections[-1]
            sections[-1] = (prev_title, pd.concat([prev, part], ignore_index=True))
        else:
            sections.append((title, part))
    return [(title, part.sort_values(sort_col, ascending=False)) for title, part in sections]


def _layout_rows(sections: list[tuple[str, pd.DataFrame]]) -> tuple[list[tuple[float, pd.Series]], dict[str, float]]:
    rows = []
    headers = {}
    y = 0.0
    row_step = 0.68
    section_gap = 0.44
    for title, part in sections:
        headers[title] = y - 0.38
        for _, row in part.iterrows():
            rows.append((y, row))
            y += row_step
        y += section_gap
    return rows, headers


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", facecolor="white", edgecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=450, facecolor="white", edgecolor="white")
    plt.close(fig)


def _plot_seed_distribution(ax, seed_values, y_center, color, x_jitter, y_jitter=0.08, point_size=4, point_alpha=0.24):
    if seed_values is None or len(seed_values) == 0:
        return
    x_noise = np.random.uniform(-x_jitter, x_jitter, len(seed_values))
    y_noise = np.random.uniform(-y_jitter, y_jitter, len(seed_values))
    ax.scatter(
        seed_values + x_noise,
        y_center + y_noise,
        s=point_size,
        color=color,
        alpha=point_alpha,
        linewidths=0,
        zorder=1,
    )


_SEED_AUC_DF = None


def load_seed_auc_values(system: str, endpoint: str, gender: str):
    """Per-seed AUC values from the precomputed de-identified CSV (replaces reading
    per-target metrics.csv from the cluster)."""
    global _SEED_AUC_DF
    if _SEED_AUC_DF is None:
        _SEED_AUC_DF = pd.read_csv(SEED_AUC_CSV)
    sub = _SEED_AUC_DF[
        (_SEED_AUC_DF["system"] == system)
        & (_SEED_AUC_DF["endpoint"] == endpoint)
        & (_SEED_AUC_DF["gender"] == gender)
    ].sort_values("seed_idx")
    if sub.empty:
        return None, None
    return np.maximum(sub["baseline_auc"].values, 0.5), sub["gait_auc"].values


def auc_endpoint_keys(sens: pd.DataFrame, utility: pd.DataFrame, gender: str) -> pd.DataFrame:
    individual_medical = sens["system"].eq("medical_conditions") & ~sens["endpoint"].isin(["Insomnia", "Sleep Apnea", "AHI"])
    grouped_sleep = sens["system"].eq("medical_conditions_grouped") & sens["endpoint"].eq("Sleep_disorder_any")
    keys = sens[
        sens["gender"].eq(gender)
        & (individual_medical | grouped_sleep)
    ][
        [
            "endpoint",
            "system",
            "display_endpoint",
            "sensitivity_base_top10",
            "sensitivity_gait_top10",
            "delta_sensitivity_top10",
            "fdr_significant_gain_q_lt_0p1",
        ]
    ].copy()
    keys["display_endpoint"] = keys["display_endpoint"].map(_pretty_label)
    auc = utility[utility["gender"].eq(gender)].groupby(["endpoint", "system"], as_index=False).agg(
        baseline_auc_raw=("baseline_auc_selected", "mean"),
        gait_auc=("gait_auc_selected", "mean"),
    )
    auc["baseline_auc"] = auc["baseline_auc_raw"].clip(lower=0.5)
    auc["delta_auc"] = auc["gait_auc"] - auc["baseline_auc"]
    out = keys.merge(auc, on=["endpoint", "system"], how="left")
    return out.dropna(subset=["baseline_auc", "gait_auc"])[lambda d: d["gait_auc"] > 0.52].copy()


def plot_sensitivity_panel(sens: pd.DataFrame, utility: pd.DataFrame, gender: str) -> None:
    color = SEX_COLORS[gender]
    sub = auc_endpoint_keys(sens, utility, gender)
    sub["plot_label"] = np.where(
        sub["fdr_significant_gain_q_lt_0p1"],
        sub["display_endpoint"].astype(str) + " *",
        sub["display_endpoint"].astype(str),
    )
    sections = _ordered_sections(sub, "delta_sensitivity_top10")
    rows, headers = _layout_rows(sections)

    fig_h = max(3.9, 0.30 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(4.8, fig_h))
    fig.patch.set_facecolor("white")
    max_sens = max(
        sub["sensitivity_base_top10"].max(skipna=True),
        sub["sensitivity_gait_top10"].max(skipna=True),
    ) * 100
    x_max = max(10, np.ceil((max_sens + 2) / 5) * 5)

    for y, row in rows:
        base = row["sensitivity_base_top10"] * 100
        gait = row["sensitivity_gait_top10"] * 100
        delta_pp = row["delta_sensitivity_top10"] * 100
        lw = 1.5
        alpha = 0.92

        seed_rows = utility[
            utility["gender"].eq(gender)
            & utility["endpoint"].eq(row["endpoint"])
            & utility["system"].eq(row["system"])
        ]
        base_seed_pct = seed_rows["sens_base_top10"].values * 100
        gait_seed_pct = seed_rows["sens_gait_top10"].values * 100
        _plot_seed_distribution(ax, base_seed_pct, y, BASELINE, x_jitter=0.8, point_size=3, point_alpha=0.18)
        _plot_seed_distribution(ax, gait_seed_pct, y, color, x_jitter=0.8, point_size=4, point_alpha=0.26)

        ax.plot([base, gait], [y, y], color=color, lw=lw, alpha=alpha, zorder=2)
        ax.scatter([base], [y], s=18, facecolors="white", edgecolors=BASELINE, linewidths=0.9, zorder=3)
        ax.scatter([gait], [y], s=28, color=color, edgecolors="white", linewidths=0.45, zorder=4)
        ax.text(-1.0, y, row["plot_label"], ha="right", va="center", fontsize=10.1, color=TEXT)
        ax.text(
            1.025,
            y,
            f"{delta_pp:+.1f} pp",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9.9,
            color=color,
            fontweight="bold",
            clip_on=False,
        )

    for title, y in headers.items():
        ax.text(-1.0, y, title, ha="right", va="center", fontsize=10.5, color="black", fontweight="bold")

    if rows:
        ax.text(
            1.025,
            min(headers.values()),
            "ΔSensitivity",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9.9,
            color="#666666",
            fontweight="bold",
            clip_on=False,
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.9, rows[-1][0] + 0.45)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, x_max + 0.1, 10 if x_max > 25 else 5))
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.85)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", labelsize=10.1, colors="#666666", width=0.5)
    ax.set_xlabel("Sensitivity among top-risk 10% referred (%)", fontsize=10.5, color="#666666")
    ax.set_title(f"{SEX_TITLES[gender]} (n = {SEX_N[gender]:,})", loc="left", fontsize=12.4, fontweight="bold", color=TEXT, pad=4)

    legend = [
        Line2D([0], [0], marker="o", markersize=4.4, markerfacecolor="white", markeredgecolor=BASELINE, lw=0, label="Age, BMI, Height"),
        Line2D([0], [0], marker="o", markersize=4.8, markerfacecolor=color, markeredgecolor="white", lw=0, label="Age, BMI, Height & Gait"),
    ]
    legend_y = 1.07 if gender == "female" else 1.05
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.55, legend_y),
        frameon=False,
        fontsize=9.9,
        handletextpad=0.35,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.45, right=0.78, top=0.84, bottom=0.09)
    _save(fig, f"scenario_sensitivity_top10_{gender}")


def plot_auc_panel(sens: pd.DataFrame, utility: pd.DataFrame, gender: str) -> None:
    color = SEX_COLORS[gender]
    plot_df = auc_endpoint_keys(sens, utility, gender)
    sections = _ordered_sections(plot_df, "delta_sensitivity_top10")
    rows, headers = _layout_rows(sections)

    fig_h = max(3.9, 0.30 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(4.6, fig_h))
    fig.patch.set_facecolor("white")

    for y, row in rows:
        base = row["baseline_auc"]
        gait = row["gait_auc"]
        base_seed_vals, gait_seed_vals = load_seed_auc_values(row["system"], row["endpoint"], gender)
        _plot_seed_distribution(ax, base_seed_vals, y, BASELINE, x_jitter=0.003, point_size=3, point_alpha=0.18)
        _plot_seed_distribution(ax, gait_seed_vals, y, color, x_jitter=0.003, point_size=4, point_alpha=0.26)
        ax.plot([base, gait], [y, y], color=color, lw=1.2, alpha=0.78, zorder=2)
        ax.scatter([base], [y], s=18, facecolors="white", edgecolors=BASELINE, linewidths=0.9, zorder=3)
        ax.scatter([gait], [y], s=30, color=color, edgecolors="white", linewidths=0.45, zorder=4)
        ax.text(0.492, y, row["display_endpoint"], ha="right", va="center", fontsize=10.5, color=TEXT)
        ax.text(0.716, y, f"+{row['delta_auc']:.3f}", ha="left", va="center", fontsize=10.0, color=color, fontweight="bold")

    for title, y in headers.items():
        ax.text(0.492, y, title, ha="right", va="center", fontsize=10.8, color="black", fontweight="bold")

    if rows:
        ax.text(0.716, min(headers.values()), "ΔAUC", ha="left", va="center", fontsize=10.0, color="#666666", fontweight="bold")

    ax.set_xlim(0.46, 0.755)
    ax.set_ylim(-0.9, rows[-1][0] + 0.45 if rows else 1)
    ax.invert_yaxis()
    ax.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.85)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", labelsize=10.1, colors="#666666", width=0.5)
    ax.set_xlabel("AUC-ROC", fontsize=10.5, color="#666666")
    ax.set_title(f"{SEX_TITLES[gender]} (n = {SEX_N[gender]:,})", loc="left", fontsize=12.4, fontweight="bold", color=TEXT, pad=4)

    legend = [
        Line2D([0], [0], marker="o", markersize=4.4, markerfacecolor="white", markeredgecolor=BASELINE, lw=0, label="Age, BMI, Height"),
        Line2D([0], [0], marker="o", markersize=4.8, markerfacecolor=color, markeredgecolor="white", lw=0, label="Age, BMI, Height & Gait"),
    ]
    legend_y = 1.07 if gender == "female" else 1.05
    ax.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.55, legend_y),
        frameon=False,
        fontsize=9.9,
        handletextpad=0.35,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.46, right=0.93, top=0.83, bottom=0.09)
    _save(fig, f"auc_sensitivity_significant_top10_{gender}")


def main() -> None:
    np.random.seed(7)
    sens = pd.read_csv(SENS_CSV)
    utility = pd.read_csv(UTILITY_CSV)
    for gender in ["male", "female"]:
        plot_sensitivity_panel(sens, utility, gender)
        plot_auc_panel(sens, utility, gender)
    print(f"Wrote final figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
