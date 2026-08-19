"""
Dumbbell plot for medical conditions and medications (male vs female).
Publication-ready: 9 cm wide, 7.5 pt font, PNG + PDF per gender.
Delta = score - max(baseline_score, 0.5)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.size": 7.5})

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DATA_PATH = os.path.join(_RESULTS_DIR, "medical_conditions_pearson.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

MODEL = "long_seq"
BASELINE_MODEL = "Age_Gender_BMI_height_VAT"
FDR_THRESHOLD = 0.10
MIN_DELTA = 0.02
MIN_SCORE = 0.51
MIN_POSITIVES = 10

N_MEDICAL = 12
N_MEDICATIONS = 5

# Publication figure dimensions: 9 cm wide
FIG_WIDTH_CM = 9.0
FIG_WIDTH_IN = FIG_WIDTH_CM / 2.54
FONT_SIZE = 7.5

MALE_COLOR = "#2F66B3"
FEMALE_COLOR = "#C23A64"
BASELINE_COLOR = "#8A8A8A"
GRID_COLOR = "#DADADA"
TEXT_COLOR = "#333333"

# x-axis layout (in data units)
LABEL_X = 0.492
DELTA_X = 0.718
DELTA_HEADER_X = 0.718
X_LIM_MIN = 0.46
X_LIM_MAX = 0.73

EXCLUDE_SYSTEMS = {"medical_conditions_grouped"}
EXCLUDE_LABELS = {
    "LIPIDMODIFYINGAGENTSCOMBINATIONS",
    "LIPIDMODIFYINGAGENTSPLAIN",
    "urinary_tract_infection",
}

LABEL_RENAME_DICT = {
    "ADRENERGICSINHALANTS": "Adrenergic Inhalants",
    "ANTIDEPRESSANTS": "Antidepressants",
    "ANTIEPILEPTICS": "Antiepileptics",
    "ANTIGLAUCOMAPREPARATIONSANDMIOTICS": "Anti-Glaucoma",
    "ANTIHISTAMINESFORSYSTEMICUSE": "Antihistamines",
    "ANTIMIGRAINEPREPARATIONS": "Anti-Migraine",
    "ASCORBICACIDVITAMINCINCLCOMBINATIONS": "Vitamin C",
    "BLOODGLUCOSELOWERINGDRUGSEXCLINSULINS": "Glucose Lowering",
    "DRUGSAFFECTINGBONESTRUCTUREANDMINERALIZATION": "Bone Structure Drugs",
    "DRUGSFORPEPTICULCERANDGASTROOESOPHAGEALREFLUXDISEASEGORD": "PPI / GERD Drugs",
    "DRUGSUSEDINBENIGNPROSTATICHYPERTROPHY": "BPH Drugs",
    "HORMONEANTAGONISTSANDRELATEDAGENTS": "Hormone Antagonists",
    "HYPNOTICSANDSEDATIVES": "Hypnotics & Sedatives",
    "IRONANTIANEMICPREPARATIONS": "Iron Supplements",
    "OTHERGYNECOLOGICALSinATC": "Gynecological Medications",
    "OTHERPLAINVITAMINPREPARATIONSinATC": "Other Vitamins",
    "PROGESTOGENSEXHORMONESANDMODULATORSOFTHEGENITALSYSTEM": "Progestogens",
    "PSYCHOSTIMULANTSAGENTSUSEDFORADHDANDNOOTROPICS": "ADHD / Nootropics",
    "SELECTIVECALCIUMCHANNELBLOCKERSWITHMAINLYVASCULAREFFECTS": "Calcium Channel Blockers",
    "THYROIDPREPARATIONS": "Thyroid Preparations",
    "UROLOGICALS": "Urologicals",
    "VITAMINAANDDINCLCOMBINATIONSOFTHETWO": "Vitamin A & D",
    "VITAMINBn12nANDFOLICACID": "B12 & Folic Acid",
    "VITAMINBn1nPLAINANDINCOMBINATIONWITHVITAMINBn6nANDBn12n": "B1, B6 & B12",
    "VITAMINKANDOTHERHEMOSTATICS": "Vitamin K / Hemostatics",
}


def clean_label_name(label):
    if str(label).lower() == "ibs":
        return "IBS"
    if label in LABEL_RENAME_DICT:
        return LABEL_RENAME_DICT[label]
    return label.replace("_", " ").title()


# Precomputed, de-identified tables (built on the cluster by build_figure5_data.py;
# replaces reads of raw_predictions.csv and per-target metrics.csv).
_N_POSITIVE_DF = pd.read_csv(os.path.join(_RESULTS_DIR, "medical_conditions_n_positive.csv"))
_SEED_AUC_DF = pd.read_csv(os.path.join(_RESULTS_DIR, "medical_conditions_seed_auc.csv"))


def get_n_positive_cached(system, label, gender):
    sub = _N_POSITIVE_DF[
        (_N_POSITIVE_DF["system"] == system)
        & (_N_POSITIVE_DF["label"] == label)
        & (_N_POSITIVE_DF["gender"] == gender)
    ]
    return int(sub["n_positive"].iloc[0]) if not sub.empty else 0


def load_seed_values(system, label, gender):
    sub = _SEED_AUC_DF[
        (_SEED_AUC_DF["system"] == system)
        & (_SEED_AUC_DF["label"] == label)
        & (_SEED_AUC_DF["gender"] == gender)
    ].sort_values("seed_idx")
    if sub.empty:
        return None, None
    baseline_vals = np.maximum(sub["baseline_auc"].values, 0.5)
    gait_vals = sub["gait_auc"].values
    return baseline_vals, gait_vals


def select_rows_for_gender(df, gender):
    # Compute delta as score - max(baseline_score, 0.5) for proper filtering
    df = df.copy()
    df["delta_corrected"] = df["score"] - df[["baseline_score"]].assign(
        b=lambda x: x["baseline_score"].clip(lower=0.5)
    )["b"]

    filtered = df[
        (df["model"] == MODEL)
        & (df["gender"] == gender)
        & (df["sub_model"] == "ensemble")
        & (df["score_type"] == "auc")
        & (df["wilcox_pvalue_fdr"] < FDR_THRESHOLD)
        & (df["delta_corrected"] >= MIN_DELTA)
        & (df["score"] >= MIN_SCORE)
    ].copy()

    filtered = filtered[~filtered["system"].isin(EXCLUDE_SYSTEMS)]
    filtered = filtered[~filtered["label"].isin(EXCLUDE_LABELS)]
    cohort_n = int(filtered["n_subjects"].max()) if "n_subjects" in filtered.columns and not filtered.empty else None

    selected_rows = []
    for system, top_n in [("medical_conditions", N_MEDICAL), ("medications", N_MEDICATIONS)]:
        sys_df = filtered[filtered["system"] == system].copy()
        sys_df = sys_df.sort_values("delta_corrected", ascending=False)
        count = 0
        for _, row in sys_df.iterrows():
            n_pos = get_n_positive_cached(system, row["label"], gender)
            if n_pos < MIN_POSITIVES:
                continue

            baseline_seed, gait_seed = load_seed_values(system, row["label"], gender)
            if baseline_seed is None or gait_seed is None:
                continue

            baseline_val = float(max(row["baseline_score"], 0.5))
            gait_val = float(row["score"])
            delta_val = gait_val - baseline_val

            selected_rows.append(
                {
                    "system": system,
                    "label": row["label"],
                    "label_pretty": clean_label_name(row["label"]),
                    "baseline": baseline_val,
                    "gait": gait_val,
                    "delta": delta_val,
                    "n_positive": int(n_pos),
                    "baseline_seed": baseline_seed,
                    "gait_seed": gait_seed,
                }
            )
            count += 1
            if count >= top_n:
                break

    return pd.DataFrame(selected_rows), cohort_n


def _build_layout_rows(panel_df):
    rows = []
    y = 0.0
    section_top = {}

    for system in ["medical_conditions", "medications"]:
        sys_rows = panel_df[panel_df["system"] == system].copy()
        if sys_rows.empty:
            continue
        section_top[system] = y
        for _, row in sys_rows.iterrows():
            rows.append((y, row))
            y += 1.0
        y += 0.9

    return rows, section_top


def _plot_seed_distribution(ax, seed_values, y_center, color, x_jitter=0.0032, y_jitter=0.10, point_size=4, point_alpha=0.24):
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


def draw_gender_panel(ax, panel_df, gender, panel_color, cohort_n=None, shared_y_max=None):
    rows, section_top = _build_layout_rows(panel_df)
    if not rows:
        ax.text(0.5, 0.5, f"No significant rows for {gender}", transform=ax.transAxes, ha="center")
        return

    fs = FONT_SIZE

    for y, row in rows:
        _plot_seed_distribution(ax, row["baseline_seed"], y, BASELINE_COLOR, point_size=3, point_alpha=0.16)
        _plot_seed_distribution(ax, row["gait_seed"], y, panel_color, point_size=4, point_alpha=0.26)

        ax.plot([row["baseline"], row["gait"]], [y, y], color=panel_color, lw=0.9, alpha=0.75, zorder=2)

        ax.scatter([row["baseline"]], [y], s=8, facecolors="white", edgecolors=BASELINE_COLOR, linewidths=0.7, zorder=3)
        ax.scatter([row["gait"]], [y], s=18, facecolors=panel_color, edgecolors="white", linewidths=0.4, zorder=4)

        ax.text(LABEL_X, y, row["label_pretty"], ha="right", va="center", fontsize=fs, color=TEXT_COLOR)
        ax.text(DELTA_X, y, f"+{row['delta']:.3f}", ha="left", va="center", fontsize=fs * 0.93, color=panel_color, fontweight="bold")

    section_headers = {"medical_conditions": "MEDICAL CONDITIONS", "medications": "MEDICATIONS"}
    for system, header in section_headers.items():
        if system in section_top:
            y_header = section_top[system] - 0.55
            if system == "medications":
                y_header -= 0.14
            ax.text(LABEL_X, y_header, header, ha="right", va="center", fontsize=fs, color="black", fontweight="bold")

    if section_top:
        top_header_y = min(section_top.values()) - 0.55
        ax.text(DELTA_HEADER_X, top_header_y, "ΔAUC", ha="left", va="center", fontsize=fs * 0.93, color="#666666", fontweight="bold")

    ax.set_xlim(X_LIM_MIN, X_LIM_MAX)
    y_top = shared_y_max if shared_y_max is not None else rows[-1][0]
    ax.set_ylim(-1.0, y_top + 0.4)
    ax.invert_yaxis()
    ax.set_xticks([0.50, 0.55, 0.60, 0.65, 0.70])
    ax.set_yticks([])
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.8)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.spines["bottom"].set_linewidth(0.5)

    ax.tick_params(axis="x", labelsize=fs, colors="#666666", width=0.5)

    if cohort_n is None:
        cohort_n = int(panel_df["n_positive"].max()) if not panel_df.empty else 0
    panel_title = f"{'Male' if gender == 'male' else 'Female'} (n = {cohort_n:,})"
    ax.set_title(panel_title, loc="left", fontsize=fs, fontweight="bold", color=TEXT_COLOR, pad=4)


def create_single_gender_plot(panel_df, gender, cohort_n, panel_color):
    if panel_df.empty:
        print(f"No rows for {gender}, skipping.")
        return

    n_rows = len(panel_df)
    # Height scales with number of rows; ~0.38 cm per row plus header/footer space
    fig_height_in = max((n_rows * 0.38 + 2.5) / 2.54, 4.0)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, fig_height_in))
    fig.patch.set_facecolor("white")

    draw_gender_panel(ax, panel_df, gender, panel_color, cohort_n=cohort_n)
    ax.set_xlabel("AUC-ROC", fontsize=FONT_SIZE, color="#666666")

    legend_handles = [
        Line2D([0], [0], marker="o", markersize=4, markerfacecolor="white", markeredgecolor=BASELINE_COLOR, lw=0, label="Age, BMI, VAT, Height"),
        Line2D([0], [0], marker="o", markersize=5, markerfacecolor=panel_color, markeredgecolor="white", lw=0, label="Age, BMI, VAT, Height & Gait"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.993),
        ncol=1,
        frameon=False,
        fontsize=FONT_SIZE * 0.9,
        handletextpad=0.4,
        labelspacing=0.3,
    )

    plt.subplots_adjust(left=0.38, right=0.88, top=0.87, bottom=0.10)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = f"medical_conditions_medications_dumbbell_{MODEL}_{gender}"
    for ext in ("png", "pdf"):
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.{ext}")
        dpi = 300 if ext == "png" else None
        plt.savefig(out_path, dpi=dpi, facecolor="white", edgecolor="white")
        print(f"Saved: {out_path}")

    plt.close()


def _panel_y_max(panel_df):
    rows, _ = _build_layout_rows(panel_df)
    return rows[-1][0] if rows else 0.0


def create_combined_plot(male_df, female_df, male_n, female_n):
    if male_df.empty or female_df.empty:
        print("One gender has no rows, skipping combined plot.")
        return

    # Size the figure to the taller panel; each panel fills its own axis height
    # (rows spaced out to fill, rather than leaving whitespace at the bottom).
    n_rows = int(round(max(_panel_y_max(male_df), _panel_y_max(female_df)))) + 1
    fig_height_in = max((n_rows * 0.52 + 3.0) / 2.54, 5.0)
    fig_width_in = 2.25 * FIG_WIDTH_IN

    fig, axes = plt.subplots(1, 2, figsize=(fig_width_in, fig_height_in))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], male_df, "male", MALE_COLOR, male_n, 0.30),
        (axes[1], female_df, "female", FEMALE_COLOR, female_n, 0.80),
    ]

    for ax, panel_df, gender, color, cohort_n, legend_x in panels:
        draw_gender_panel(ax, panel_df, gender, color, cohort_n=cohort_n)
        ax.set_xlabel("AUC-ROC", fontsize=FONT_SIZE, color="#666666")

        legend_handles = [
            Line2D([0], [0], marker="o", markersize=4, markerfacecolor="white", markeredgecolor=BASELINE_COLOR, lw=0, label="Age, BMI, VAT, Height"),
            Line2D([0], [0], marker="o", markersize=5, markerfacecolor=color, markeredgecolor="white", lw=0, label="Age, BMI, VAT, Height & Gait"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(legend_x, 0.995),
            ncol=1,
            frameon=False,
            fontsize=FONT_SIZE * 0.9,
            handletextpad=0.4,
            labelspacing=0.3,
        )

    plt.subplots_adjust(left=0.155, right=0.945, top=0.88, bottom=0.08, wspace=0.95)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_name = f"medical_conditions_medications_dumbbell_{MODEL}_combined"
    for ext in ("png", "pdf"):
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}.{ext}")
        dpi = 300 if ext == "png" else None
        plt.savefig(out_path, dpi=dpi, facecolor="white", edgecolor="white")
        print(f"Saved: {out_path}")

    plt.close()


def create_dumbbell_plots():
    np.random.seed(7)

    df = pd.read_csv(DATA_PATH, low_memory=False)

    male_df, _ = select_rows_for_gender(df, "male")
    female_df, _ = select_rows_for_gender(df, "female")

    create_single_gender_plot(male_df, "male", 1652, MALE_COLOR)
    create_single_gender_plot(female_df, "female", 1762, FEMALE_COLOR)
    create_combined_plot(male_df, female_df, 1652, 1762)


if __name__ == "__main__":
    create_dumbbell_plots()
