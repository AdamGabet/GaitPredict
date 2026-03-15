"""
Dumbbell plot for medical conditions and medications (male vs female).
Publication-ready: 9 cm wide, 7.5 pt font, PNG + PDF per gender.
Delta = score - max(baseline_score, 0.5)
Std error bars from pre-computed gait_std / baseline_std columns in CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams.update({"font.size": 7.5})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_CSV = os.path.join(RESULTS_DIR, 'medical_conditions_pearson.csv')

MODEL = "long_seq"
BASELINE_MODEL = "Age_Gender_BMI_height_VAT"
FDR_THRESHOLD = 0.10
MIN_DELTA = 0.02
MIN_SCORE = 0.54
MIN_POSITIVES = 10
N_MEDICAL = 8
N_MEDICATIONS = 5

GAIT_COLOR = "#2471A3"
BASELINE_COLOR = "#AAB7B8"

LABEL_RENAME_DICT = {
    "Anxiety": "Anxiety Disorder",
    "Depression": "Depression",
    "Hypothyroidism": "Hypothyroidism",
    "Hypertension": "Hypertension",
    "Type 2 Diabetes": "T2 Diabetes",
    "Osteoporosis": "Osteoporosis",
    "Hyperlipidemia": "Hyperlipidemia",
    "Sleep Apnea": "Sleep Apnea",
    "Atrial Fibrillation": "Atrial Fibrillation",
    "Heart Failure": "Heart Failure",
}

EXCLUDE_LABELS = [
    "Obesity", "Overweight", "Underweight",
    "BMI_OVER_30", "BMI_OVER_25",
]


def clean_label_name(label):
    if str(label).lower() == "ibs":
        return "IBS"
    if label in LABEL_RENAME_DICT:
        return LABEL_RENAME_DICT[label]
    return label.replace("_", " ").title()


def select_rows_for_gender(df, gender):
    df = df.copy()
    df['delta_corrected'] = df['score'] - df['baseline_score'].clip(lower=0.5)

    filtered = df[
        (df['gender'] == gender) &
        (df['model'] == MODEL) &
        (df['score_type'] == 'auc') &
        (df['wilcox_pvalue_fdr'] < FDR_THRESHOLD) &
        (df['delta_corrected'] >= MIN_DELTA) &
        (df['score'] >= MIN_SCORE)
    ].copy()
    filtered = filtered[~filtered['label'].isin(EXCLUDE_LABELS)]

    cohort_n = int(filtered['n_subjects'].max()) if 'n_subjects' in filtered.columns and not filtered.empty else None

    selected_rows = []
    for system, top_n in [("medical_conditions", N_MEDICAL), ("medications", N_MEDICATIONS)]:
        sys_df = filtered[filtered['system'] == system].copy()
        sys_df = sys_df.sort_values("delta_corrected", ascending=False)
        count = 0
        for _, row in sys_df.iterrows():
            n_pos = int(row['n_positive']) if pd.notna(row.get('n_positive')) else 0
            if n_pos < MIN_POSITIVES:
                continue

            baseline_val = float(max(row['baseline_score'], 0.5))
            gait_val = float(row['score'])
            delta_val = gait_val - baseline_val
            gait_std = float(row['gait_std']) if pd.notna(row.get('gait_std')) else None
            baseline_std = float(row['baseline_std']) if pd.notna(row.get('baseline_std')) else None

            selected_rows.append({
                "system": system,
                "label": row["label"],
                "label_pretty": clean_label_name(row["label"]),
                "baseline": baseline_val,
                "gait": gait_val,
                "delta": delta_val,
                "n_positive": n_pos,
                "gait_std": gait_std,
                "baseline_std": baseline_std,
            })
            count += 1
            if count >= top_n:
                break

    return pd.DataFrame(selected_rows), cohort_n


def arrange_rows(panel_df):
    rows = []
    section_top = {}
    y = 0.0

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


def plot_dumbbell(gender):
    df = pd.read_csv(DATA_CSV, low_memory=False)
    panel_df, cohort_n = select_rows_for_gender(df, gender)

    if panel_df.empty:
        print(f"No data for {gender}")
        return

    rows, section_top = arrange_rows(panel_df)
    n_rows = len(rows)

    fig_h = max(4.0, n_rows * 0.38 + 1.2)
    fig, ax = plt.subplots(figsize=(3.54, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    panel_color = "#1A5276" if gender == "male" else "#922B21"

    for y, row in rows:
        # Connecting line
        ax.plot([row["baseline"], row["gait"]], [y, y],
                color="#CCCCCC", linewidth=1.2, zorder=1)

        # Baseline dot with error bar
        ax.scatter(row["baseline"], y, color=BASELINE_COLOR, s=18, zorder=3)
        if row["baseline_std"] is not None:
            ax.errorbar(row["baseline"], y, xerr=row["baseline_std"],
                        fmt='none', elinewidth=0.7, capsize=2, ecolor=BASELINE_COLOR, zorder=2)

        # Gait dot with error bar
        ax.scatter(row["gait"], y, color=panel_color, s=18, zorder=3)
        if row["gait_std"] is not None:
            ax.errorbar(row["gait"], y, xerr=row["gait_std"],
                        fmt='none', elinewidth=0.7, capsize=2, ecolor=panel_color, zorder=2)

        # Label
        ax.text(-0.01, y, row["label_pretty"], ha="right", va="center",
                fontsize=6.5, transform=ax.get_yaxis_transform())

        # n_positive annotation
        ax.text(1.01, y, f"n={row['n_positive']}", ha="left", va="center",
                fontsize=5.5, color="#555555", transform=ax.get_yaxis_transform())

    # Section labels
    for system, y_top in section_top.items():
        label = "Medical Conditions" if system == "medical_conditions" else "Medications"
        ax.text(-0.01, y_top - 0.55, label, ha="right", va="bottom",
                fontsize=6.5, fontweight="bold", color="#333333",
                transform=ax.get_yaxis_transform())

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.8, n_rows + 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("AUC", fontsize=7.5, fontweight="bold")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=BASELINE_COLOR,
               markersize=5, label='Age, Gender, BMI & Height'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=panel_color,
               markersize=5, label='+ Gait Embeddings'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6, framealpha=0.9)

    title = f"{'Male' if gender == 'male' else 'Female'}"
    if cohort_n:
        title += f" (n={cohort_n:,})"
    ax.set_title(title, fontsize=8, fontweight="bold", pad=6)

    plt.tight_layout(rect=[0.22, 0.0, 0.97, 1.0])

    fname = f"medical_conditions_medications_dumbbell_{MODEL}_{gender}"
    for ext in ["png", "pdf"]:
        out_path = os.path.join(OUTPUT_DIR, f"{fname}.{ext}")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path}")
    plt.close()


def main():
    for gender in ["male", "female"]:
        plot_dumbbell(gender)


if __name__ == "__main__":
    main()
