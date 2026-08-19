"""Bar chart: Pearson r for age, BMI, and VAT — legs-only vs full-body, by gender.

Data (de-identified aggregates in the repo results/ dir):
  Legs only: lower_body9_asbv_pearson.csv  (lower_body9_predicting_asbv)
  Full body:  asbv_only_gait_pearson.csv    (predicting_asbv_only_gait, long_seq)
"""

from __future__ import annotations
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR  = Path(__file__).parent
RESULTS  = Path(os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")).resolve()
ASBV_CSV = RESULTS / "lower_body9_asbv_pearson.csv"
FULL_CSV = RESULTS / "asbv_only_gait_pearson.csv"

TARGETS = ["age", "bmi", "total_scan_vat_area"]
TARGET_LABELS = {"age": "Age", "bmi": "BMI", "total_scan_vat_area": "VAT"}
GENDERS = ["male", "female"]

COLOR_LEGS = "#2166ac"
COLOR_FULL = "#555555"
BAR_W = 0.32


def _load(path: Path, model_filter: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df[
        (df["sub_model"] == "ensemble")
        & (df["gender"].isin(GENDERS))
        & (df["label"].isin(TARGETS))
    ].copy()
    if model_filter:
        df = df[df["model"] == model_filter]
    return df[["label", "gender", "score", "delta", "baseline_score"]]


def plot() -> None:
    legs_df = _load(ASBV_CSV)
    full_df = _load(FULL_CSV, model_filter="long_seq")

    # merge
    merged = legs_df.rename(columns={"score": "legs_score", "delta": "legs_delta"}).merge(
        full_df.rename(columns={"score": "full_score", "delta": "full_delta",
                                "baseline_score": "baseline_score"})[
            ["label", "gender", "full_score", "full_delta"]],
        on=["label", "gender"], how="outer",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.87, bottom=0.12, wspace=0.35)

    for ax, gender in zip(axes, GENDERS):
        sub = merged[merged["gender"] == gender]
        xs = np.arange(len(TARGETS))

        for i, target in enumerate(TARGETS):
            row = sub[sub["label"] == target]
            if row.empty:
                continue
            legs_val = float(row["legs_score"].iloc[0]) if not row["legs_score"].isna().all() else 0
            full_val = float(row["full_score"].iloc[0]) if not row["full_score"].isna().all() else 0

            ax.bar(xs[i] - BAR_W / 2, legs_val, width=BAR_W,
                   color=COLOR_LEGS, alpha=0.85, label="Legs only" if i == 0 else "")
            ax.bar(xs[i] + BAR_W / 2, full_val, width=BAR_W,
                   color=COLOR_FULL, alpha=0.85, label="Full body" if i == 0 else "")

            # annotate values
            for val, offset in [(legs_val, -BAR_W / 2), (full_val, BAR_W / 2)]:
                ax.text(xs[i] + offset, val + 0.008, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels([TARGET_LABELS[t] for t in TARGETS], fontsize=10)
        ax.set_ylabel("Pearson r", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title(gender.capitalize(), fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Legs-only vs full-body: age, BMI, and VAT prediction",
                 fontsize=11, fontweight="bold")

    stem = OUT_DIR / "bar_chart_asbv"
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stem}.{{pdf,png}}")


if __name__ == "__main__":
    plot()
