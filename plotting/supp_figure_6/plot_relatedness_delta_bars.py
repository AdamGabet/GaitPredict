"""
Supplementary Figure 6 — relatedness sensitivity.

Grouped bar chart of Δ Pearson r (gait improvement over baseline) for three anchor
phenotypes, comparing the full cohort ("Original") against an unrelated-only subset
("Unrelated-only"), per sex.

Reads the precomputed, de-identified per-seed Δr table from results/. The subject-level
computation (predictions, folds, relatedness graph, cohort gender) runs only on the
cluster (see review1/cohort_ancestry_handoff/scripts/build_relatedness_deltas.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
DELTAS_CSV = os.path.join(_RESULTS_DIR, "relatedness_per_seed_deltas.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

LABEL_NAMES = {
    "hand_grip_left":   "Grip strength",
    "liver_elasticity": "Liver elasticity",
    "spine_l1_l4_area": "Spine BMD",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_df = pd.read_csv(DELTAS_CSV)

    # Aggregate
    summary = (all_df.groupby(["label", "gender"])[["delta_full", "delta_clean"]]
               .agg(["mean", "sem"]).reset_index())
    summary.columns = ["label", "gender", "full_mean", "full_sem", "clean_mean", "clean_sem"]

    # Wilcoxon per group
    def wilcox_p(grp):
        d = grp["delta_full"].values - grp["delta_clean"].values
        if np.all(d == 0) or len(d) < 5:
            return 1.0
        _, p = wilcoxon(grp["delta_full"].values, grp["delta_clean"].values)
        return p

    pvals = all_df.groupby(["label", "gender"]).apply(wilcox_p).reset_index()
    pvals.columns = ["label", "gender", "wilcox_p"]
    summary = summary.merge(pvals, on=["label", "gender"])

    # Plot
    order = [(l, g) for l in ["hand_grip_left", "liver_elasticity", "spine_l1_l4_area"]
             for g in ["male", "female"]]
    summary["_order"] = summary.apply(lambda r: order.index((r["label"], r["gender"])), axis=1)
    summary = summary.sort_values("_order").reset_index(drop=True)

    label_order = ["hand_grip_left", "liver_elasticity", "spine_l1_l4_area"]
    xlabels = [LABEL_NAMES[l] for l in label_order]
    x = np.arange(len(label_order))
    w = 0.35
    y_max = summary[["full_mean", "clean_mean"]].max().max() * 1.25

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

    for ax, gender_name, title in zip(axes, ["male", "female"], ["Male", "Female"]):
        s = summary[summary["gender"] == gender_name].set_index("label").loc[label_order]
        ax.bar(x - w/2, s["full_mean"],  w, yerr=s["full_sem"],
               label="Original", color="#2166ac", edgecolor="black", linewidth=0.7,
               capsize=4, error_kw={"linewidth": 1.2})
        ax.bar(x + w/2, s["clean_mean"], w, yerr=s["clean_sem"],
               label="Unrelated-only", color="#b2182b", edgecolor="black", linewidth=0.7,
               alpha=0.85, capsize=4, error_kw={"linewidth": 1.2})
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel("Δ Predictive Power (Δ Pearson r)", fontsize=12)
    axes[1].legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/relatedness_delta_bars.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/relatedness_delta_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved.")
    print(summary[["label", "gender", "full_mean", "full_sem", "clean_mean", "clean_sem", "wilcox_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
