"""Extended Data Figure 4: Normalized keypoints vs GaitMAE absolute Pearson r gain.

Grouped bar chart per sex comparing predictive power (Pearson r) of normalized
raw keypoint inputs against GaitMAE gait embeddings across six targets, with the
percent gain annotated above each pair.

Reads the de-identified plot-ready CSV results/normalized_vs_gaitmae_gains.csv
(per target/sex/model seed-mean and seed-std of Pearson r) and writes
PDF + PNG to plotting/extended_figure_4/output/.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GAINS_CSV = os.path.join(RESULTS_DIR, 'normalized_vs_gaitmae_gains.csv')
SEED_VALUES_CSV = GAINS_CSV.replace("normalized_vs_gaitmae_gains.csv", "normalized_vs_gaitmae_seed_values.csv")

MODEL_KEYPOINTS = "Normalized keypoints"
MODEL_GAITMAE = "GaitMAE gait embeddings"
SEXES = ["Male", "Female"]
TARGETS = ["Age", "BMI", "VAT area", "Heart rate", "Grip right", "Liver elasticity"]
SHORT_LABELS = {
    "Age": "Age",
    "BMI": "BMI",
    "VAT area": "VAT\narea",
    "Heart rate": "Heart\nrate",
    "Grip right": "Grip\nright",
    "Liver elasticity": "Liver\nelasticity",
}
BAR_COLORS = {MODEL_KEYPOINTS: "#9bb5b1", MODEL_GAITMAE: "#5a8a5f"}


def build_compact_table(seed_table):
    rows = []
    for target in TARGETS:
        for sex in SEXES:
            keypoints = seed_table[
                (seed_table["target"] == target)
                & (seed_table["gender"] == sex)
                & (seed_table["model"] == MODEL_KEYPOINTS)
            ].iloc[0]
            gaitmae = seed_table[
                (seed_table["target"] == target)
                & (seed_table["gender"] == sex)
                & (seed_table["model"] == MODEL_GAITMAE)
            ].iloc[0]
            keypoint_r = float(keypoints["pearson_mean"])
            gaitmae_r = float(gaitmae["pearson_mean"])
            delta = gaitmae_r - keypoint_r
            rows.append(
                {
                    "target": target,
                    "sex": sex,
                    "normalized_keypoints_r": keypoint_r,
                    "gaitmae_embeddings_r": gaitmae_r,
                    "delta_pearson_r": delta,
                    "percent_gain_vs_keypoints": 100 * delta / keypoint_r,
                }
            )
    return pd.DataFrame(rows)


def metric_std(seed_table, target, sex, model):
    row = seed_table[
        (seed_table["target"] == target)
        & (seed_table["gender"] == sex)
        & (seed_table["model"] == model)
    ].iloc[0]
    return float(row["pearson_std"])


def set_final_style():
    # Nature: all figure text must print between 5 and 7 pt at the 180 mm
    # two-column width. This figure's canvas is 362 mm wide, so printed pt =
    # fontsize x 180/362 = x0.497. The previous hierarchy (14-19 pt) printed at
    # 7.0-9.4 pt. Compressed below to 12.5/13.5 pt -> 6.2/6.7 pt, which keeps the
    # title above the labels while leaving headroom against the 7 pt ceiling
    # (this figure saves with bbox_inches="tight", so the canvas width -- and
    # therefore the printed size -- shifts slightly with the text).
    plt.rcParams.update(
        {
            "font.size": 12.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.5,
        }
    )


def save_figure(fig, stem):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_absolute_gain(seed_table, compact_table, per_seed):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)
    for ax, sex in zip(axes, SEXES):
        sub = compact_table[compact_table["sex"] == sex].set_index("target").loc[TARGETS]
        x = np.arange(len(TARGETS))
        width = 0.34
        key_vals = sub["normalized_keypoints_r"].values
        gaitmae_vals = sub["gaitmae_embeddings_r"].values
        key_errs = [metric_std(seed_table, target, sex, MODEL_KEYPOINTS) for target in TARGETS]
        gaitmae_errs = [metric_std(seed_table, target, sex, MODEL_GAITMAE) for target in TARGETS]
        gains = sub["percent_gain_vs_keypoints"].values
        ax.bar(
            x - width / 2,
            key_vals,
            width,
            yerr=key_errs,
            capsize=4,
            color=BAR_COLORS[MODEL_KEYPOINTS],
            label=MODEL_KEYPOINTS,
            edgecolor="black",
            linewidth=0.7,
            error_kw={"elinewidth": 1.1},
        )
        ax.bar(
            x + width / 2,
            gaitmae_vals,
            width,
            yerr=gaitmae_errs,
            capsize=4,
            color=BAR_COLORS[MODEL_GAITMAE],
            label=MODEL_GAITMAE,
            edgecolor="black",
            linewidth=0.7,
            error_kw={"elinewidth": 1.1},
        )
        # Nature policy: bar charts must show the individual data points, and it
        # is mandatory here because the keypoints arm has only n=5 seeds.
        for offset, model in [(-width / 2, MODEL_KEYPOINTS), (width / 2, MODEL_GAITMAE)]:
            for xi, target in zip(x, TARGETS):
                vals = per_seed[
                    (per_seed["target"] == target)
                    & (per_seed["gender"] == sex)
                    & (per_seed["model"] == model)
                ]["value"].to_numpy(dtype=float)
                if len(vals) == 0:
                    continue
                jitter = np.linspace(-width * 0.17, width * 0.17, len(vals))
                ax.scatter(xi + offset + jitter, vals, s=11, color="black",
                           alpha=0.6, linewidths=0.3, edgecolors="white", zorder=6)

        for xi, y0, y1, pct in zip(x, key_vals, gaitmae_vals, gains):
            ax.text(
                xi,
                min(0.955, max(y0, y1) + 0.045),
                f"+{pct:.0f}%",
                ha="center",
                va="bottom",
                fontsize=11,
            )
        ax.set_title(sex, pad=9)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[target] for target in TARGETS])
        ax.tick_params(axis="x", pad=7)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Predictive Power (Pearson r)", fontsize=12.5, labelpad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.91], w_pad=2.0)
    save_figure(fig, "normalized_vs_gaitmae_absolute_r_gain")


def main():
    seed_table = pd.read_csv(GAINS_CSV)
    per_seed = pd.read_csv(SEED_VALUES_CSV)
    compact_table = build_compact_table(seed_table)
    set_final_style()
    plot_absolute_gain(seed_table, compact_table, per_seed)
    print(f"Wrote {os.path.join(OUTPUT_DIR, 'normalized_vs_gaitmae_absolute_r_gain.png')}")
    print(f"Wrote {os.path.join(OUTPUT_DIR, 'normalized_vs_gaitmae_absolute_r_gain.pdf')}")


if __name__ == "__main__":
    main()
