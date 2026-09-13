"""
2×2 ancestry figure (Supplementary Figure 5).
  a (top-left):   Gait PCs coloured by genetic ancestry group
  b (top-right):  AUC of gait-embeddings vs age+gender discriminating ancestry
  c (bottom-left):  Age – within-ancestry CV vs cross-ancestry transfer
  d (bottom-right): Liver elasticity – same layout
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_THIS_DIR, "..", "..", "results")
OUT_DIR      = os.path.join(_THIS_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PCA_CSV         = os.path.join(RESULTS_DIR, "ancestry_gait_genetic_pca_subset.csv")
TRANSFER_CSV    = os.path.join(RESULTS_DIR, "ancestry_cross_ancestry_transfer_summary.csv")
WITHIN_CSV      = os.path.join(RESULTS_DIR, "ancestry_within_ancestry_cv_summary.csv")
ASH_VS_NON_CSV  = os.path.join(RESULTS_DIR, "ancestry_discrim_bmi_binary_summary_agg.csv")
BACKGROUND_CSV  = os.path.join(RESULTS_DIR, "ancestry_discrim_bmi_multiway_summary_agg.csv")

PNG_OUT = os.path.join(OUT_DIR, "ancestry_2x2_grid.png")
PDF_OUT = os.path.join(OUT_DIR, "ancestry_2x2_grid.pdf")

C_WITHIN   = "#D8DEE9"
C_TRANSFER = "#2F6FBB"
C_CHANCE   = "#B04040"

PANEL_SPECS = [
    {"target": "age",              "feature_set": "embeddings_only",
     "title": "Age",               "subtitle": "Embeddings only"},
    {"target": "liver_elasticity", "feature_set": "embeddings_plus_age_gender_bmi",
     "title": "Liver elasticity",  "subtitle": "Embeddings + age/sex/BMI"},
]

ANCESTRY_SHORT = {"ashkenazi": "Ashk.", "non_ashkenazi": "Non-Ashk."}

# colours / draw order for the genetic-PCA scatter (panel a)
ANCESTRY_COLORS = {
    "Ashkenazi": "#2F4B7C",
    "Middle Eastern": "#1B998B",
    "North African": "#7CB342",
    "Sephardi": "#225560",
    "Yemenite": "#2A9D8F",
    "Unknown": "#9E9E9E",
}
PCA_DRAW_ORDER = ["Unknown", "Sephardi", "Yemenite", "North African",
                  "Middle Eastern", "Ashkenazi"]


# ── panel a: genetic-PCA scatter (drawn natively, sharp in PDF) ───────────────

def plot_pca(ax, x_col="PC1", y_col="PC2"):
    df = pd.read_csv(PCA_CSV, usecols=["dominant_self_report", x_col, y_col])
    df = df.dropna(subset=[x_col, y_col, "dominant_self_report"])

    order = [a for a in PCA_DRAW_ORDER if a in set(df["dominant_self_report"])]
    counts = {}
    for ancestry in order:
        g = df[df["dominant_self_report"] == ancestry]
        counts[ancestry] = len(g)
        ax.scatter(g[x_col], g[y_col], s=10, alpha=0.75,
                   color=ANCESTRY_COLORS.get(ancestry, "#777777"),
                   edgecolors="none", rasterized=True)

    ax.set_xlabel(x_col, fontsize=11, color="#2C3440")
    ax.set_ylabel(y_col, fontsize=11, color="#2C3440")
    ax.tick_params(labelsize=9, colors="#2C3440")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B9C0CC")
    ax.spines["bottom"].set_color("#B9C0CC")

    # solid-colour proxy markers, ordered most→least frequent
    leg_order = sorted(counts, key=counts.get, reverse=True)
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=7,
                          markerfacecolor=ANCESTRY_COLORS.get(a, "#777777"),
                          markeredgecolor="none",
                          label=f"{a} (n={counts[a]})")
               for a in leg_order]
    leg = ax.legend(handles=handles, fontsize=8, loc="lower right",
                    handletextpad=0.3, labelspacing=0.3, borderpad=0.4,
                    frameon=True, framealpha=1.0, edgecolor="none")
    leg.get_frame().set_facecolor("white")


# ── panel b: ancestry discrimination ─────────────────────────────────────────

def plot_ancestry_discrimination(ax, ash_vs_non, background):
    groups = [
        {"label": "Gait predicts\nAshk. vs Non-Ashk.",   "df": ash_vs_non, "task": None},
        {"label": "Gait predicts\n3-way (Ashk./NAfr./NE)", "df": background, "task": "three_way"},
    ]

    bw = 0.42

    xpos = []
    for gi, grp in enumerate(groups):
        df = grp["df"]
        if grp["task"] is not None:
            df = df[df["task"] == grp["task"]]

        row  = df[df["feature_set"] == "gait_embeddings"].iloc[0]
        mean = float(row["auc_mean"])
        sd   = float(row["auc_sd"])
        x    = float(gi)
        xpos.append(x)

        ax.bar(x, mean, width=bw, color=C_TRANSFER, edgecolor="none", zorder=3)
        ax.errorbar(x, mean, yerr=sd, fmt="none",
                    color="#444", capsize=3, linewidth=1.2, zorder=4)
        ax.text(x, mean + sd + 0.010, f"{mean:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#1F2933")

    # chance line
    ax.axhline(0.5, color=C_CHANCE, linewidth=1.2, linestyle="--",
               zorder=0, alpha=0.75, label="_nolegend_")
    ax.text(0.98, 0.5 + 0.005, "chance (0.5)",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=C_CHANCE, alpha=0.8)

    ax.set_xticks(xpos)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=10)
    ax.set_xlim(-0.6, len(groups) - 1 + 0.6)
    # Nature policy: y-axis must start at 0 (was 0.38, which truncated the bars).
    ax.set_ylim(0, 0.75)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.grid(axis="y", color="#E7EAF0", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B9C0CC")
    ax.spines["bottom"].set_color("#B9C0CC")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=10, colors="#2C3440")
    ax.set_ylabel("AUC (balanced, matched)", fontsize=11, color="#2C3440")
    ax.set_title("Gait ancestry discrimination", fontsize=11, color="#2C3440")


# ── panels c / d: cross-ancestry transfer ────────────────────────────────────

def select_row(df, target, feature_set, direction):
    row = df[
        df["target"].eq(target)
        & df["feature_set"].eq(feature_set)
        & df["direction"].eq(direction)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected 1 row for {target}/{feature_set}/{direction}, got {len(row)}")
    return row.iloc[0]


def build_transfer_rows(transfer, within):
    rows = []
    for pidx, spec in enumerate(PANEL_SPECS):
        for tg in ["ashkenazi", "non_ashkenazi"]:
            trg = "non_ashkenazi" if tg == "ashkenazi" else "ashkenazi"
            wr  = select_row(within,   spec["target"], spec["feature_set"], f"{tg}_within_5fold")
            tr_ = select_row(transfer, spec["target"], spec["feature_set"], f"{trg}_to_{tg}")
            rows.append({"pidx": pidx, "title": spec["title"], "subtitle": spec["subtitle"],
                         "test_group": tg, "bar_type": "within",
                         "label": f"Within\n{ANCESTRY_SHORT[tg]}",
                         "pearson": wr["pearson"], "n": int(wr["n_scored_subjects"])})
            rows.append({"pidx": pidx, "title": spec["title"], "subtitle": spec["subtitle"],
                         "test_group": tg, "bar_type": "transfer",
                         "label": f"{ANCESTRY_SHORT[trg]} →\n{ANCESTRY_SHORT[tg]}",
                         "pearson": tr_["pearson"], "n": int(tr_["n_scored_subjects"])})
    return pd.DataFrame(rows)


def plot_transfer_panel(ax, panel_df, is_first):
    x      = np.array([0.0, 0.72, 1.85, 2.57])
    colors = [C_WITHIN if k == "within" else C_TRANSFER for k in panel_df["bar_type"]]
    bars   = ax.bar(x, panel_df["pearson"], width=0.58, color=colors, edgecolor="none")

    for bar, row in zip(bars, panel_df.itertuples(index=False)):
        ax.text(bar.get_x() + bar.get_width() / 2, row.pearson + 0.016,
                f"{row.pearson:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#1F2933")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.015,
                f"n={row.n}", ha="center", va="bottom",
                fontsize=8, color="#56616F", rotation=90)

    ax.axvline(1.285, color="#C9CED8", linewidth=1.0, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(panel_df["label"], fontsize=10)
    ax.set_ylim(0.0, 0.80)
    ax.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax.grid(axis="y", color="#E7EAF0", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B9C0CC")
    ax.spines["bottom"].set_color("#B9C0CC")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", labelsize=11, colors="#2C3440")
    if is_first:
        ax.set_ylabel("Pearson r", fontsize=12, color="#2C3440")
    ax.set_title(panel_df["title"].iloc[0], fontsize=12, color="#2C3440")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    plt.rcParams.update({
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    transfer   = pd.read_csv(TRANSFER_CSV)
    within     = pd.read_csv(WITHIN_CSV)
    ash_vs_non = pd.read_csv(ASH_VS_NON_CSV)
    background = pd.read_csv(BACKGROUND_CSV)
    plot_df    = build_transfer_rows(transfer, within)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.28})
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── a: genetic PCA (native, sharp) ───────────────────────────────────────
    plot_pca(ax_a, "PC1", "PC2")

    # ── b: discrimination ────────────────────────────────────────────────────
    plot_ancestry_discrimination(ax_b, ash_vs_non, background)

    # ── c: Age transfer ──────────────────────────────────────────────────────
    plot_transfer_panel(ax_c, plot_df[plot_df["pidx"] == 0], is_first=True)

    # ── d: Liver elasticity transfer ─────────────────────────────────────────
    plot_transfer_panel(ax_d, plot_df[plot_df["pidx"] == 1], is_first=False)

    # shared legend for c/d
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_WITHIN),
               plt.Rectangle((0, 0), 1, 1, color=C_TRANSFER)]
    fig.legend(handles, ["Within-ancestry 5-fold CV", "Cross-ancestry transfer"],
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=2, frameon=False, fontsize=11)

    # panel labels — sit clear of the y-tick labels and titles
    for lbl, ax in zip("abcd", [ax_a, ax_b, ax_c, ax_d]):
        ax.text(-0.13, 1.06, lbl, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="left",
                color="#1F2933")

    fig.savefig(PNG_OUT, dpi=250, bbox_inches="tight")
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PNG_OUT}")
    print(f"Wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
