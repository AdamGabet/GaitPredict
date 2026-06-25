"""
Extended Figure 5 — liver ultrasound prediction: gait embeddings vs Age+BMI vs
Age+BMI + routine liver blood biomarkers (liver elasticity, per sex).

Grouped bars = mean Pearson r over 15 seeds (±std); paired-Wilcoxon significance
brackets across seeds. Reads two precomputed, de-identified aggregate CSVs from results/
(per-source mean/std summary + per-seed Pearson r). The model refitting on cohort blood
panels runs only on the cluster (see
review1/liver_bloodtest_vs_gait/build_ext5_data.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
SUMMARY_CSV = os.path.join(_RESULTS_DIR, "liver_predictive_power_summary.csv")
PERSEED_CSV = os.path.join(_RESULTS_DIR, "liver_perseed_pearson.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SOURCES = [("gait", "Gait Embeddings", "#2c7fb8"),
           ("demo", "Age + BMI", "#74add1"),
           ("clinical", "Age+BMI + routine liver biomarkers", "#d95f02")]

# Reconstruct the per-seed lookup: (target, setting, source, model, gender, metric) -> {seed: value}
_PS = pd.read_csv(PERSEED_CSV)
PERSEED = {
    key: dict(zip(grp["seed"], grp["value"]))
    for key, grp in _PS.groupby(["target", "setting", "source", "model", "gender", "metric"])
}


def best_of(df, target, source, gender, metric):
    sel = df[(df.target == target) & (df.source == source) &
             (df.gender == gender) & (df.metric == metric)]
    if sel.empty:
        return (np.nan, np.nan)
    row = sel.loc[sel["mean"].idxmax()]
    return (row["mean"], row["std"])


def best_series(df, target, source, gender, metric):
    sel = df[(df.target == target) & (df.source == source) &
             (df.gender == gender) & (df.metric == metric)]
    if sel.empty:
        return None
    row = sel.loc[sel["mean"].idxmax()]
    setting = row["setting"] if "setting" in sel.columns else None
    return PERSEED.get((target, setting, source, row["model"], gender, metric))


def _sig_label(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


_BAR_W = 0.26
_OFFSET = {"gait": -_BAR_W, "demo": 0.0, "clinical": _BAR_W}


def annotate_sig(ax, df, items, metric, pairs, higher_better=True, fontsize=6.8):
    ymin, ymax = ax.get_ylim()
    step = (ymax - ymin) * 0.045
    for xi, (tgt, gender) in enumerate(items):
        tops = [best_of(df, tgt, s, gender, metric) for s in _OFFSET]
        tops = [m + (sd if np.isfinite(sd) else 0) for m, sd in tops if np.isfinite(m)]
        if not tops:
            continue
        base, level = max(tops) + step * 0.8, 0
        for a, b in pairs:
            sa, sb = best_series(df, tgt, a, gender, metric), best_series(df, tgt, b, gender, metric)
            if not sa or not sb:
                continue
            seeds = sorted(set(sa) & set(sb))
            av = np.array([sa[s] for s in seeds]); bv = np.array([sb[s] for s in seeds])
            if len(seeds) < 5 or np.allclose(av, bv):
                continue
            try:
                p = wilcoxon(av, bv).pvalue
            except ValueError:
                continue
            txt = _sig_label(p)
            y = base + level * step * 1.7
            x1, x2 = xi + _OFFSET[a], xi + _OFFSET[b]
            ax.plot([x1, x1, x2, x2], [y, y + step * 0.45, y + step * 0.45, y],
                    lw=0.9, color="black", clip_on=False)
            ax.text((x1 + x2) / 2, y + step * 0.5, txt, ha="center", va="bottom",
                    fontsize=fontsize, color="black")
            level += 1


def _grouped(ax, labels, triples, title, ylabel):
    x = np.arange(len(labels)); w = 0.26
    for i, (key, lab, col) in enumerate(SOURCES):
        means = [triples[key][j][0] for j in range(len(labels))]
        stds = [triples[key][j][1] for j in range(len(labels))]
        ax.bar(x + (i - 1) * w, means, w, yerr=stds, capsize=3, label=lab,
               color=col, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df.setting == "full"]

    fig, ax = plt.subplots(figsize=(7, 5))
    triples = {k: [best_of(df, "liver_elasticity", k, g, "pearson_r") for g in ["male", "female"]]
               for k, _, _ in SOURCES}
    _grouped(ax, ["Male", "Female"], triples, "", "Predictive Power (Pearson r)")
    top = max(m + s for k, _, _ in SOURCES for m, s in triples[k])
    ax.set_ylim(0, top * 1.9)
    annotate_sig(ax, df, [("liver_elasticity", "male"), ("liver_elasticity", "female")],
                 "pearson_r", pairs=[("gait", "demo"), ("gait", "clinical")], fontsize=13)
    ax.set_ylabel("Liver Elasticity\nPredictive Power (Pearson r)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    for lab in ax.get_xticklabels():
        lab.set_fontsize(14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=12,
              framealpha=0.95, ncol=3)
    fig.tight_layout()
    for e in ["pdf", "png"]:
        fig.savefig(f"{OUT_DIR}/liver_elasticity_gait_vs_bloodpanel.{e}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved liver_elasticity_gait_vs_bloodpanel")


if __name__ == "__main__":
    main()
