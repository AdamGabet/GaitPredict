"""Render the four panels of Supplementary Figure 4 as vector PDFs.

Panels: asbv (age/BMI/VAT) legs-vs-full bars for each sex on top (a, b), and the
figure-4-style system boxplot for each sex below (c, d).

Split out of grid_asbv_figure4.py so the panels stay matplotlib (which needs
pandas/statsmodels) while the grid is assembled with pymupdf, which lives in a
different environment. This is the same arrangement Extended Data Figure 6 uses.

Each panel is rendered at a common width so the columns align, and saved WITHOUT
bbox_inches="tight" so its canvas is exactly PANEL_W x the row height -- the grid
places panels at their native size and relies on that.

Run from the repo root, then run grid_asbv_figure4.py:
    python plotting/supp_figure_4/individual_plots/make_grid_panels.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure4_style_legs_vs_full as f4   # noqa: E402
import bar_chart_asbv as asbv             # noqa: E402

OUT_DIR = HERE / "grid_panels"

PANEL_DPI = 300      # only affects the PNG copies; the PDFs are vector
PANEL_W = 7.2        # common panel width (inches) so the columns align cleanly

# Nature: figure text must print between 5 and 7 pt at the 180 mm two-column
# width. The grid places these 7.2 in panels at native size on a 14.6 in page, so
# printed pt = fontsize x 180/371 = x0.485. The original 8/9/10 pt sizes printed
# at 3.9/4.4/4.8 pt -- below the 5 pt floor, which was invisible while this figure
# was a flattened raster. The canvas is a fixed figsize with no tight bbox, so
# enlarging these does not change the page geometry; tight_layout() reflows the
# axes inside it. Sizes already at or above 11 pt (ticks, titles) were left alone.
FS_BAR_VALUE = 10.5   # prints at 5.1 pt
FS_LEGEND = 10.5      # prints at 5.1 pt
FS_YLABEL = 11.0      # prints at 5.3 pt
FS_YTICK = 11.0       # prints at 5.3 pt; previously the 10 pt rcParams default
FS_LEGEND_F4 = 11.0   # prints at 5.3 pt

PANELS = {
    "a": "asbv_male",
    "b": "asbv_female",
    "c": "figure4_male",
    "d": "figure4_female",
}

# Per-seed Pearson r values (for std-across-seeds error bars on the asbv bars).
SEED_VALUES_CSV = asbv.RESULTS / "lower_body9_asbv_seed_values.csv"
_SEED_VALUES = pd.read_csv(SEED_VALUES_CSV)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def seed_vals(source: str, label: str, gender: str):
    """Per-seed ensemble Pearson r for one target/gender."""
    sub = _SEED_VALUES[(_SEED_VALUES["source"] == source) & (_SEED_VALUES["label"] == label)]
    return pd.to_numeric(sub[f"{gender}_pearson_r"], errors="coerce").dropna()


def seed_std(source: str, label: str, gender: str) -> float:
    """Std of the per-seed ensemble Pearson r for one target/gender."""
    v = seed_vals(source, label, gender)
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0


def _save(fig, stem: str) -> None:
    """Vector PDF for the grid, PNG alongside it for quick viewing."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", facecolor="white", edgecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=PANEL_DPI, facecolor="white", edgecolor="white")
    plt.close(fig)


def render_asbv_panel(merged, gender: str, title: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W, 4.0))
    sub = merged[merged["gender"] == gender]
    xs = np.arange(len(asbv.TARGETS))
    bar_w = asbv.BAR_W
    for i, target in enumerate(asbv.TARGETS):
        row = sub[sub["label"] == target]
        if row.empty:
            continue
        legs_val = float(row["legs_score"].iloc[0]) if not row["legs_score"].isna().all() else 0.0
        full_val = float(row["full_score"].iloc[0]) if not row["full_score"].isna().all() else 0.0
        legs_err = seed_std("legs", target, gender)
        full_err = seed_std("full", target, gender)
        ebar = dict(ecolor="black", elinewidth=1.0, capsize=4, capthick=1.0)
        ax.bar(xs[i] - bar_w / 2, legs_val, width=bar_w, color=asbv.COLOR_LEGS,
               alpha=0.85, label="Legs only" if i == 0 else "",
               yerr=legs_err, error_kw=ebar)
        ax.bar(xs[i] + bar_w / 2, full_val, width=bar_w, color=asbv.COLOR_FULL,
               alpha=0.85, label="Full body" if i == 0 else "",
               yerr=full_err, error_kw=ebar)
        # Nature policy: show the per-seed distribution behind each bar.
        for src, off in [("legs", -bar_w / 2), ("full", bar_w / 2)]:
            vals = seed_vals(src, target, gender).to_numpy(dtype=float)
            if not len(vals):
                continue
            jitter = np.linspace(-bar_w * 0.17, bar_w * 0.17, len(vals))
            ax.scatter(xs[i] + off + jitter, vals, s=8, color="black", alpha=0.55,
                       linewidths=0.3, edgecolors="white", zorder=6)
        for val, off in [(legs_val, -bar_w / 2), (full_val, bar_w / 2)]:
            ax.text(xs[i] + off, val + 0.008, f"{val:.2f}", ha="center",
                    va="bottom", fontsize=FS_BAR_VALUE, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([asbv.TARGET_LABELS[t] for t in asbv.TARGETS], fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=FS_YLABEL, fontweight="bold")
    ax.tick_params(axis="y", labelsize=FS_YTICK)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    # Pinned upper-left rather than the default loc="best": "best" avoids the
    # bars but not the value labels above them, and at the enlarged label size
    # it collided with the VAT bar's "0.81". The Age pair is always the lowest
    # in both sexes, so the upper left is the reliably empty corner.
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, stem)


def render_figure4_panel(df, sys_order, gender: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(PANEL_W, 5.0))
    sub = df[df["gender"] == gender].copy()
    panel_sys = [s for s in sys_order if s in sub["system"].values]
    rng = np.random.default_rng(42)
    # title="" and panel_letter="" — the grid adds Male/Female headers and a/b/c/d.
    f4._draw_panel(ax, sub, panel_sys, rng, "", "")
    # _draw_panel hardcodes small fonts (7.5pt ticks / 8pt label); enlarge for the grid.
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(40)
        lbl.set_ha("right")
    ax.yaxis.label.set_fontsize(13)
    ax.yaxis.label.set_fontweight("bold")
    legend_handles = [
        Patch(facecolor=(*mcolors.to_rgb(c), 0.25), edgecolor=c, lw=1.0, label=lbl)
        for lbl, _, c in f4.GROUPS
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=FS_LEGEND_F4,
              frameon=True, framealpha=0.85)
    fig.tight_layout()
    _save(fig, stem)


def main() -> None:
    # asbv data (reuse bar_chart_asbv loaders/merge)
    legs_df = asbv._load(asbv.ASBV_CSV).rename(
        columns={"score": "legs_score", "delta": "legs_delta"})
    full_df = asbv._load(asbv.FULL_CSV, model_filter="long_seq").rename(
        columns={"score": "full_score", "delta": "full_delta"})
    asbv_merged = legs_df.merge(
        full_df[["label", "gender", "full_score", "full_delta"]],
        on=["label", "gender"], how="outer")

    # figure-4-style data
    f4_df = f4._load_data()
    sys_order = f4._sys_order(f4_df)

    render_asbv_panel(asbv_merged, "male", "Male", PANELS["a"])
    render_asbv_panel(asbv_merged, "female", "Female", PANELS["b"])
    render_figure4_panel(f4_df, sys_order, "male", PANELS["c"])
    render_figure4_panel(f4_df, sys_order, "female", PANELS["d"])
    print(f"Wrote 4 panel PDFs to {OUT_DIR}")


if __name__ == "__main__":
    main()
