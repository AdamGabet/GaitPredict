"""2x2 paper grid: asbv (age/BMI/VAT) bars on top, figure-4-style system boxplot below.

Columns = Male | Female. Rows = asbv (a, b) / figure-4-style legs-vs-full (c, d).

Each panel is rendered as its own PNG at a common width/dpi (no resizing — preserves
identical effective font sizes), then composited with PIL.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "individual_plots"))
import figure4_style_legs_vs_full as f4   # noqa: E402
import bar_chart_asbv as asbv             # noqa: E402

PANEL_DIR = HERE / "individual_plots" / "grid_panels"
OUT_PNG = HERE / "output" / "grid_asbv_figure4.png"
OUT_PDF = HERE / "output" / "grid_asbv_figure4.pdf"

PANEL_DPI = 300
PANEL_W = 7.2  # common panel width (inches) so columns align cleanly

# Per-seed Pearson r values (for std-across-seeds error bars on the asbv bars).
SEED_VALUES_CSV = asbv.RESULTS / "lower_body9_asbv_seed_values.csv"
_SEED_VALUES = pd.read_csv(SEED_VALUES_CSV)


def seed_std(source: str, label: str, gender: str) -> float:
    """Std of the per-seed ensemble Pearson r for one target/gender."""
    sub = _SEED_VALUES[(_SEED_VALUES["source"] == source) & (_SEED_VALUES["label"] == label)]
    v = pd.to_numeric(sub[f"{gender}_pearson_r"], errors="coerce").dropna()
    return float(v.std(ddof=1)) if len(v) > 1 else 0.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── panel renderers ─────────────────────────────────────────────────────────
def render_asbv_panel(merged, gender: str, title: str, path: Path) -> None:
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
        for val, off in [(legs_val, -bar_w / 2), (full_val, bar_w / 2)]:
            ax.text(xs[i] + off, val + 0.008, f"{val:.2f}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([asbv.TARGET_LABELS[t] for t in asbv.TARGETS], fontsize=11)
    ax.set_ylabel("Pearson r", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=PANEL_DPI, facecolor="white", edgecolor="white")
    plt.close(fig)


def render_figure4_panel(df, sys_order, gender: str, path: Path) -> None:
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
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
              frameon=True, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=PANEL_DPI, facecolor="white", edgecolor="white")
    plt.close(fig)


# ── PIL compositor ───────────────────────────────────────────────────────────
def load_font(size: int) -> ImageFont.ImageFont:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/google-droid/DroidSans-Bold.ttf",
    ]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def pad_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    out = Image.new("RGB", (img.width, height), "white")
    out.paste(img, (0, 0))
    return out


def trim_grid_vertical(img: Image.Image, border: int = 10) -> Image.Image:
    bbox = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    if bbox is None:
        return img
    top = max(0, bbox[1] - border)
    bottom = min(img.size[1], bbox[3] + border)
    return img.crop((0, top, img.size[0], bottom))


def composite(panels: list[tuple[str, Path]]) -> None:
    imgs = [(letter, Image.open(p).convert("RGB")) for letter, p in panels]
    top_h = max(imgs[0][1].height, imgs[1][1].height)
    bot_h = max(imgs[2][1].height, imgs[3][1].height)
    imgs = [
        (imgs[0][0], pad_to_height(imgs[0][1], top_h)),
        (imgs[1][0], pad_to_height(imgs[1][1], top_h)),
        (imgs[2][0], pad_to_height(imgs[2][1], bot_h)),
        (imgs[3][0], pad_to_height(imgs[3][1], bot_h)),
    ]
    label_font = load_font(69)
    gap_x, gap_y, outer = 48, 2, 12

    col_w = [max(imgs[0][1].width, imgs[2][1].width),
             max(imgs[1][1].width, imgs[3][1].width)]
    row_h = [max(imgs[0][1].height, imgs[1][1].height),
             max(imgs[2][1].height, imgs[3][1].height)]

    canvas_w = outer * 2 + col_w[0] + gap_x + col_w[1]
    canvas_h = outer * 2 + row_h[0] + gap_y + row_h[1]
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    positions = [
        (outer, outer, col_w[0]),
        (outer + col_w[0] + gap_x, outer, col_w[1]),
        (outer, outer + row_h[0] + gap_y, col_w[0]),
        (outer + col_w[0] + gap_x, outer + row_h[0] + gap_y, col_w[1]),
    ]
    for (letter, img), (x, y, cell_w) in zip(imgs, positions):
        canvas.paste(img, (x + (cell_w - img.width) // 2, y))
        draw.text((x + 10, y + 10), letter, font=label_font, fill="black")

    canvas = trim_grid_vertical(canvas, border=10)
    canvas.save(OUT_PNG, dpi=(450, 450))
    canvas.save(OUT_PDF, resolution=450)
    print(OUT_PNG)
    print(OUT_PDF)


def main() -> None:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # asbv data (reuse bar_chart_asbv loaders/merge)
    legs_df = asbv._load(asbv.ASBV_CSV).rename(columns={"score": "legs_score", "delta": "legs_delta"})
    full_df = asbv._load(asbv.FULL_CSV, model_filter="long_seq").rename(
        columns={"score": "full_score", "delta": "full_delta"})
    asbv_merged = legs_df.merge(
        full_df[["label", "gender", "full_score", "full_delta"]],
        on=["label", "gender"], how="outer")

    # figure-4-style data
    f4_df = f4._load_data()
    sys_order = f4._sys_order(f4_df)

    p = {
        "a": PANEL_DIR / "asbv_male.png",
        "b": PANEL_DIR / "asbv_female.png",
        "c": PANEL_DIR / "figure4_male.png",
        "d": PANEL_DIR / "figure4_female.png",
    }
    render_asbv_panel(asbv_merged, "male", "Male", p["a"])
    render_asbv_panel(asbv_merged, "female", "Female", p["b"])
    render_figure4_panel(f4_df, sys_order, "male", p["c"])
    render_figure4_panel(f4_df, sys_order, "female", p["d"])

    composite([("a", p["a"]), ("b", p["b"]), ("c", p["c"]), ("d", p["d"])])


if __name__ == "__main__":
    main()
