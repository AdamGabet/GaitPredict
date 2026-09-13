"""Stage the figure files for submission (NMED-A150882A author checklist).

Nature wants "Separate Figure files (one file per figure)" and "Separate Extended
Data files (one file per figure)", named so the editor can tell at a glance which
is which. The build scripts write to plotting/<figure>/output/ under working
names, so this collects the shipped PDF for each figure into submission/ under the
name the checklist expects, and re-checks each one against the artwork rules:

  * 180 mm two-column width, text between 5 and 7 pt once placed at that width
  * no taller than ~225 mm so the figure fits an A4 page with its caption
  * vector with editable text (no Type 3 fonts, no flattened raster panels)

Nothing is regenerated here -- run the figure scripts first (see README). This
only copies and verifies, so it is safe to re-run.

    python plotting/stage_submission.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "submission")

# name -> path of the PDF that is actually shipped for that figure.
# Several build scripts emit more than one PDF; the one named here is the panel
# that goes in the paper, and the alternatives are noted in NOTES below.
FIGURES = [
    ("Fig_1", "plotting/figure_1/output/figure_1_composite.pdf"),
    ("Fig_2", "plotting/figure_2/output/figure_2_composite.pdf"),
    ("Fig_3", "plotting/figure_3/output/figure3_ensemble_all_pearson_r.pdf"),
    ("Fig_4", "plotting/figure_4_grid/output/long_seq_figure4_grid.pdf"),
    ("Fig_5", "plotting/figure_5_medical/output/"
              "medical_conditions_medications_dumbbell_long_seq_combined.pdf"),
    ("Fig_6", "plotting/figure_6/output/figure_6_composite.pdf"),

    ("Extended_Data_Fig_1", "plotting/extended_figure_1/output/gait_pearson_combined.pdf"),
    ("Extended_Data_Fig_2", "plotting/extended_figure_2/output/"
                            "duration_scaling_intermittent_age_bmi_vat_tm_3kmh.pdf"),
    # Extended Data Fig. 3 is the sex-overlay radar, built by the Figure 3 script.
    ("Extended_Data_Fig_3", "plotting/figure_3/output/"
                            "figure3_extended_gender_overlay_pearson_r.pdf"),
    ("Extended_Data_Fig_3_legend", "plotting/figure_3/output/figure3_extended_gender_legend.pdf"),
    ("Extended_Data_Fig_4", "plotting/extended_figure_4/output/"
                            "normalized_vs_gaitmae_absolute_r_gain.pdf"),
    ("Extended_Data_Fig_5", "plotting/extended_figure_5/output/"
                            "liver_elasticity_gait_vs_bloodpanel.pdf"),
    ("Extended_Data_Fig_6", "plotting/extended_figure_6/output/"
                            "clinical_scenario_medical_conditions_grid_ABCD.pdf"),

    ("Supplementary_Fig_2", "plotting/supp_figure_2/output/stratified_delta_r_figure4_grid.pdf"),
    ("Supplementary_Fig_3", "plotting/supp_figure_3/output/"
                            "longitudinal_significant_improvements_pearson.pdf"),
    ("Supplementary_Fig_4", "plotting/supp_figure_4/output/grid_asbv_figure4.pdf"),
    ("Supplementary_Fig_5", "plotting/supp_figure_5/output/ancestry_2x2_grid.pdf"),
    ("Supplementary_Fig_6", "plotting/supp_figure_6/output/relatedness_delta_bars.pdf"),
]

# Things a human has to settle; printed after the table so they are not missed.
NOTES = [
    ("Extended_Data_Fig_3", "ships with a separate legend file (staged as "
                            "Extended_Data_Fig_3_legend.pdf). Confirm whether Nature wants "
                            "the legend merged into the figure or supplied alongside it."),
    ("Supplementary_Fig_2", "supp_figure_2 emits 6 PDFs (age/BMI barplots and heatmaps, a "
                            "figure4-style grid and a robustness summary). The grid is staged "
                            "here -- confirm it is the one that was submitted."),
    ("Extended_Data_Fig_7", "is referenced by the author checklist but has no build script or "
                            "output in this repo. Confirm the Extended Data numbering matches "
                            "what was submitted."),
]

MAX_HEIGHT_MM = 225.0
COLUMN_MM = 180.0

# Companion files that are placed next to a figure rather than printed at the full
# two-column width, so the 180 mm text rules do not apply to them as written.
NOT_FULL_WIDTH = {"Extended_Data_Fig_3_legend"}


def audit(path):
    """(width_mm, height_at_180mm, min_pt, max_pt, n_type3, raster_only) or None."""
    try:
        import fitz
    except ImportError:
        return None
    d = fitz.open(path)
    pg = d[0]
    w = pg.rect.width / 72 * 25.4
    h = pg.rect.height / 72 * 25.4
    s = COLUMN_MM / w
    sizes = []
    for b in pg.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                if sp["text"].strip():
                    sizes.append(sp["size"] * s)
    t3 = sum(1 for p in d for f in p.get_fonts(full=True) if f[2] == "Type3")
    raster = not sizes and bool(pg.get_images())
    d.close()
    return (w, h * s, min(sizes) if sizes else None,
            max(sizes) if sizes else None, t3, raster)


def main():
    os.makedirs(OUT, exist_ok=True)
    missing, flagged = [], []

    print(f"Staging figures into {os.path.relpath(OUT, ROOT)}/\n")
    print(f"{'file':32} {'mm wide':>8} {'h@180':>6} {'text pt':>12}  notes")
    for name, rel in FIGURES:
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            missing.append((name, rel))
            print(f"{name + '.pdf':32} {'-':>8} {'-':>6} {'-':>12}  MISSING: {rel}")
            continue
        shutil.copy2(src, os.path.join(OUT, name + ".pdf"))

        a = audit(src)
        if a is None:
            print(f"{name + '.pdf':32} (install pymupdf to audit)")
            continue
        w, h180, lo, hi, t3, raster = a
        bad = []
        if name in NOT_FULL_WIDTH:
            rng = f"{lo:.1f}-{hi:.1f}" if lo is not None else "none"
            print(f"{name + '.pdf':32} {w:8.0f} {h180:6.0f} {rng:>12}  "
                  f"legend chip, placed at its own size")
            continue
        if h180 > MAX_HEIGHT_MM:
            bad.append(f"{h180:.0f}mm tall")
        if t3:
            bad.append(f"{t3} Type 3 fonts")
        if raster:
            bad.append("raster only")
        if lo is not None and hi is not None:
            if lo < 5:
                bad.append(f"text down to {lo:.1f}pt")
            if hi > 7:
                bad.append(f"text up to {hi:.1f}pt")
            rng = f"{lo:.1f}-{hi:.1f}"
        else:
            rng = "none"
        if bad:
            flagged.append((name, bad))
        print(f"{name + '.pdf':32} {w:8.0f} {h180:6.0f} {rng:>12}  "
              f"{'; '.join(bad) if bad else 'OK'}")

    sd = os.path.join(ROOT, "source_data")
    if os.path.isdir(sd):
        dst = os.path.join(OUT, "source_data")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(sd, dst)
        per_figure = {}
        for root, _, files in os.walk(dst):
            csvs = [f for f in files if f.endswith(".csv")]
            if csvs:
                per_figure[os.path.relpath(root, dst)] = len(csvs)
        total = sum(per_figure.values())
        print(f"\nsource_data/  {total} CSVs -- plotted values + exact P values, "
              f"one directory per figure")
        for figure in sorted(per_figure):
            print(f"  {figure:28} {per_figure[figure]} CSVs")

    if flagged:
        print("\nOutside Nature's artwork rules "
              "(180 mm wide, text 5-7 pt, no taller than 225 mm):")
        for name, bad in flagged:
            print(f"  {name:28} {'; '.join(bad)}")

    print("\nNeeds a decision before submitting:")
    for name, note in NOTES:
        print(f"  {name:28} {note}")

    if missing:
        print("\nNot built -- run the figure scripts (see README) and re-run this:")
        for name, rel in missing:
            print(f"  {name:28} {rel}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
