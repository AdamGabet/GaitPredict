"""Subgroup Δr stability (age + BMI strata) for reviewer response.

Feature-selection rule: frozen Figure-4 FDR-significant set (q<0.1, sex-stratified
from all_comparisons_pearson.csv). No model retraining, no re-selection within strata.
We subset saved nested-CV OOF predictions and recompute Pearson r within each
(sex × stratum) band, then aggregate by body system.

Outputs (all in OUT_DIR):
  stratified_subgroup_delta_r_fdr.csv         — system-level tidy summary (FDR set)
  stratified_subgroup_delta_r_full.csv        — same but full feature set (robustness)
  stratified_delta_r_{age,bmi}_strata_barplot.{pdf,png}
  stratified_delta_r_{age,bmi}_strata_heatmap.{pdf,png}
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # plotting/
from publication_colors import SYSTEM_RENAME_DICT, SYSTEM_COLOR_MAP


def load_columns_as_df(*args, **kwargs):
    """Subject-level covariate loader — only needed to BUILD the delta-r tables on the
    cluster. The public repo ships the precomputed de-identified tables, so this is never
    called when regenerating the figure from CSV."""
    raise RuntimeError(
        "load_columns_as_df needs the private cohort data; regenerate from the shipped "
        "results/ CSVs via replot_from_csv() instead of recomputing."
    )

# ── Constants ──────────────────────────────────────────────────────────────────

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")
RESULTS_ROOT = _RESULTS_DIR  # per-target prediction reads are build-only; unused in plot path
COMPARISON_CSV = os.path.join(_RESULTS_DIR, "gait_vs_confounders_pearson.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
BASELINE_MODEL = "Age_Gender_BMI_height_VAT"
GAIT_MODEL_DIR = "long_seq/ensemble"
FDR_THRESHOLD = 0.1
MIN_SUBJECTS = 30
SEEDS = list(range(15))
EXCLUDED_SYSTEMS = {"wearable_weekly", "wearable_monthly", "subject", "proteomics"}
# height is a baseline covariate — predicting DEXA height from gait is trivial and inflates Δr
EXCLUDED_LABELS = {"height"}
# merge sub-systems into a single canonical system name before grouping
SYSTEM_MERGE_MAP: dict[str, str] = {
    "metabolites_annotated": "metabolites",
    "metabolites_unannotated": "metabolites",
}
MAX_WORKERS = 16

AGE_BINS = [-np.inf, 50.0, 65.0, np.inf]
AGE_LABELS = ["<50", "50–65", "≥65"]
BMI_BINS = [-np.inf, 25.0, 30.0, np.inf]
BMI_LABELS = ["<25", "25–30", "≥30"]
STRATA_LABELS = {"age": AGE_LABELS, "bmi": BMI_LABELS}

# One color per stratum band (blue / orange / red) — same across both M and F panels
STRATUM_COLORS = ["#4C78A8", "#F58518", "#E45756"]
COHORT_COLOR = "#555555"   # dark gray reference bar for full-cohort Δr

def canonical_system(s: str) -> str:
    return SYSTEM_MERGE_MAP.get(s, s)


# ── Dataclass ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Target:
    display_name: str
    system: str       # canonical (merged) — used for grouping and output
    label: str
    raw_system: str   # original system name — used for file-path lookup


# ── Path helpers ───────────────────────────────────────────────────────────────


def gait_predictions_path(t: Target) -> str:
    return os.path.join(RESULTS_ROOT, t.raw_system, t.label, GAIT_MODEL_DIR, "predictions.csv")


def baseline_predictions_path(t: Target) -> str:
    return os.path.join(RESULTS_ROOT, t.raw_system, t.label, BASELINE_MODEL, "predictions.csv")


# ── Load frozen set ────────────────────────────────────────────────────────────


def load_frozen_set(all_features: bool = False) -> dict[str, list[Target]]:
    """Load sex-specific frozen target lists from all_comparisons_pearson.csv.

    Returns dict keyed by 'M'/'F' with deduplicated Target lists.
    When all_features=True, skips FDR and is_better filters (robustness run).
    """
    df = pd.read_csv(COMPARISON_CSV, low_memory=False)

    # Gender column uses 'male'/'female' in this CSV; map to 'M'/'F' internally
    GENDER_MAP = {"male": "M", "female": "F"}

    mask = (
        (df["sub_model"] == "ensemble")
        & (df["gender"].isin(GENDER_MAP))
        & (~df["system"].isin(EXCLUDED_SYSTEMS))
    )
    df = df[mask].copy()

    if not all_features:
        df = df[
            (df["wilcox_pvalue_fdr"] < FDR_THRESHOLD)
            & (df["is_better"] == True)  # noqa: E712
            & (df["score"] > 0)
            & (df["score_pvalue"] < 0.05)
        ]

    df = df[~df["label"].isin(EXCLUDED_LABELS)]

    result: dict[str, list[Target]] = {"M": [], "F": []}
    for _, row in df.iterrows():
        name = row.get("description") or row["label"]
        raw_sys = str(row["system"])
        t = Target(str(name), canonical_system(raw_sys), str(row["label"]), raw_sys)
        sex_key = GENDER_MAP[row["gender"]]
        result[sex_key].append(t)

    for sex in ("M", "F"):
        seen: set[tuple[str, str]] = set()
        unique: list[Target] = []
        for t in result[sex]:
            key = (t.raw_system, t.label)  # deduplicate on raw to keep both metabolite sub-systems
            if key not in seen:
                seen.add(key)
                unique.append(t)
        result[sex] = unique

    return result


# ── Prediction loaders (mirrors stratified_performance.py) ────────────────────


def safe_pearson(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = pd.DataFrame({"a": y_true, "b": y_pred}).dropna()
    if len(valid) < 3:
        return np.nan
    if valid["a"].nunique() < 2 or valid["b"].nunique() < 2:
        return np.nan
    return float(valid["a"].corr(valid["b"], method="pearson"))


def load_gait_predictions(t: Target) -> pd.DataFrame:
    df = pd.read_csv(
        gait_predictions_path(t),
        usecols=["RegistrationCode", "research_stage", "seed", "y_true", "y_pred", "gender"],
    )
    df["seed"] = df["seed"].astype(int)
    return (
        df.groupby(["RegistrationCode", "research_stage", "gender", "seed"], as_index=False)
        .agg(y_true=("y_true", "first"), gait_score=("y_pred", "mean"))
    )


def load_baseline_predictions(t: Target, gender_lookup: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(baseline_predictions_path(t))
    seed_cols = [c for c in df.columns if c.startswith("seed_")]
    long = df[["RegistrationCode", "research_stage", "true_values"] + seed_cols].melt(
        id_vars=["RegistrationCode", "research_stage", "true_values"],
        value_vars=seed_cols,
        var_name="seed",
        value_name="baseline_score",
    )
    long["seed"] = long["seed"].str.replace("seed_", "", regex=False).astype(int)
    long = long.rename(columns={"true_values": "y_true"})
    long = long.merge(
        gender_lookup,
        on=["RegistrationCode", "research_stage"],
        how="inner",
        validate="many_to_one",
    )
    return (
        long.groupby(["RegistrationCode", "research_stage", "gender", "seed"], as_index=False)
        .agg(y_true=("y_true", "first"), baseline_score=("baseline_score", "mean"))
    )


# ── Age/BMI loader ─────────────────────────────────────────────────────────────


def load_age_bmi() -> pd.DataFrame:
    cov = load_columns_as_df(["age", "bmi"], merge_closest_research_stage=True)
    return cov.reset_index()[["RegistrationCode", "research_stage", "age", "bmi"]]


# ── Core computation ───────────────────────────────────────────────────────────


def _process_one_target(sex: str, t: Target, age_bmi: pd.DataFrame) -> list[dict]:
    """Process a single (sex, target) pair. Called concurrently from thread pool."""
    sex_val = 1.0 if sex == "M" else 0.0
    gpath, bpath = gait_predictions_path(t), baseline_predictions_path(t)
    if not os.path.exists(gpath) or not os.path.exists(bpath):
        return []

    gait = load_gait_predictions(t)
    gender_lookup = gait[["RegistrationCode", "research_stage", "gender"]].drop_duplicates()
    base = load_baseline_predictions(t, gender_lookup)

    merged = gait.merge(
        base,
        on=["RegistrationCode", "research_stage", "gender", "seed"],
        how="inner",
        suffixes=("", "_baseline"),
    )
    merged = merged[merged["gender"] == sex_val].copy()
    merged = merged.merge(age_bmi, on=["RegistrationCode", "research_stage"], how="inner")
    merged = merged[merged["seed"].isin(SEEDS)].copy()

    merged["age_stratum"] = pd.cut(
        merged["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    ).astype(object)
    merged["bmi_stratum"] = pd.cut(
        merged["bmi"], bins=BMI_BINS, labels=BMI_LABELS, include_lowest=True
    ).astype(object)

    rows: list[dict] = []
    for stratum_type, stratum_col in [("age", "age_stratum"), ("bmi", "bmi_stratum")]:
        for stratum in STRATA_LABELS[stratum_type]:
            for seed in SEEDS:
                sub = merged[(merged[stratum_col] == stratum) & (merged["seed"] == seed)]
                n_subj = sub["RegistrationCode"].nunique()
                if n_subj < MIN_SUBJECTS:
                    continue
                r_gait = safe_pearson(sub["y_true"], sub["gait_score"])
                r_base = safe_pearson(sub["y_true"], sub["baseline_score"])
                delta = (r_gait - r_base) if not (np.isnan(r_gait) or np.isnan(r_base)) else np.nan
                rows.append(
                    {
                        "sex": sex,
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "system": t.system,
                        "label": t.label,
                        "seed": seed,
                        "n_subjects": n_subj,
                        "r_baseline": r_base,
                        "r_gait": r_gait,
                        "delta_r": delta,
                    }
                )
    return rows


def compute_feature_seed_rows(
    sex: str,
    targets: list[Target],
    age_bmi: pd.DataFrame,
) -> list[dict]:
    """Compute per-feature × per-seed × per-stratum Δr rows for one sex, in parallel."""
    all_rows: list[dict] = []
    n_done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_process_one_target, sex, t, age_bmi): t for t in targets}
        for future in as_completed(futures):
            t = futures[future]
            n_done += 1
            try:
                rows = future.result()
                all_rows.extend(rows)
                if n_done % 50 == 0 or n_done == len(targets):
                    print(
                        f"  [{n_done}/{len(targets)}] {sex} done — "
                        f"{len(all_rows)} rows so far",
                        flush=True,
                    )
            except Exception as exc:
                print(f"  ERROR {sex}/{t.system}/{t.label}: {exc}", flush=True)

    return all_rows


def build_system_summary(feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-feature × per-seed rows to system-level tidy CSV."""
    # per-feature summary across seeds
    feat_agg = (
        feature_rows.groupby(["sex", "stratum_type", "stratum", "system", "label"])
        .agg(
            feat_mean_delta_r=("delta_r", "mean"),      # mean Δr over seeds for this feature
            feat_std_delta_r=("delta_r", "std"),         # seed-to-seed variability per feature
            feat_mean_n_subj=("n_subjects", "mean"),
        )
        .reset_index()
    )

    rows = []
    for key, grp in feat_agg.groupby(["sex", "stratum_type", "stratum", "system"]):
        sex, stratum_type, stratum, system = key
        vals = grp["feat_mean_delta_r"].dropna()
        rows.append(
            {
                "sex": sex,
                "stratum_type": stratum_type,
                "stratum": stratum,
                "body_system": system,
                "n_features": len(grp),
                "n_features_with_data": int(vals.notna().sum()),
                "n_participants_in_band": int(round(grp["feat_mean_n_subj"].median())),
                "median_delta_r": float(vals.median()) if len(vals) > 0 else np.nan,
                "dispersion_across_features": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "dispersion_across_splits": float(grp["feat_std_delta_r"].median()),
            }
        )
    return pd.DataFrame(rows)


# ── Plotting helpers ───────────────────────────────────────────────────────────


def system_label(system: str) -> str:
    return SYSTEM_RENAME_DICT.get(system, system.replace("_", " ").title())


def _system_order(df: pd.DataFrame) -> list[str]:
    return (
        df.groupby("body_system")["median_delta_r"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def load_cohort_ref(all_features: bool = False) -> pd.DataFrame:
    """System-level median Δr for the full cohort (no band subsetting) from comparison CSV.

    Used as a reference bar alongside the stratum bars so readers can see whether
    strata are above or below the overall estimate.
    """
    GENDER_MAP = {"male": "M", "female": "F"}
    df = pd.read_csv(COMPARISON_CSV, low_memory=False)
    mask = (
        (df["sub_model"] == "ensemble")
        & (df["gender"].isin(GENDER_MAP))
        & (~df["system"].isin(EXCLUDED_SYSTEMS))
    )
    df = df[mask].copy()
    if not all_features:
        df = df[
            (df["wilcox_pvalue_fdr"] < FDR_THRESHOLD)
            & (df["is_better"] == True)  # noqa: E712
            & (df["score"] > 0)
            & (df["score_pvalue"] < 0.05)
        ]
    df = df[~df["label"].isin(EXCLUDED_LABELS)]
    df["sex"] = df["gender"].map(GENDER_MAP)
    df["system"] = df["system"].map(canonical_system)
    rows = []
    for (sex, system), grp in df.groupby(["sex", "system"]):
        vals = grp["delta"].dropna()
        rows.append(
            {
                "sex": sex,
                "body_system": system,
                "cohort_median_delta_r": float(vals.median()),
                "cohort_dispersion": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def plot_bar(
    df: pd.DataFrame,
    stratum_type: str,
    cohort_ref: pd.DataFrame | None = None,
    feature_df: pd.DataFrame | None = None,
    label_suffix: str = "",
) -> None:
    """Strip + outline-bar chart: each system shows dots per feature and a ghost bar at the median.

    Layout per panel (M / F):
      - One row per body system, with 4 sub-rows (3 strata + full-cohort reference)
      - Outline bar (very light fill) = system-level median Δr
      - Dots = individual FDR-significant features, jittered within the bar height
      - Median tick = short vertical line at the median position
      - Callout = name of the highest-Δr feature in each system (across any stratum)
    """
    sub = df[df["stratum_type"] == stratum_type].copy()
    sys_order = _system_order(sub)
    strata = STRATA_LABELS[stratum_type]
    all_bars = strata + ["Full cohort"]
    all_colors = STRATUM_COLORS + [COHORT_COLOR]
    n_sys = len(sys_order)

    bar_h = 0.18
    group_gap = 0.38
    y_step = len(all_bars) * bar_h + group_gap
    y_centers = {s: i * y_step for i, s in enumerate(sys_order)}

    # ── Pre-compute per-feature dots ──────────────────────────────────────────
    # feat_dots[(sex, stratum, system)] = array of per-feature mean Δr values
    feat_dots: dict = {}
    feat_labels: dict = {}
    if feature_df is not None:
        fmean = (
            feature_df[feature_df["stratum_type"] == stratum_type]
            .groupby(["sex", "stratum", "system", "label"])["delta_r"]
            .mean()
            .reset_index()
        )
        for (sex, stratum, system), grp in fmean.groupby(["sex", "stratum", "system"]):
            s = grp.set_index("label")["delta_r"].dropna()
            feat_dots[(sex, stratum, system)] = s.values
            feat_labels[(sex, stratum, system)] = s.index.tolist()

    # cohort dots: per-feature delta from comparison CSV (no stratum — full cohort)
    comp = pd.read_csv(COMPARISON_CSV, low_memory=False)
    GENDER_MAP = {"male": "M", "female": "F"}
    comp_fdr = comp[
        (comp["sub_model"] == "ensemble")
        & (comp["gender"].isin(GENDER_MAP))
        & (~comp["system"].isin(EXCLUDED_SYSTEMS))
        & (~comp["label"].isin(EXCLUDED_LABELS))
        & (comp["wilcox_pvalue_fdr"] < FDR_THRESHOLD)
        & (comp["is_better"] == True)  # noqa: E712
        & (comp["score"] > 0)
        & (comp["score_pvalue"] < 0.05)
    ].copy()
    comp_fdr["sex"] = comp_fdr["gender"].map(GENDER_MAP)
    comp_fdr["system"] = comp_fdr["system"].map(canonical_system)
    cohort_dots: dict = {}   # (sex, system) → array of delta values
    for (sex, system), grp in comp_fdr.groupby(["sex", "system"]):
        cohort_dots[(sex, system)] = grp["delta"].dropna().values

    # feature descriptions for callout text
    desc = (
        comp[["label", "description"]].drop_duplicates()
        .set_index("label")["description"].to_dict()
    )

    # top feature per (sex, system) across all strata — for callout annotation
    top_feat: dict = {}   # (sex, system) → (display_name, best_delta_r)
    if feature_df is not None:
        overall = (
            feature_df[feature_df["stratum_type"] == stratum_type]
            .groupby(["sex", "system", "label"])["delta_r"]
            .mean()
            .reset_index()
        )
        for (sex, system), grp in overall.groupby(["sex", "system"]):
            if grp.empty:
                continue
            best = grp.loc[grp["delta_r"].idxmax()]
            lbl = str(best["label"])
            top_feat[(sex, system)] = (lbl, float(best["delta_r"]))

    # ── Plotting ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, max(5, 0.65 * n_sys)),
        constrained_layout=True,
    )
    rng = np.random.default_rng(42)

    for ax, sex in zip(axes, ["M", "F"]):
        sex_df = sub[sub["sex"] == sex].set_index("body_system")
        cref_vals: dict = {}
        if cohort_ref is not None:
            cr = cohort_ref[cohort_ref["sex"] == sex].set_index("body_system")
            for s in sys_order:
                cref_vals[s] = float(cr.loc[s, "cohort_median_delta_r"]) if s in cr.index else np.nan

        for bi, (bar_label, color) in enumerate(zip(all_bars, all_colors)):
            offset = (bi - (len(all_bars) - 1) / 2.0) * bar_h
            rgba_face = (*mcolors.to_rgb(color), 0.10)

            for sys in sys_order:
                yc = y_centers[sys] + offset

                # median value and feature dots for this bar
                if bar_label == "Full cohort":
                    median_val = cref_vals.get(sys, np.nan)
                    dots = cohort_dots.get((sex, sys), np.array([]))
                else:
                    row = sex_df[sex_df["stratum"] == bar_label]
                    median_val = float(row.loc[sys, "median_delta_r"]) if sys in row.index else np.nan
                    dots = feat_dots.get((sex, bar_label, sys), np.array([]))

                if np.isnan(median_val):
                    continue

                # ghost (outline) bar
                ax.barh(
                    yc, median_val,
                    height=bar_h * 0.92,
                    facecolor=rgba_face,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=2,
                )
                # median tick
                ax.plot(
                    [median_val, median_val],
                    [yc - bar_h * 0.44, yc + bar_h * 0.44],
                    color=color, linewidth=2.0, zorder=4, solid_capstyle="butt",
                )
                # feature dots
                if len(dots) > 0:
                    jitter = rng.uniform(-bar_h * 0.33, bar_h * 0.33, size=len(dots))
                    ax.scatter(
                        dots, yc + jitter,
                        color=color, s=10, alpha=0.55,
                        linewidths=0, zorder=3,
                    )

        # callout: name of top feature per system
        for sys in sys_order:
            if (sex, sys) not in top_feat:
                continue
            name, delta = top_feat[(sex, sys)]
            ax.annotate(
                name,
                xy=(delta, y_centers[sys]),
                xytext=(delta + 0.004, y_centers[sys]),
                fontsize=5.5,
                color="#333333",
                va="center",
                ha="left",
                zorder=5,
            )

        ax.axvline(0, color="black", linewidth=0.9, zorder=3)
        ax.set_yticks([y_centers[s] for s in sys_order])
        ax.set_yticklabels([system_label(s) for s in sys_order], fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlabel("Δr  (gait − baseline)", fontsize=9)
        ax.set_title("Male" if sex == "M" else "Female", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.22, linewidth=0.6)

        handles = [
            Patch(facecolor=(*mcolors.to_rgb(c), 0.15), edgecolor=c, linewidth=1.0, label=lbl)
            for lbl, c in zip(all_bars, all_colors)
        ]
        ax.legend(
            handles=handles,
            title=stratum_type.upper() + " stratum",
            fontsize=8, title_fontsize=8,
            framealpha=0.75, loc="lower right",
        )

    fig.suptitle(
        f"Gait Δr stability — {stratum_type.upper()} strata{label_suffix}",
        fontsize=12, fontweight="bold", y=1.01,
    )

    base = os.path.join(OUT_DIR, f"stratified_delta_r_{stratum_type}_strata_barplot{label_suffix}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart → {base}.{{pdf,png}}", flush=True)


def plot_heatmap(df: pd.DataFrame, stratum_type: str, label_suffix: str = "") -> None:
    """Heatmap: systems × strata, M/F panels, diverging colormap."""
    sub = df[df["stratum_type"] == stratum_type].copy()
    sys_order = _system_order(sub)
    strata = STRATA_LABELS[stratum_type]

    vmax = max(sub["median_delta_r"].abs().quantile(0.95), 0.02)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, max(4, 0.44 * len(sys_order))),
        constrained_layout=True,
    )

    for ax, sex in zip(axes, ["M", "F"]):
        sex_df = sub[sub["sex"] == sex].set_index("body_system")
        mat = np.full((len(sys_order), len(strata)), np.nan)
        for xi, stratum in enumerate(strata):
            st = sex_df[sex_df["stratum"] == stratum]
            for yi, sys in enumerate(sys_order):
                if sys in st.index:
                    mat[yi, xi] = float(st.loc[sys, "median_delta_r"])

        im = ax.imshow(
            mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"
        )
        ax.set_xticks(range(len(strata)))
        ax.set_xticklabels(strata, fontsize=9)
        ax.set_yticks(range(len(sys_order)))
        ax.set_yticklabels([system_label(s) for s in sys_order], fontsize=8.5)
        title_sex = "Male" if sex == "M" else "Female"
        ax.set_title(title_sex, fontsize=11, fontweight="bold")

        for yi in range(len(sys_order)):
            for xi in range(len(strata)):
                val = mat[yi, xi]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > vmax * 0.55 else "black"
                    ax.text(xi, yi, f"{val:.2f}", ha="center", va="center",
                            fontsize=7.5, color=text_color, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.75, label="Δr", aspect=20)

    fig.suptitle(
        f"Gait Δr heatmap — {stratum_type.upper()} strata{label_suffix}",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    base = os.path.join(
        OUT_DIR,
        f"stratified_delta_r_{stratum_type}_strata_heatmap{label_suffix}",
    )
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap      → {base}.{{pdf,png}}", flush=True)


# ── Figure-4-style 2×2 grid ───────────────────────────────────────────────────


def plot_figure4_grid(
    system_df: pd.DataFrame,
    feature_df: pd.DataFrame | None = None,
    cohort_ref: pd.DataFrame | None = None,
    label_suffix: str = "",
) -> None:
    """2×2 grid of grouped vertical boxplots mirroring Figure 4's style.

    Panels:  (a) Age / Male   (b) Age / Female
             (c) BMI / Male   (d) BMI / Female
    Each panel: one group of boxes per body system, one box per stratum + full cohort.
    Box distribution = per-feature mean Δr within that (system × stratum × sex).
    """
    GENDER_MAP = {"male": "M", "female": "F"}
    PANEL_LABELS = ["a", "b", "c", "d"]
    PANEL_META = [
        ("age", "M", "Age strata — Male"),
        ("age", "F", "Age strata — Female"),
        ("bmi", "M", "BMI strata — Male"),
        ("bmi", "F", "BMI strata — Female"),
    ]

    # ── pre-build feature-level mean Δr per (sex, stratum_type, stratum, system, label) ──
    feat_means: pd.DataFrame | None = None
    if feature_df is not None:
        feat_means = (
            feature_df.groupby(["sex", "stratum_type", "stratum", "system", "label"])["delta_r"]
            .mean()
            .reset_index()
        )

    # ── cohort dots from comparison CSV (full-cohort, no stratum) ──
    cohort_feat: dict = {}   # (sex, system) → np.array of per-feature delta values
    comp = pd.read_csv(COMPARISON_CSV, low_memory=False)
    comp_fdr = comp[
        (comp["sub_model"] == "ensemble")
        & (comp["gender"].isin(GENDER_MAP))
        & (~comp["system"].isin(EXCLUDED_SYSTEMS))
        & (~comp["label"].isin(EXCLUDED_LABELS))
        & (comp["wilcox_pvalue_fdr"] < FDR_THRESHOLD)
        & (comp["is_better"] == True)  # noqa: E712
        & (comp["score"] > 0)
        & (comp["score_pvalue"] < 0.05)
    ].copy()
    comp_fdr["sex"] = comp_fdr["gender"].map(GENDER_MAP)
    comp_fdr["system"] = comp_fdr["system"].map(canonical_system)
    for (sex, system), grp in comp_fdr.groupby(["sex", "system"]):
        cohort_feat[(sex, system)] = grp["delta"].dropna().values

    # global system order: sort by mean full-cohort median Δr across sexes desc
    all_sys = sorted(system_df["body_system"].unique())
    sys_medians = (
        system_df.groupby("body_system")["median_delta_r"].median().reindex(all_sys).fillna(0)
    )
    sys_order = sys_medians.sort_values(ascending=False).index.tolist()
    n_sys = len(sys_order)

    strata_meta = {
        "age": (AGE_LABELS, "Age stratum"),
        "bmi": (BMI_LABELS, "BMI stratum"),
    }
    all_strata_labels = {"age": AGE_LABELS + ["Full cohort"], "bmi": BMI_LABELS + ["Full cohort"]}
    all_strata_colors = {"age": STRATUM_COLORS + [COHORT_COLOR], "bmi": STRATUM_COLORS + [COHORT_COLOR]}

    n_boxes = 4   # 3 strata + full cohort
    box_w = 0.16
    group_w = n_boxes * box_w + 0.22   # total width per system group (wider gap)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 2, figsize=(max(14, n_sys * 0.75), 10.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.16, hspace=0.62, wspace=0.28)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, label, (st, sex, title) in zip(axes_flat, PANEL_LABELS, PANEL_META):
        strata = AGE_LABELS if st == "age" else BMI_LABELS
        bar_labels = strata + ["Full cohort"]
        colors = all_strata_colors[st]

        sub = system_df[(system_df["stratum_type"] == st) & (system_df["sex"] == sex)]

        # drop systems with no data for this sex (e.g. renal for females)
        panel_sys_order = [
            s for s in sys_order
            if s in sub["body_system"].values or (sex, s) in cohort_feat
        ]
        # additionally require at least one non-empty stratum box
        panel_sys_order = [
            s for s in panel_sys_order
            if sub[sub["body_system"] == s]["median_delta_r"].notna().any()
            or (sex, s) in cohort_feat
        ]

        for si, sys in enumerate(panel_sys_order):
            sys_color = SYSTEM_COLOR_MAP.get(sys, "#888888")
            cx = si * group_w   # center of this system's group

            for bi, (bar_lbl, color) in enumerate(zip(bar_labels, colors)):
                bx = cx + (bi - (n_boxes - 1) / 2.0) * box_w   # x center of this box

                if bar_lbl == "Full cohort":
                    vals = cohort_feat.get((sex, sys), np.array([]))
                else:
                    if feat_means is not None:
                        mask = (
                            (feat_means["sex"] == sex)
                            & (feat_means["stratum_type"] == st)
                            & (feat_means["stratum"] == bar_lbl)
                            & (feat_means["system"] == sys)
                        )
                        vals = feat_means[mask]["delta_r"].dropna().values
                    else:
                        row = sub[sub["stratum"] == bar_lbl]
                        if sys in row["body_system"].values:
                            vals = np.array([float(row[row["body_system"] == sys]["median_delta_r"].iloc[0])])
                        else:
                            vals = np.array([])

                if len(vals) == 0:
                    continue

                # box
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                iqr = q3 - q1
                wlo = max(vals.min(), q1 - 1.5 * iqr)
                whi = min(vals.max(), q3 + 1.5 * iqr)
                rect = plt.Rectangle(
                    (bx - box_w / 2, q1), box_w, q3 - q1,
                    facecolor=(*mcolors.to_rgb(color), 0.18),
                    edgecolor=color, linewidth=0.9, zorder=3,
                )
                ax.add_patch(rect)
                ax.plot([bx - box_w / 2, bx + box_w / 2], [med, med], color=color, lw=1.4, zorder=4)
                ax.plot([bx, bx], [wlo, q1], color=color, lw=0.7, zorder=2)
                ax.plot([bx, bx], [q3, whi], color=color, lw=0.7, zorder=2)

                # dots — same color as stratum box, less prominent
                jitter = rng.uniform(-box_w * 0.3, box_w * 0.3, size=len(vals))
                ax.scatter(bx + jitter, vals, color=color, s=4, alpha=0.35,
                           linewidths=0, zorder=5)

        n_panel_sys = len(panel_sys_order)
        ax.axhline(0, color="black", linewidth=0.8, zorder=1)
        ax.set_xticks([si * group_w for si in range(n_panel_sys)])
        ax.set_xticklabels(
            [system_label(s) for s in panel_sys_order],
            rotation=40, ha="right", fontsize=11,
        )
        ax.set_ylabel("Pearson r improvement\n(Δ from baseline)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-group_w * 0.6, (n_panel_sys - 1) * group_w + group_w * 0.6)

        # panel letter
        ax.text(-0.06, 1.02, label, transform=ax.transAxes,
                fontsize=17, fontweight="bold", va="bottom", ha="left")

    def _make_handles(labels):
        return [
            Patch(facecolor=(*mcolors.to_rgb(c), 0.25), edgecolor=c, linewidth=1.0, label=lbl)
            for lbl, c in zip(labels, STRATUM_COLORS + [COHORT_COLOR])
        ]

    # age legend — just below the top row
    fig.legend(
        handles=_make_handles(AGE_LABELS + ["Full cohort"]),
        title="Age stratum", fontsize=11, title_fontsize=11, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, 0.55),
        bbox_transform=fig.transFigure, framealpha=0.85,
    )
    # BMI legend — just below the bottom row
    fig.legend(
        handles=_make_handles(BMI_LABELS + ["Full cohort"]),
        title="BMI stratum", fontsize=11, title_fontsize=11, ncol=4,
        loc="upper center", bbox_to_anchor=(0.5, 0.045),
        bbox_transform=fig.transFigure, framealpha=0.85,
    )

    base = os.path.join(OUT_DIR, f"stratified_delta_r_figure4_grid{label_suffix}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure-4 grid → {base}.{{pdf,png}}", flush=True)


# ── Robustness summary table ──────────────────────────────────────────────────


def plot_robustness_table(
    feature_df: pd.DataFrame,
    system_df: pd.DataFrame,
    label_suffix: str = "",
) -> None:
    """Compact table figure: for each system × (age/BMI × M/F), fraction of FDR-significant
    features that maintain Δr > 0 in at least one stratum. Saved as PDF + CSV."""

    feat = feature_df[~feature_df["label"].isin(EXCLUDED_LABELS)].copy()

    # per-feature mean Δr across seeds, then flag if positive in ANY stratum
    fmean = (
        feat.groupby(["sex", "stratum_type", "stratum", "system", "label"])["delta_r"]
        .mean()
        .reset_index()
    )
    any_pos = (
        fmean.groupby(["sex", "stratum_type", "system", "label"])["delta_r"]
        .apply(lambda x: (x > 0).any())
        .reset_index(name="any_positive")
    )
    total_c = any_pos.groupby(["sex", "stratum_type", "system"]).size().reset_index(name="n_total")
    pos_c = (
        any_pos.groupby(["sex", "stratum_type", "system"])["any_positive"]
        .sum()
        .reset_index(name="n_positive")
    )
    summary = total_c.merge(pos_c, on=["sex", "stratum_type", "system"])
    summary["fraction"] = summary["n_positive"] / summary["n_total"]
    summary["label_text"] = summary["n_positive"].astype(str) + "/" + summary["n_total"].astype(str)

    # save CSV
    csv_path = os.path.join(OUT_DIR, f"stratified_robustness_summary{label_suffix}.csv")
    summary.to_csv(csv_path, index=False)

    # system order from system_df (same as main figure)
    all_sys = sorted(system_df["body_system"].unique())
    sys_medians = system_df.groupby("body_system")["median_delta_r"].median().reindex(all_sys).fillna(0)
    sys_order = sys_medians.sort_values(ascending=False).index.tolist()

    COLS = [("age", "M"), ("age", "F"), ("bmi", "M"), ("bmi", "F")]
    COL_LABELS = ["Age\nMale", "Age\nFemale", "BMI\nMale", "BMI\nFemale"]

    # build matrix
    idx = {s: i for i, s in enumerate(sys_order)}
    frac_mat = np.full((len(sys_order), 4), np.nan)
    text_mat = [[""] * 4 for _ in sys_order]
    for _, row in summary.iterrows():
        si = idx.get(row["system"])
        if si is None:
            continue
        ci = COLS.index((row["stratum_type"], row["sex"])) if (row["stratum_type"], row["sex"]) in COLS else None
        if ci is None:
            continue
        frac_mat[si, ci] = row["fraction"]
        text_mat[si][ci] = row["label_text"]

    n_sys = len(sys_order)
    fig, ax = plt.subplots(figsize=(5, max(4, n_sys * 0.32)))
    fig.subplots_adjust(left=0.32, right=0.98, top=0.92, bottom=0.08)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(frac_mat, aspect="auto", cmap=cmap, vmin=0.8, vmax=1.0,
                   origin="upper")

    for si in range(n_sys):
        for ci in range(4):
            txt = text_mat[si][ci]
            if txt:
                frac = frac_mat[si, ci]
                tc = "black" if frac > 0.88 else "white"
                ax.text(ci, si, txt, ha="center", va="center", fontsize=7, color=tc)

    ax.set_xticks(range(4))
    ax.set_xticklabels(COL_LABELS, fontsize=8)
    ax.set_yticks(range(n_sys))
    ax.set_yticklabels([system_label(s) for s in sys_order], fontsize=7.5)
    ax.set_title(
        "Features with Δr > 0 in ≥1 stratum\n(n positive / n total FDR-sig)",
        fontsize=8, fontweight="bold",
    )
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticks(range(4))
    ax.set_xticklabels(COL_LABELS, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    base = os.path.join(OUT_DIR, f"stratified_robustness_summary{label_suffix}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Robustness table → {base}.{{pdf,png}}", flush=True)


# ── Analysis runner ────────────────────────────────────────────────────────────


def run_analysis(all_features: bool, age_bmi: pd.DataFrame) -> pd.DataFrame:
    suffix = "_full" if all_features else "_fdr"
    tag = " (all features — robustness)" if all_features else " (FDR q<0.1)"

    print(f"\n{'='*60}", flush=True)
    print(f"Loading frozen set{tag}", flush=True)
    targets_by_sex = load_frozen_set(all_features=all_features)
    for sex in ("M", "F"):
        n_sys = len({t.system for t in targets_by_sex[sex]})
        print(
            f"  {sex}: {len(targets_by_sex[sex])} features across {n_sys} systems",
            flush=True,
        )

    all_rows: list[dict] = []
    for sex in ("M", "F"):
        print(f"\n  → Processing sex={sex}", flush=True)
        rows = compute_feature_seed_rows(sex, targets_by_sex[sex], age_bmi)
        all_rows.extend(rows)

    if not all_rows:
        print("  No rows computed — check prediction paths.", flush=True)
        return pd.DataFrame()

    feature_df = pd.DataFrame(all_rows)
    feat_csv = os.path.join(OUT_DIR, f"stratified_subgroup_delta_r_feature_level{suffix}.csv")
    feature_df.to_csv(feat_csv, index=False)
    print(f"\n  Feature-level CSV: {feat_csv}  ({len(feature_df)} rows)", flush=True)

    system_df = build_system_summary(feature_df)
    sys_csv = os.path.join(OUT_DIR, f"stratified_subgroup_delta_r{suffix}.csv")
    system_df.to_csv(sys_csv, index=False)
    print(f"  System-level CSV:  {sys_csv}  ({len(system_df)} rows)", flush=True)

    if not all_features:
        print("\nGenerating figures...", flush=True)
        cohort_ref = load_cohort_ref(all_features=False)
        feat_level_csv = os.path.join(OUT_DIR, f"stratified_subgroup_delta_r_feature_level{suffix}.csv")
        feature_df_plot = pd.read_csv(feat_level_csv) if os.path.exists(feat_level_csv) else None
        for st in ("age", "bmi"):
            plot_bar(system_df, st, cohort_ref=cohort_ref, feature_df=feature_df_plot)
            plot_heatmap(system_df, st)

    return system_df


def replot_from_csv(csv_path: str | None = None) -> None:
    """Regenerate bar + heatmap figures from an already-saved system-level CSV.

    Useful for tweaking plot aesthetics without re-running the full computation.
    Defaults to the FDR CSV in OUT_DIR.
    """
    if csv_path is None:
        csv_path = os.path.join(_RESULTS_DIR, "stratified_subgroup_delta_r_fdr.csv")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Replotting from {csv_path}", flush=True)
    system_df = pd.read_csv(csv_path)
    # apply canonical system names to saved CSVs (in case they were written before the merge)
    if "body_system" in system_df.columns:
        system_df["body_system"] = system_df["body_system"].map(canonical_system)
        system_df = (
            system_df.groupby(["sex", "stratum_type", "stratum", "body_system"], as_index=False)
            .agg(
                n_features=("n_features", "sum"),
                n_features_with_data=("n_features_with_data", "sum"),
                n_participants_in_band=("n_participants_in_band", "median"),
                median_delta_r=("median_delta_r", "median"),
                dispersion_across_features=("dispersion_across_features", "mean"),
                dispersion_across_splits=("dispersion_across_splits", "mean"),
            )
        )
    cohort_ref = load_cohort_ref(all_features=False)
    feat_level_csv = csv_path.replace("_fdr.csv", "_feature_level_fdr.csv")
    feature_df = pd.read_csv(feat_level_csv) if os.path.exists(feat_level_csv) else None
    if feature_df is None:
        print("  (no feature-level CSV found — dots will be omitted)", flush=True)
    else:
        feature_df = feature_df[~feature_df["label"].isin(EXCLUDED_LABELS)]
        if "system" in feature_df.columns:
            feature_df["system"] = feature_df["system"].map(canonical_system)
    for st in ("age", "bmi"):
        plot_bar(system_df, st, cohort_ref=cohort_ref, feature_df=feature_df)
        plot_heatmap(system_df, st)
    plot_figure4_grid(system_df, feature_df=feature_df, cohort_ref=cohort_ref)
    if feature_df is not None:
        plot_robustness_table(feature_df, system_df)
    print("Done.", flush=True)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading age/BMI covariates...", flush=True)
    age_bmi = load_age_bmi()
    print(
        f"  {len(age_bmi)} rows, {age_bmi['RegistrationCode'].nunique()} unique subjects",
        flush=True,
    )

    # Primary: FDR-significant set + figures
    run_analysis(all_features=False, age_bmi=age_bmi)

    # Robustness: full feature set (CSV only, no figures)
    run_analysis(all_features=True, age_bmi=age_bmi)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    # Public repo: regenerate figures from the shipped de-identified delta-r tables
    # (the subject-level computation in main() runs only on the cluster).
    replot_from_csv()
