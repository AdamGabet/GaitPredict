"""Build the Source Data tables for the figures (NMED-A150882A author checklist).

One directory per figure, one CSV per panel (plus a `*_per_seed` companion wherever
the panel summarises repeated cross-validation runs). Every file carries the exact
values that are drawn -- bar heights, box statistics, spoke radii, heatmap cells --
together with the exact P values and the n behind each of them, so a legend can say
"exact P values are provided in the Source Data" instead of printing them on the
artwork.

Correctness note: each figure applies its OWN filters and display names, and several
figures read the SAME results CSV with a different `model` value (Extended Data
Fig. 1 uses model='gait_no_room_A' while Figure 4 uses model='long_seq', both out of
gait_vs_confounders_pearson.csv). Re-implementing any of that here would risk
silently exporting the wrong rows, so this script imports each figure's own
selection functions and exports exactly what that figure plots, in plot order.

Run from the repo root with the panel-script environment (needs statsmodels):
    ~/miniconda3/envs/NewtonModels/bin/python plotting/build_source_data.py
"""
import importlib.util
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "results")
OUT_DIR = os.path.join(ROOT, "source_data")

PEARSON = os.path.join(RESULTS, "gait_vs_confounders_pearson.csv")
FDR_THRESHOLD = 0.1

_WRITTEN = []


def _import(name, relpath):
    """Import a figure script by path, with its own sys.path shims in place."""
    path = os.path.join(ROOT, relpath)
    d = os.path.dirname(path)
    for extra in (d, os.path.join(d, ".."), os.path.join(d, "..", "..")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tidy_integers(df):
    """Write counts as integers, not 3405.0 -- nullable so blanks stay blank."""
    for col in df.columns:
        if not (col.startswith("n_") or col in ("seed", "fold", "spoke_index", "epoch",
                                                "x_position", "bar_order", "row_order",
                                                "list_position", "row_order", "legend_order",
                                                "rank_in_system", "row_in_section")):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any() and (values.dropna() % 1 == 0).all():
            df[col] = values.astype("Int64")
    return df


def _write(df, figure_dir, stem, note):
    """Write one Source Data table under source_data/<figure_dir>/."""
    df = _tidy_integers(df.copy())
    d = os.path.join(OUT_DIR, figure_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{stem}.csv")
    df.to_csv(path, index=False)
    rel = os.path.join(figure_dir, f"{stem}.csv")
    _WRITTEN.append(rel)
    print(f"  {rel:<62} {len(df):>5} rows   {note}")


def _stats_lookup(df, keys=("system", "label")):
    """(system, label) -> row, asserting the key is unique."""
    assert not df.duplicated(list(keys)).any(), "non-unique statistics key"
    return df.set_index(list(keys))


# ---------------------------------------------------------------- Figure 1 ---
def figure_1():
    """1c: the training-loss curve (one point per epoch)."""
    loss = _import("sd_f1_loss", "plotting/figure_1/loss_plot.py")
    epoch_avg, loss_col = loss.load_epoch_averages()

    out = pd.DataFrame({
        "panel": "c",
        "epoch": epoch_avg["epoch"].astype(int),
        "reconstruction_loss_m": epoch_avg[loss_col],
        "plotted_y_natural_log_of_loss": epoch_avg["loss_log"],
        "n_logged_steps_averaged": epoch_avg["n_steps"].astype(int),
    })
    _write(out, "Figure_1", "Fig1c_training_loss",
           f"mean L_3D_pos per epoch, {loss.STEPS_PER_EPOCH} steps/epoch")


# ---------------------------------------------------------------- Figure 2 ---
def figure_2():
    """2a sex-classification AUC, 2b activity one-vs-all AUC, 2c-h prediction r."""
    roc = _import("sd_f2_roc", "plotting/figure_2/individual_plots/plot_gender_roc.py")
    umap = _import("sd_f2_umap", "plotting/figure_2/individual_plots/umap_activity_embeddings.py")
    asbv = _import("sd_f2_asbv", "plotting/figure_2/individual_plots/plot_asbv.py")

    # --- 2a: one ROC curve per activity; the legend prints the AUC -----------
    summary = pd.read_csv(os.path.join(RESULTS, "gender_auc_summary.csv"))
    seeds = pd.read_csv(os.path.join(RESULTS, "gender_auc_seed_values.csv"))

    # The plotted curve is computed once on the seed-averaged prediction per
    # subject-visit, so check the exported number against the shipped curve.
    curve = pd.read_csv(os.path.join(RESULTS, "roc_gender.csv")).groupby("activity")["auc"].first()
    merged = summary.set_index("activity")["auc_pooled"]
    assert np.allclose(merged.loc[curve.index].values, curve.values), "2a AUC differs from roc_gender.csv"

    summary = summary.sort_values("auc_pooled", ascending=False)  # legend order
    summary.insert(0, "series", summary["activity"].map(roc.ACTIVITY_NAMES))
    summary.insert(0, "legend_order", range(1, len(summary) + 1))
    summary.insert(0, "panel", "a")
    summary = summary.rename(columns={"activity": "series_key",
                                      "auc_pooled": "auc_plotted"})
    _write(summary[["panel", "legend_order", "series", "series_key", "auc_plotted",
                    "auc_seed_mean", "auc_seed_sd", "n_seeds", "n_samples",
                    "n_subjects", "n_male", "n_female"]],
           "Figure_2", "Fig2a_sex_classification_auc",
           "AUC printed in the legend + the seed distribution it summarises")

    seeds = seeds.merge(summary[["series_key", "series", "legend_order"]],
                        left_on="activity", right_on="series_key")
    seeds.insert(0, "panel", "a")
    _write(seeds.sort_values(["legend_order", "seed"])[
               ["panel", "series", "series_key", "seed", "auc", "n_samples", "n_positive"]],
           "Figure_2", "Fig2a_sex_classification_auc_per_seed",
           "AUC of every individual cross-validation seed")

    # --- 2b: one-vs-all activity classification (the AUCs in the UMAP legend) -
    folds = pd.read_csv(os.path.join(RESULTS, "activity_one_vs_all_auc_folds.csv"))
    agg = (folds.groupby("target_activity")
                .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                     accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                     n_seeds=("seed", "nunique"), n_folds=("fold", "count"))
                .reset_index())
    # The folds of one seed partition the data, so summing them gives the class
    # sizes; every seed must see the same ones.
    totals = folds.groupby(["target_activity", "seed"])[["n_positive", "n_negative"]].sum()
    assert (totals.groupby("target_activity").nunique() == 1).all().all(), "class sizes differ by seed"
    totals = totals.groupby("target_activity").first().astype(int)
    agg = agg.merge(totals, on="target_activity")
    agg["auc_in_legend"] = agg["target_activity"].map(umap.CLASSIFICATION_AUC)
    for _, r in agg.iterrows():
        assert round(r["auc_mean"], 4) == r["auc_in_legend"], f"2b AUC differs for {r['target_activity']}"
    agg.insert(0, "activity", agg["target_activity"].map(umap.ACTIVITY_NAMES))
    agg.insert(0, "panel", "b")
    agg = agg.rename(columns={"target_activity": "activity_key",
                              "auc_mean": "auc_plotted"})
    agg = agg[["panel", "activity", "activity_key", "auc_plotted", "auc_in_legend",
               "auc_sd", "accuracy_mean", "accuracy_sd", "n_seeds", "n_folds",
               "n_positive", "n_negative"]]
    _write(agg.sort_values("auc_plotted", ascending=False),
           "Figure_2", "Fig2b_activity_one_vs_all_auc",
           "AUC printed in the UMAP legend (mean over seeds x folds)")

    folds = folds.merge(agg[["activity_key", "activity"]],
                        left_on="target_activity", right_on="activity_key")
    folds.insert(0, "panel", "b")
    _write(folds.sort_values(["activity_key", "seed", "fold"])[
               ["panel", "activity", "activity_key", "seed", "fold", "auc",
                "accuracy", "n_positive", "n_negative"]],
           "Figure_2", "Fig2b_activity_one_vs_all_auc_per_fold",
           "every seed x fold behind those means")

    # --- 2c-h: Gait Fusion vs single activities, embeddings vs features ------
    metrics, movement, seed_vals = asbv.load_data()

    # P values live in two tables and neither is the one the bars are read from:
    # the embedding series in asbv_only_gait_pearson.csv, the per-activity feature
    # series in movement_features_asbv.csv. Key them the way the figure keys bars.
    emb_p = pd.read_csv(os.path.join(RESULTS, "asbv_only_gait_pearson.csv"))
    emb_p = emb_p[emb_p["score_type"] == "pearson_r"]
    emb_p = _stats_lookup(emb_p, ("label", "sub_model", "gender"))
    mov_p = _stats_lookup(pd.read_csv(os.path.join(RESULTS, "movement_features_asbv.csv")),
                          ("label", "model", "gender"))

    panel_of = {("age", "male"): "c", ("age", "female"): "d",
                ("bmi", "male"): "e", ("bmi", "female"): "f",
                ("total_scan_vat_area", "male"): "g", ("total_scan_vat_area", "female"): "h"}

    bars, per_seed = [], []
    for (target, gender), panel in panel_of.items():
        # Bar groups in drawing order: Gait Fusion first, then the five activities.
        groups = [("Gait Fusion", "ensemble", "gait", "ensemble")]
        groups += [(asbv.ACTIVITY_NAMES[a].replace("\n", " "), a, a,
                    asbv.ACTIVITY_TO_MOVEMENT_MODEL[a]) for a in asbv.ACTIVITY_ORDER]

        for x_pos, (x_label, emb_key, mov_metrics_key, mov_model) in enumerate(groups, start=1):
            emb_mean, emb_sd = asbv.get_val(metrics, target, emb_key, gender)
            emb_seeds = asbv.get_seed_vals(seed_vals, target, emb_key, gender, "embeddings")
            mov_seeds = asbv.get_seed_vals(seed_vals, target, emb_key, gender, "features")
            if emb_key == "ensemble":
                mov_mean, _ = asbv.get_val(metrics, target, "gait", gender)
                mov_pval = np.nan          # no test shipped for the feature ensemble
                mov_n = np.nan
            else:
                mov_mean = asbv.get_movement_val(movement, target, emb_key, gender)
                row = mov_p.loc[(target, mov_model, gender)]
                mov_pval, mov_n = row["score_pvalue"], row["n_subjects"]
            mov_sd = float(np.std(mov_seeds, ddof=1)) if mov_seeds is not None else np.nan

            erow = emb_p.loc[(target, emb_key, gender)]
            for series, value, sd, pval, n_sub, vals in (
                    ("Gait Embeddings", emb_mean, emb_sd, erow["score_pvalue"], erow["n_subjects"], emb_seeds),
                    ("Gait Features", mov_mean, mov_sd, mov_pval, mov_n, mov_seeds)):
                bars.append(dict(
                    panel=panel, target=target, target_display=asbv.TARGETS[target]["long"],
                    gender=gender, x_position=x_pos, x_label=x_label, series=series,
                    pearson_r=value, error_bar_sd=sd, score_pvalue=pval,
                    n_seeds=(0 if vals is None else len(vals)), n_subjects=n_sub))
                if vals is None:
                    continue
                for seed, v in enumerate(vals):
                    per_seed.append(dict(panel=panel, target=target, gender=gender,
                                         x_position=x_pos, x_label=x_label,
                                         series=series, seed=seed, pearson_r=v))

    bars = pd.DataFrame(bars).sort_values(["panel", "x_position", "series"])
    _write(bars, "Figure_2", "Fig2c_h_prediction_r",
           "bar height (mean r) + error bar (SD over seeds) + exact P")
    _write(pd.DataFrame(per_seed).sort_values(["panel", "x_position", "series", "seed"]),
           "Figure_2", "Fig2c_h_prediction_r_per_seed",
           "the dots on the bars: one point per seed")


# ------------------------------------------------------- Figures 3 / ED 3 ---
def _radar_spokes(r3, frame, selection_frame=None):
    """Rows for the spokes the radar draws, in clockwise plot order from 12 o'clock."""
    picked = r3.get_top_labels_by_system(
        selection_frame if selection_frame is not None else frame,
        n_per_system=5, exclude_systems=None)
    rows = []
    for spoke, (system, label, _) in enumerate(picked):
        sel = frame[(frame["system"] == system) & (frame["label"] == label)]
        assert len(sel) == 1, f"expected one row for {system}/{label}, got {len(sel)}"
        row = sel.iloc[0].to_dict()
        row["spoke_index"] = spoke
        row["system_display"] = r3.SYSTEM_RENAME_OVERRIDES.get(
            system, r3.SYSTEM_RENAME_DICT.get(system, system.replace("_", " ").title()))
        row["label_display"] = r3.rename_label(label)
        rows.append(row)
    return pd.DataFrame(rows)


RADAR_COLS = ["spoke_index", "system", "system_display", "label", "label_display",
              "pearson_r", "score_pvalue", "score_pvalue_fdr",
              "n_seeds", "n_subjects", "description"]

GAIT_ONLY_N = os.path.join(RESULTS, "gait_only_n_subjects_by_gender.csv")
CONFOUNDERS_N = os.path.join(RESULTS, "gait_vs_confounders_n_subjects_by_gender.csv")
_CORRECTED_CACHE = {}


def _true_subject_counts(df, counts_csv):
    """The corrected n_subjects for every row of a results CSV, as a Series.

    The n_subjects that ships in both results CSVs is wrong twice over, because
    compare_results_gait.py fills it with the single pooled number in metrics.csv:

      1. that number is copied unchanged onto the 'all', 'male' and 'female' rows,
         so the sex-split rows report the whole cohort rather than their own half;
      2. for the gait runs the pooled number is itself a count of test rows, not of
         people -- one row per (participant, research_stage), so a participant
         scanned at two visits is counted twice (waist: 3405 rows, 3328 people).

    The *_n_subjects_by_gender.csv tables recount both honestly, from the
    per-subject predictions the models scored, with sex merged in from the 10K
    subject table (GaitPredict/review1/build_n_subjects_by_gender.py). Only the
    ensemble rows are recounted, which is all of Figs 3/4 and ED 1/3 draw; the rest
    keep the shipped value. Joined on the RAW (system, label, model) key, so this
    must happen before merge_systems() rewrites it.
    """
    counts = pd.read_csv(counts_csv).set_index(["system", "label", "model"])
    assert not counts.index.duplicated().any(), "non-unique key in counts table"

    key = pd.MultiIndex.from_frame(df[["system", "label", "model"]])
    is_ensemble = df["sub_model"] == "ensemble"

    n = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for gender in ("all", "male", "female"):
        rows = is_ensemble & (df["gender"] == gender)
        n[rows] = counts[f"n_subjects_{gender}"].reindex(key[rows]).to_numpy()

    missing = int(n[is_ensemble].isna().sum())
    assert missing == 0, f"{missing} ensemble rows have no recounted n_subjects"
    return n


def _with_true_subject_counts(df, counts_csv):
    """A results frame with n_subjects corrected. See _true_subject_counts()."""
    df = df.copy()
    df["n_subjects"] = _true_subject_counts(df, counts_csv).fillna(df["n_subjects"])
    return df


def _corrected_pearson_path(results_csv, counts_csv):
    """Path to a copy of results_csv with only the n_subjects column corrected.

    Figure 4 and ED Fig 1 reach their data through loaders that take a FILE PATH
    and call merge_systems() themselves, so the correction cannot be handed to them
    as a frame. Writing a corrected copy and passing that path keeps every one of
    their own filters and display rules intact -- re-implementing the selection here
    is exactly what this script exists to avoid.

    Every column is carried across as TEXT. Parsing the floats and writing them back
    would not round-trip: pandas' CSV reader is a fast, not correctly-rounded, float
    parser, so values drift by an ulp and reappear as spurious churn in the exported
    r and P values. Untouched columns therefore keep their original digits exactly.
    """
    if results_csv not in _CORRECTED_CACHE:
        raw = pd.read_csv(results_csv, dtype=str, keep_default_na=False,
                          na_filter=False, low_memory=False)
        n = _true_subject_counts(raw, counts_csv)
        raw["n_subjects"] = [
            str(int(v)) if pd.notna(v) else old
            for v, old in zip(n, raw["n_subjects"])
        ]
        fd, path = tempfile.mkstemp(prefix="corrected_n_", suffix=".csv")
        os.close(fd)
        raw.to_csv(path, index=False)
        _CORRECTED_CACHE[results_csv] = path
    return _CORRECTED_CACHE[results_csv]


def figures_3_and_ed3():
    """Radar of Pearson r per endpoint: Fig 3 (all subjects), ED Fig 3 (sex overlay).

    gait_only_pearson.csv carries no FDR column -- the correction is applied at plot
    time by significant_by_fdr(), so re-running it here is what puts an exact
    adjusted P against every spoke. get_top_labels_by_system() then picks the spokes
    that are actually drawn (<=5 per system, with the keyword limits and display-name
    de-duplication), so only plotted points are exported.

    n_subjects is recounted by _with_true_subject_counts() -- the value in the
    results CSV is neither sex-specific nor a subject count. See that docstring.
    """
    r3 = _import("sd_fig3", "plotting/figure_3/radar_gait_only.py")
    gait_only = _with_true_subject_counts(
        pd.read_csv(os.path.join(RESULTS, "gait_only_pearson.csv")), GAIT_ONLY_N)
    gait_only = r3.merge_systems(gait_only)

    fig3 = _radar_spokes(r3, r3.significant_by_fdr(gait_only, "all"))
    fig3 = fig3.rename(columns={"score": "pearson_r"})
    _write(fig3[RADAR_COLS], "Figure_3", "Fig3_radar_endpoints",
           f"gender=all, BH-FDR q<{r3.FDR_Q}, clockwise from 12 o'clock")

    # ED Fig. 3 overlays the sexes and keeps only endpoints significant in BOTH,
    # with male used as the reference for spoke order (as the figure does).
    ed3_m = r3.significant_by_fdr(gait_only, "male")
    ed3_f = r3.significant_by_fdr(gait_only, "female")
    common = set(ed3_m["label"]) & set(ed3_f["label"])
    ref = ed3_m[ed3_m["label"].isin(common)].copy()
    male = _radar_spokes(r3, ref, selection_frame=ref)
    female = _radar_spokes(r3, ed3_f[ed3_f["label"].isin(common)].copy(), selection_frame=ref)

    # One row per spoke with both lines on it, rather than two files to cross-read.
    ed3 = male[["spoke_index", "system", "system_display", "label", "label_display",
                "description", "score", "score_pvalue", "score_pvalue_fdr",
                "n_seeds", "n_subjects"]].rename(columns={
        "score": "pearson_r_male", "score_pvalue": "score_pvalue_male",
        "score_pvalue_fdr": "score_pvalue_fdr_male",
        "n_seeds": "n_seeds_male", "n_subjects": "n_subjects_male"})
    f = female.set_index("spoke_index")
    ed3["pearson_r_female"] = ed3["spoke_index"].map(f["score"])
    ed3["score_pvalue_female"] = ed3["spoke_index"].map(f["score_pvalue"])
    ed3["score_pvalue_fdr_female"] = ed3["spoke_index"].map(f["score_pvalue_fdr"])
    ed3["n_seeds_female"] = ed3["spoke_index"].map(f["n_seeds"])
    ed3["n_subjects_female"] = ed3["spoke_index"].map(f["n_subjects"])
    _write(ed3[["spoke_index", "system", "system_display", "label", "label_display",
                "pearson_r_male", "pearson_r_female",
                "score_pvalue_male", "score_pvalue_fdr_male",
                "score_pvalue_female", "score_pvalue_fdr_female",
                "n_seeds_male", "n_subjects_male", "n_seeds_female", "n_subjects_female",
                "description"]],
           "Extended_Data_Figure_3", "EDFig3_radar_endpoints",
           f"BH-FDR q<{r3.FDR_Q} in both sexes, male spoke order")


# ----------------------------------------------------- Figures 4 / ED 1 -----
BOX_POINT_COLS = ["panel", "gender", "system", "system_display", "label",
                  "pearson_r", "baseline_pearson_r", "pearson_r_improvement",
                  "score_pvalue", "wilcox_pvalue", "wilcox_pvalue_fdr",
                  "n_seeds", "n_subjects", "description"]


def _box_points(gdata, panel, gender, rename):
    """One row per dot in a box column, with the exact statistics behind it."""
    df = gdata["filt"].copy()
    order = {s: i for i, s in enumerate(gdata["systems"], start=1)}
    df["panel"] = panel
    df["gender"] = gender
    df["system_display"] = df["system"].map(
        lambda s: rename.get(s, s.replace("_", " ").title()))
    df["x_position"] = df["system"].map(order)
    df = df.rename(columns={"score": "pearson_r", "baseline_score": "baseline_pearson_r",
                            "delta": "pearson_r_improvement"})
    cols = [c for c in BOX_POINT_COLS if c in df.columns]
    cols = cols[:2] + ["x_position"] + cols[2:]
    return df.sort_values(["x_position", "label"])[cols]


def _box_statistics(gdata, panel, gender, rename, rows):
    """The drawn box itself: median, quartiles and whiskers per system."""
    out = []
    for value_col, drawn_as in (("delta", "Pearson r improvement (delta from baseline)"),
                                ("score", "Pearson r")):
        for x, system in enumerate(gdata["systems"], start=1):
            vals = gdata["filt"].loc[gdata["filt"]["system"] == system, value_col].values
            st = boxplot_stats(vals, whis=1.5)[0]
            out.append(dict(
                panel=panel, gender=gender, box_row=drawn_as, x_position=x,
                system=system,
                system_display=rename.get(system, system.replace("_", " ").title()),
                n_endpoints_in_box=int(len(vals)),
                n_significant=int(gdata["sig_counts"].get(system, 0)),
                n_tested=int(gdata["total_counts"].get(system, 0)),
                median=st["med"], q1=st["q1"], q3=st["q3"],
                whisker_low=st["whislo"], whisker_high=st["whishi"],
                minimum=float(np.min(vals)), maximum=float(np.max(vals))))
    rows.extend(out)


def figure_4():
    """4a/4c boxplots of every significant endpoint, 4b/4d the biomarker bars."""
    # make_figure_4_grid owns both halves of the figure: importing it (rather than
    # its two component modules) also picks up the display-name overrides it adds.
    f4 = _import("sd_fig4", "plotting/figure_4_grid/make_figure_4_grid.py")
    from publication_colors import SYSTEM_RENAME_DICT, merge_systems

    # n_subjects in the shipped CSV is neither sex-specific nor a subject count;
    # see _with_true_subject_counts(). Everything else is read unchanged.
    pearson = _corrected_pearson_path(PEARSON, CONFOUNDERS_N)

    panels = {"male": "a", "female": "c"}
    points, stats = [], []
    for gender, panel in panels.items():
        gdata = f4.load_gender_data(pearson, "long_seq", gender, fdr_threshold=FDR_THRESHOLD)
        points.append(_box_points(gdata, panel, gender, SYSTEM_RENAME_DICT))
        _box_statistics(gdata, panel, gender, SYSTEM_RENAME_DICT, stats)

    _write(pd.concat(points), "Figure_4", "Fig4a_c_boxplot_points",
           f"every dot in 4a/4c: FDR<{FDR_THRESHOLD} endpoints, model=long_seq")
    _write(pd.DataFrame(stats), "Figure_4", "Fig4a_c_box_statistics",
           "the boxes themselves + the (significant/tested) counts on the x axis")

    # --- 4b / 4d: two biomarkers per system, baseline bar vs gait bar ---------
    stats_src = merge_systems(pd.read_csv(pearson, low_memory=False))
    unified = f4.get_unified_system_order(pearson, "long_seq")

    bars, per_seed = [], []
    for gender, panel in (("male", "b"), ("female", "d")):
        bio, _ = f4.get_top_biomarkers_combined(
            pearson, "long_seq", gender, top_n=2, candidate_top_k=5,
            use_manual_selection=True, fixed_system_order=unified)
        look = _stats_lookup(stats_src[
            (stats_src["sub_model"] == "ensemble") & (stats_src["model"] == "long_seq") &
            (stats_src["gender"] == gender) & (stats_src["score_type"] == "pearson_r")])

        for order, (_, row) in enumerate(bio.iterrows(), start=1):
            key = (row["system"], row["label"])
            st = look.loc[key]
            base_seeds, gait_seeds = f4.load_seed_values(None, row["system"], row["label"],
                                                         "long_seq", gender)
            bars.append(dict(
                panel=panel, gender=gender, bar_order=order, system=row["system"],
                system_display=SYSTEM_RENAME_DICT.get(row["system"],
                                                      row["system"].replace("_", " ").title()),
                label=row["label"], label_display=f4.pretty_label(row["label"]),
                baseline_pearson_r=row["baseline_score"], gait_pearson_r=row["score"],
                pearson_r_improvement=row["delta"],
                baseline_seed_sd=(np.std(base_seeds, ddof=1) if base_seeds is not None else np.nan),
                gait_seed_sd=(np.std(gait_seeds, ddof=1) if gait_seeds is not None else np.nan),
                score_pvalue=st["score_pvalue"], wilcox_pvalue=st["wilcox_pvalue"],
                wilcox_pvalue_fdr=st["wilcox_pvalue_fdr"],
                n_seeds=st["n_seeds"], n_subjects=st["n_subjects"],
                description=st.get("description", "")))
            if base_seeds is None:
                continue
            for seed, (b, g) in enumerate(zip(base_seeds, gait_seeds)):
                per_seed.append(dict(panel=panel, gender=gender, bar_order=order,
                                     system=row["system"], label=row["label"],
                                     label_display=f4.pretty_label(row["label"]),
                                     seed=seed, baseline_pearson_r=b, gait_pearson_r=g))

    _write(pd.DataFrame(bars), "Figure_4", "Fig4b_d_biomarker_bars",
           "the two bars per biomarker, top to bottom as drawn")
    _write(pd.DataFrame(per_seed), "Figure_4", "Fig4b_d_biomarker_bars_per_seed",
           "the dots on those bars: one point per seed")


def extended_data_figure_1():
    """Same boxplots as Fig 4a/4c for the gait-feature baseline model."""
    ed1 = _import("sd_ed1", "plotting/extended_figure_1/publication_figure_gm.py")
    from publication_colors import SYSTEM_RENAME_DICT

    pearson = _corrected_pearson_path(PEARSON, CONFOUNDERS_N)

    points, stats = [], []
    for gender, panel in (("male", "Male"), ("female", "Female")):
        filt, orig = ed1.load_filtered_data(pearson, "gait_no_room_A", gender,
                                            "pearson_r", fdr_threshold=FDR_THRESHOLD)
        systems, sig, total = ed1.get_system_order_and_counts(filt, orig, "delta")
        gdata = dict(filt=filt, systems=systems, sig_counts=sig, total_counts=total)
        points.append(_box_points(gdata, panel, gender, SYSTEM_RENAME_DICT))
        _box_statistics(gdata, panel, gender, SYSTEM_RENAME_DICT, stats)

    _write(pd.concat(points), "Extended_Data_Figure_1", "EDFig1_boxplot_points",
           f"every dot: FDR<{FDR_THRESHOLD} endpoints, model=gait_no_room_A")
    _write(pd.DataFrame(stats), "Extended_Data_Figure_1", "EDFig1_box_statistics",
           "the boxes themselves + the (significant/tested) counts on the x axis")


# ---------------------------------------------------------------- Figure 5 ---
def figure_5():
    """5a/5b: medical conditions and medications, baseline AUC -> gait AUC."""
    f5 = _import("sd_fig5", "plotting/figure_5_medical/medical_conditions_dumbbell.py")
    med = pd.read_csv(os.path.join(RESULTS, "medical_conditions_pearson.csv"), low_memory=False)

    rows = []
    for gender, panel in (("male", "a"), ("female", "b")):
        sel, _ = f5.select_rows_for_gender(med, gender)
        # select_rows_for_gender rebuilds its rows with display fields only, so the
        # exact statistics have to be re-attached from the source table. Restrict the
        # source to the same panel first so the join stays one-to-one.
        look = _stats_lookup(med[(med["model"] == f5.MODEL) & (med["gender"] == gender) &
                                 (med["sub_model"] == "ensemble") &
                                 (med["score_type"] == "auc")])
        section_counter = {}
        for order, (_, row) in enumerate(sel.iterrows(), start=1):
            st = look.loc[(row["system"], row["label"])]
            section = "MEDICAL CONDITIONS" if row["system"] == "medical_conditions" else "MEDICATIONS"
            section_counter[section] = section_counter.get(section, 0) + 1
            seed_columns = {}
            for seed, (b, g) in enumerate(zip(row["baseline_seed"], row["gait_seed"])):
                seed_columns[f"seed_{seed}_baseline_auc"] = b
                seed_columns[f"seed_{seed}_gait_auc"] = g

            rows.append(dict(
                panel=panel, gender=gender, row_order=order, section=section,
                row_in_section=section_counter[section],
                system=row["system"], label=row["label"], label_display=row["label_pretty"],
                baseline_auc=row["baseline"], gait_auc=row["gait"],
                delta_auc_printed=row["delta"],
                score_pvalue=st["score_pvalue"], wilcox_pvalue=st["wilcox_pvalue"],
                wilcox_pvalue_fdr=st["wilcox_pvalue_fdr"],
                n_seeds=st["n_seeds"], n_positive=row["n_positive"],
                n_in_panel_title=f5.COHORT_N[gender],
                description=st.get("description", ""), **seed_columns))

    rows = pd.DataFrame(rows)
    assert rows["wilcox_pvalue_fdr"].notna().all(), "missing P value after join"
    _write(rows, "Figure_5", "Fig5a_b_dumbbell_points",
           f"both dots, printed delta-AUC, and raw per-seed AUC columns, FDR<{f5.FDR_THRESHOLD}")


# ---------------------------------------------------------------- Figure 6 ---
def figure_6():
    """6a body-system x body-group heatmap, 6b/6c the ranked label lists."""
    f6a = _import("sd_fig6a", "plotting/figure_6/individual_plots/plot_body_system_heatmap.py")
    f6b = _import("sd_fig6b", "plotting/figure_6/individual_plots/plot_grouped_by_body_part.py")
    masking = os.path.join(RESULTS, "masking_ablation_pearson.csv")
    groups = f6a.BODY_GROUPS

    # --- 6a: the shipped panel is the both-sexes heatmap (top-10 per system) --
    df = f6a.load_and_prepare(masking, score_type="pearson_r")
    male_li = f6a.build_label_importance_table(df, gender="male", model_prefix="long")
    female_li = f6a.build_label_importance_table(df, gender="female", model_prefix="long")
    male_li.insert(0, "gender", "male")
    female_li.insert(0, "gender", "female")
    both_li = pd.concat([male_li, female_li], ignore_index=True)

    raw = f6a.aggregate_topk_by_system(both_li.drop(columns=["gender"]), top_k=10)
    heat = f6a.sort_by_top_group(f6a.normalize_rows_to_unit_interval(raw))

    cells = []
    for order, system in enumerate(heat.index, start=1):
        row = heat.loc[system, groups].astype(float)
        top = row.idxmax()
        for group in groups:
            cells.append(dict(
                row_order=order, system=system,
                system_display=f6a._system_display_name(system),
                body_group=group.title(),
                relative_importance=float(row[group]),          # the printed cell value
                mean_top10_importance=float(raw.loc[system, group]),
                is_row_max=bool(group == top)))                 # the black box on the cell
    _write(pd.DataFrame(cells), "Figure_6", "Fig6a_heatmap_cells",
           "every cell, top to bottom as drawn (both sexes, top-10 per system)")

    # The label-level importances the cells are the top-10 mean of.
    li = both_li.melt(id_vars=["gender", "system", "label"], value_vars=groups,
                      var_name="body_group", value_name="importance").dropna(subset=["importance"])
    li["system_display"] = li["system"].map(f6a._system_display_name)
    li["label_display"] = li["label"].map(f6b.get_label_display)
    li["rank_in_system"] = li.groupby(["system", "body_group"])["importance"].rank(
        ascending=False, method="first").astype(int)
    li["in_top10_mean"] = li["rank_in_system"] <= 10
    li["body_group"] = li["body_group"].str.title()
    order = {s: i for i, s in enumerate(heat.index, start=1)}
    li["row_order"] = li["system"].map(order)
    _write(li.sort_values(["row_order", "body_group", "rank_in_system"])[
               ["row_order", "system", "system_display", "body_group", "gender",
                "label", "label_display", "importance", "rank_in_system", "in_top10_mean"]],
           "Figure_6", "Fig6a_label_importance",
           "per-endpoint importance averaged over the two walking activities")

    # --- 6b / 6c: the ranked label lists under each skeleton -----------------
    raw_masking = pd.read_csv(masking)   # 6b/6c reads the table unfiltered, as the figure does
    listed, components = [], []
    manual = {("male", "b"): f6b.MANUAL_LABELS_PEARSON_MALE,
              ("female", "c"): f6b.MANUAL_LABELS_PEARSON_FEMALE}
    for (gender, panel), by_group in manual.items():
        for group in ["head", "arms", "torso", "legs"]:
            for position, label in enumerate(by_group.get(group, []), start=1):
                imp = f6b.compute_group_importance_averaged(
                    raw_masking, label=label, gender=gender,
                    model_prefix="long", activities=f6b.WALKING_ACTIVITIES) or {}
                listed.append(dict(
                    panel=panel, gender=gender, body_group=group.title(),
                    list_position=position, label=label,
                    label_display=f6b.get_label_display(label),
                    importance_in_listed_group=imp.get(group, np.nan),
                    data_top_group=(f6b.get_top_group(imp) or ""),
                    **{f"importance_{g}": imp.get(g, np.nan) for g in groups}))
                components.append((panel, gender, label))

    _write(pd.DataFrame(listed), "Figure_6", "Fig6b_c_ranked_labels",
           "the endpoints printed under each skeleton, in printed order")

    # The masking-ablation scores those importances are computed from, with the
    # exact P value of every underlying model. The importance itself is a
    # normalised contrast of these and carries no test of its own.
    # compute_group_importance_averaged() falls back to the pooled (gender='all')
    # row when a sex-specific one is missing and then takes the first match, so
    # the same rule is applied here and the consumed row is flagged.
    models = [("full model", "long_seq")]
    models += [(f"masked {g}", f"long_masked_{g}") for g in groups]
    models += [(f"all but {g}", f"long_all_but_{g}") for g in groups]

    comp = []
    for panel, gender, label in sorted(components):
        for activity in f6b.WALKING_ACTIVITIES:
            act_rows = raw_masking[(raw_masking["sub_model"] == activity) &
                                   (raw_masking["label"] == label)]
            for role, model in models:
                sel = act_rows[(act_rows["model"] == model) & (act_rows["gender"] == gender)]
                if sel.empty:
                    sel = act_rows[(act_rows["model"] == model) & (act_rows["gender"] == "all")]
                for i, (_, r) in enumerate(sel.iterrows()):
                    comp.append(dict(
                        panel=panel, gender_of_panel=gender, label=label,
                        label_display=f6b.get_label_display(label), system=r["system"],
                        activity=activity, role=role, model=model,
                        gender_of_row=r["gender"], score_type=r["score_type"],
                        pearson_r=r["score"], score_pvalue=r["score_pvalue"],
                        n_seeds=r["n_seeds"], n_subjects=r["n_subjects"],
                        used_by_figure=(i == 0), description=r["description"]))
    _write(pd.DataFrame(comp), "Figure_6", "Fig6_masking_ablation_components",
           "raw Pearson r per masked / unmasked model + exact P")


# -------------------------------------------------- Extended Data Figure 2 ---
def extended_data_figure_2():
    """Duration scaling with the intermittent-sampling overlay (treadmill 3 km/h)."""
    ed2 = _import("sd_ed2", "plotting/extended_figure_2/plot_intermittent_overlay.py")
    group = ed2.PLOT_GROUP
    targets = [t for t in ed2.TARGETS if t["group"] == group["name"]]
    names = {t["name"]: t["display"] for t in targets}
    # x_order as run_per_activity builds it; the 60 s point is deliberately not drawn.
    drawn_seconds = [2, 5, 10, 15, 30, 180]

    cont = pd.read_csv(os.path.join(RESULTS, "duration_scaling_plot_data.csv"))
    cont = cont[(cont["activity"] == "tm_3kmh") & (cont["target"].isin(names)) &
                (cont["gender"] == "all") & (cont["duration_seconds"].isin(drawn_seconds))].copy()
    cont["seed"] = cont.groupby(["target", "duration_seconds"]).cumcount()

    curves = (cont.groupby(["target", "duration_seconds"])["pearson_r"]
                  .agg(pearson_r_mean="mean", error_bar_sd="std", n_seeds="size")
                  .reset_index())
    curves["target_display"] = curves["target"].map(names)
    curves["x_tick_label"] = curves["duration_seconds"].astype(str) + "s"
    _write(curves[["target", "target_display", "duration_seconds", "x_tick_label",
                   "pearson_r_mean", "error_bar_sd", "n_seeds"]],
           "Extended_Data_Figure_2", "EDFig2_duration_curves",
           "each marker on the curves + its SD error bar (all subjects)")
    cont["target_display"] = cont["target"].map(names)
    _write(cont.sort_values(["target", "duration_seconds", "seed"])[
               ["target", "target_display", "duration_seconds", "seed", "pearson_r"]],
           "Extended_Data_Figure_2", "EDFig2_duration_curves_per_seed",
           "the seeds those means and SDs are taken over")

    # Intermittent overlay: marker at the seed mean, whisker at min-max.
    interm = ed2.load_intermittent_rows(group)
    interm = interm[interm["gender"] == "all"].copy()
    interm["seed"] = interm.groupby(["target", "interm_name"]).cumcount()
    meta = {m["name"]: m for m in ed2.models_for_group(group["name"])}
    pts = (interm.groupby(["target", "interm_name"])["pearson_r"]
                 .agg(pearson_r_mean="mean", whisker_min="min", whisker_max="max", n_seeds="size")
                 .reset_index())
    pts["protocol"] = pts["interm_name"].map(lambda n: meta[n]["label"])
    pts["total_seconds"] = pts["interm_name"].map(lambda n: meta[n]["total_seconds"])
    pts["x_position"] = pts["interm_name"].map(lambda n: meta[n].get("plot_x", meta[n]["total_seconds"]))
    pts["target_display"] = pts["target"].map(names)
    _write(pts[["target", "target_display", "protocol", "interm_name", "total_seconds",
                "x_position", "pearson_r_mean", "whisker_min", "whisker_max", "n_seeds"]],
           "Extended_Data_Figure_2", "EDFig2_intermittent_points",
           "the overlaid markers; whisker is min-max over seeds, not SD")
    interm["protocol"] = interm["interm_name"].map(lambda n: meta[n]["label"])
    interm["target_display"] = interm["target"].map(names)
    _write(interm.sort_values(["target", "interm_name", "seed"])[
               ["target", "target_display", "protocol", "interm_name", "seed", "pearson_r"]],
           "Extended_Data_Figure_2", "EDFig2_intermittent_points_per_seed",
           "the seeds behind each overlaid marker")


# -------------------------------------------------- Extended Data Figure 4 ---
def extended_data_figure_4():
    """Normalized keypoints vs GaitMAE embeddings, per sex."""
    ed4 = _import("sd_ed4", "plotting/extended_figure_4/normalized_vs_gaitmae_absolute_r_gain.py")
    seed_table = pd.read_csv(ed4.GAINS_CSV)
    per_seed = pd.read_csv(ed4.SEED_VALUES_CSV)
    compact = ed4.build_compact_table(seed_table).set_index(["target", "sex"])

    bars = []
    for target in ed4.TARGETS:
        for x_pos, sex in enumerate(ed4.SEXES, start=1):
            pair = compact.loc[(target, sex)]
            for model in (ed4.MODEL_KEYPOINTS, ed4.MODEL_GAITMAE):
                row = seed_table[(seed_table["target"] == target) &
                                 (seed_table["gender"] == sex) &
                                 (seed_table["model"] == model)].iloc[0]
                bars.append(dict(
                    panel=sex, target=target,
                    x_label=ed4.SHORT_LABELS[target].replace("\n", " "),
                    series=model, pearson_r=float(row["pearson_mean"]),
                    error_bar_sd=float(row["pearson_std"]), n_seeds=int(row["n_seeds"]),
                    delta_pearson_r=float(pair["delta_pearson_r"]),
                    percent_gain_printed=float(pair["percent_gain_vs_keypoints"])))
    _write(pd.DataFrame(bars), "Extended_Data_Figure_4", "EDFig4_keypoints_vs_gaitmae",
           "bar height, SD error bar and the +% gain printed above each pair")
    per_seed = per_seed.rename(columns={"gender": "panel", "value": "pearson_r",
                                        "model": "series"})
    _write(per_seed.sort_values(["panel", "target", "series", "seed"])[
               ["panel", "target", "series", "seed", "pearson_r"]],
           "Extended_Data_Figure_4", "EDFig4_keypoints_vs_gaitmae_per_seed",
           "the dots on those bars (keypoints n=5, GaitMAE n=15)")


# -------------------------------------------------- Extended Data Figure 5 ---
def extended_data_figure_5():
    """Liver elasticity: gait embeddings vs Age+BMI vs Age+BMI+liver blood panel."""
    from scipy.stats import wilcoxon
    ed5 = _import("sd_ed5", "plotting/extended_figure_5/plot_liver_gait_vs_bloodpanel.py")
    summary = pd.read_csv(ed5.SUMMARY_CSV)
    summary = summary[summary["setting"] == "full"]
    source_label = {key: label for key, label, _ in ed5.SOURCES}

    bars, per_seed, brackets = [], [], []
    for x_pos, gender in enumerate(("male", "female"), start=1):
        for key, label, _ in ed5.SOURCES:
            mean, sd = ed5.best_of(summary, "liver_elasticity", key, gender, "pearson_r")
            series = ed5.best_series(summary, "liver_elasticity", key, gender, "pearson_r") or {}
            bars.append(dict(panel="Male" if gender == "male" else "Female", gender=gender,
                             x_position=x_pos, series_key=key, series=label,
                             pearson_r=mean, error_bar_sd=sd, n_seeds=len(series)))
            for seed, value in sorted(series.items()):
                per_seed.append(dict(panel="Male" if gender == "male" else "Female",
                                     gender=gender, series_key=key, series=label,
                                     seed=seed, pearson_r=value))

        # The brackets: two-sided paired Wilcoxon over the seeds, as annotate_sig
        # computes them at draw time (nothing in results/ carries these).
        for a, b in (("gait", "demo"), ("gait", "clinical")):
            sa = ed5.best_series(summary, "liver_elasticity", a, gender, "pearson_r")
            sb = ed5.best_series(summary, "liver_elasticity", b, gender, "pearson_r")
            seeds = sorted(set(sa) & set(sb))
            av = np.array([sa[s] for s in seeds])
            bv = np.array([sb[s] for s in seeds])
            p = wilcoxon(av, bv).pvalue
            # _sig_label returns mathtext; render it the way the figure prints it.
            label = (ed5._sig_label(p).replace("$", "").replace("^", "")
                     .replace("{", "").replace("}", "").replace("-", "\u2212"))
            brackets.append(dict(panel="Male" if gender == "male" else "Female", gender=gender,
                                 comparison=f"{a} vs {b}",
                                 series_a=source_label[a], series_b=source_label[b],
                                 wilcoxon_pvalue=p, printed_label=label,
                                 test="two-sided paired Wilcoxon signed-rank over seeds",
                                 n_seeds=len(seeds)))
    _write(pd.DataFrame(bars), "Extended_Data_Figure_5", "EDFig5_liver_bars",
           "bar height (mean r over seeds) + SD error bar")
    _write(pd.DataFrame(per_seed), "Extended_Data_Figure_5", "EDFig5_liver_bars_per_seed",
           "the dots on those bars")
    _write(pd.DataFrame(brackets), "Extended_Data_Figure_5", "EDFig5_significance_brackets",
           "the exact P printed over each bracket")


# -------------------------------------------------- Extended Data Figure 6 ---
def extended_data_figure_6():
    """Clinical scenario: AUC dumbbells (a, c) and top-10% sensitivity dumbbells (b, d)."""
    ed6 = _import("sd_ed6", "plotting/extended_figure_6/individual_plots/make_clinical_scenario_panels.py")
    sens = pd.read_csv(ed6.SENS_CSV)
    utility = pd.read_csv(ed6.UTILITY_CSV)
    raw_seed_auc = pd.read_csv(ed6.SEED_AUC_CSV)

    auc_rows, auc_seeds, sens_rows, sens_seeds = [], [], [], []
    panels = {"male": ("a", "b"), "female": ("c", "d")}
    for gender, (auc_panel, sens_panel) in panels.items():
        selected = ed6.auc_endpoint_keys(sens, utility, gender)
        rows, headers = ed6._layout_rows(ed6._ordered_sections(selected, "delta_sensitivity_top10"))
        section_of = {}
        for title, part in ed6._ordered_sections(selected, "delta_sensitivity_top10"):
            for endpoint in part["endpoint"]:
                section_of[endpoint] = title

        for order, (_, row) in enumerate(rows, start=1):
            stat = sens[(sens["gender"] == gender) & (sens["endpoint"] == row["endpoint"]) &
                        (sens["system"] == row["system"])].iloc[0]
            common = dict(gender=gender, row_order=order, section=section_of[row["endpoint"]],
                          system=row["system"], endpoint=row["endpoint"],
                          label_display=row["display_endpoint"],
                          paired_t_pvalue=stat["paired_t_pvalue"],
                          paired_t_pvalue_fdr=stat["paired_t_pvalue_fdr"],
                          fdr_significant_gain_q_lt_0p1=bool(stat["fdr_significant_gain_q_lt_0p1"]),
                          auc_selection_fdr=stat["auc_selection_fdr"],
                          n_folds=stat["n_folds"], n_subjects_mean=stat["n_mean"],
                          n_positive_mean=stat["n_positive_mean"],
                          n_in_panel_title=ed6.SEX_N[gender])

            auc_rows.append(dict(panel=auc_panel, **common,
                                 baseline_auc_plotted=row["baseline_auc"],
                                 baseline_auc_raw=row["baseline_auc_raw"],
                                 gait_auc=row["gait_auc"],
                                 delta_auc_printed=row["delta_auc"]))
            sens_rows.append(dict(panel=sens_panel, **common,
                                  label_printed=(row["display_endpoint"] + " *"
                                                 if stat["fdr_significant_gain_q_lt_0p1"]
                                                 else row["display_endpoint"]),
                                  sensitivity_baseline_pct=100 * row["sensitivity_base_top10"],
                                  sensitivity_gait_pct=100 * row["sensitivity_gait_top10"],
                                  delta_pp_printed=100 * row["delta_sensitivity_top10"]))

            # AUC seed clouds: the panel floors the baseline seeds at 0.5, so give
            # both the drawn value and the raw one.
            raw = raw_seed_auc[(raw_seed_auc["system"] == row["system"]) &
                               (raw_seed_auc["endpoint"] == row["endpoint"]) &
                               (raw_seed_auc["gender"] == gender)].sort_values("seed_idx")
            for _, r in raw.iterrows():
                auc_seeds.append(dict(panel=auc_panel, gender=gender, row_order=order,
                                      system=row["system"], endpoint=row["endpoint"],
                                      label_display=row["display_endpoint"],
                                      seed=int(r["seed_idx"]),
                                      baseline_auc_raw=r["baseline_auc"],
                                      baseline_auc_plotted=max(r["baseline_auc"], 0.5),
                                      gait_auc=r["gait_auc"]))

            seed_rows = utility[(utility["gender"] == gender) &
                                (utility["endpoint"] == row["endpoint"]) &
                                (utility["system"] == row["system"])].sort_values("seed")
            for _, r in seed_rows.iterrows():
                sens_seeds.append(dict(panel=sens_panel, gender=gender, row_order=order,
                                       system=row["system"], endpoint=row["endpoint"],
                                       label_display=row["display_endpoint"],
                                       seed=int(r["seed"]),
                                       sensitivity_baseline_pct=100 * r["sens_base_top10"],
                                       sensitivity_gait_pct=100 * r["sens_gait_top10"]))

    _write(pd.DataFrame(auc_rows), "Extended_Data_Figure_6", "EDFig6a_c_auc_dumbbell",
           "both dots and the printed delta-AUC (baseline floored at 0.5 as drawn)")
    _write(pd.DataFrame(auc_seeds), "Extended_Data_Figure_6", "EDFig6a_c_auc_dumbbell_per_seed",
           "the AUC seed clouds, raw and as drawn")
    _write(pd.DataFrame(sens_rows), "Extended_Data_Figure_6", "EDFig6b_d_sensitivity_dumbbell",
           "both dots and the printed delta in percentage points; * = FDR q<0.1")
    _write(pd.DataFrame(sens_seeds), "Extended_Data_Figure_6",
           "EDFig6b_d_sensitivity_dumbbell_per_seed", "the sensitivity seed clouds")


# -------------------------------------------------- Extended Data Figure 7 ---
def extended_data_figure_7():
    """The 26-joint skeleton schematic: node positions and bones, as drawn.

    A schematic rather than a measurement, so there is nothing to test; the file
    is here so the drawn geometry is reproducible. Coordinates come from the
    figure's own provenance record (one frame of one A-pose recording, projected
    to the coronal plane, pelvis at the origin); the participant identifier in
    that record is deliberately not carried over.
    """
    import json
    prov = os.path.join(ROOT, "plotting", "extended_figure_7", "figures",
                        "skeleton_schematic_provenance.json")
    if not os.path.exists(prov):
        print("  (skipped Extended Data Figure 7 -- run plot_skeleton_schematic.py first)")
        return
    with open(prov) as fh:
        rec = json.load(fh)

    offsets = rec.get("display_offsets_mm", {})
    joints = []
    for index, name in enumerate(rec["joints"]):
        x, y = rec["projected_xy_mm"][name]
        dx, dy = offsets.get(name, [0.0, 0.0])
        joints.append(dict(node_index=index, joint_name=name,
                           k4abt_index=rec["original_k4abt_indices"][index],
                           x_mm=x, y_mm=y,
                           display_offset_x_mm=dx, display_offset_y_mm=dy,
                           drawn_x_mm=x + dx, drawn_y_mm=y + dy))
    _write(pd.DataFrame(joints), "Extended_Data_Figure_7", "EDFig7_skeleton_joints",
           "coronal projection, pelvis at the origin, millimetres")

    names = rec["joints"]
    bones = [dict(bone_index=i, parent_index=a, parent_joint=names[a],
                  child_index=b, child_joint=names[b])
             for i, (a, b) in enumerate(rec["bones"])]
    _write(pd.DataFrame(bones), "Extended_Data_Figure_7", "EDFig7_skeleton_bones",
           "the segments drawn between those nodes")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Source Data: the plotted values and their exact statistics, per panel\n")
    figure_1()
    figure_2()
    figures_3_and_ed3()
    figure_4()
    figure_5()
    figure_6()
    extended_data_figure_1()
    extended_data_figure_2()
    extended_data_figure_4()
    extended_data_figure_5()
    extended_data_figure_6()
    extended_data_figure_7()
    print(f"\nWrote {len(_WRITTEN)} files to {os.path.relpath(OUT_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
