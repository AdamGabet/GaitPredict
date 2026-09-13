# Source Data

One directory per figure, one CSV per panel. Each file holds **the exact values that
are drawn** — bar heights, box statistics, spoke radii, heatmap cells, curve points —
together with the **exact P values** and the **n** behind them, so the figure legends
can point here instead of printing dozens of P values on the artwork.

Files ending `_per_seed` / `_per_fold` are the individual cross-validation runs behind
a mean (the dots drawn on the bars, and the distributions that are summarised but not
drawn). Rebuild everything with:

```bash
~/miniconda3/envs/NewtonModels/bin/python plotting/build_source_data.py   # from the repo root
```

The builder imports each figure's own selection and display functions rather than
re-deriving the filters, so what is exported is what is plotted, in plot order.

---

## Contents

| File | Panel | What it is |
|---|---|---|
| `Figure_1/Fig1c_training_loss.csv` | 1c | Mean reconstruction loss per epoch (the plotted points) |
| `Figure_2/Fig2a_sex_classification_auc.csv` | 2a | AUC printed in each ROC legend entry, + its seed mean and SD |
| `Figure_2/Fig2a_sex_classification_auc_per_seed.csv` | 2a | AUC of every individual seed |
| `Figure_2/Fig2b_activity_one_vs_all_auc.csv` | 2b | One-vs-all activity AUC printed in the UMAP legend |
| `Figure_2/Fig2b_activity_one_vs_all_auc_per_fold.csv` | 2b | Every seed × fold behind those means |
| `Figure_2/Fig2c_h_prediction_r.csv` | 2c–h | Bar height (mean Pearson r), error bar (SD over seeds), exact P |
| `Figure_2/Fig2c_h_prediction_r_per_seed.csv` | 2c–h | The dots on those bars, one row per seed |
| `Figure_3/Fig3_radar_endpoints.csv` | 3 | One row per drawn spoke: plotted r, raw and FDR-adjusted P |
| `Figure_4/Fig4a_c_boxplot_points.csv` | 4a, 4c | Every dot: r, baseline r, Δ and all three P values per endpoint |
| `Figure_4/Fig4a_c_box_statistics.csv` | 4a, 4c | The boxes: median, quartiles, whiskers, and the x-axis counts |
| `Figure_4/Fig4b_d_biomarker_bars.csv` | 4b, 4d | The two bars per biomarker, top to bottom as drawn |
| `Figure_4/Fig4b_d_biomarker_bars_per_seed.csv` | 4b, 4d | The dots on those bars, one row per seed |
| `Figure_5/Fig5a_b_dumbbell_points.csv` | 5a, 5b | Baseline AUC, gait AUC, printed ΔAUC, exact P, n positive, and raw per-seed AUC columns |
| `Figure_6/Fig6a_heatmap_cells.csv` | 6a | Every heatmap cell, top to bottom as drawn, + which cell is the row max |
| `Figure_6/Fig6a_label_importance.csv` | 6a | The per-endpoint importances each cell is the top-10 mean of |
| `Figure_6/Fig6b_c_ranked_labels.csv` | 6b, 6c | The endpoints listed under each skeleton, in printed order |
| `Figure_6/Fig6_masking_ablation_components.csv` | 6b, 6c | Raw Pearson r of every masked / unmasked model + exact P |
| `Extended_Data_Figure_1/EDFig1_boxplot_points.csv` | ED 1 | As Fig 4a/4c, for the gait-feature baseline model |
| `Extended_Data_Figure_1/EDFig1_box_statistics.csv` | ED 1 | The boxes + the x-axis counts |
| `Extended_Data_Figure_2/EDFig2_duration_curves.csv` | ED 2 | Each marker on the duration curves + its SD error bar |
| `Extended_Data_Figure_2/EDFig2_duration_curves_per_seed.csv` | ED 2 | The seeds those means and SDs are taken over |
| `Extended_Data_Figure_2/EDFig2_intermittent_points.csv` | ED 2 | The overlaid intermittent markers + their min–max whiskers |
| `Extended_Data_Figure_2/EDFig2_intermittent_points_per_seed.csv` | ED 2 | The seeds behind each overlaid marker |
| `Extended_Data_Figure_3/EDFig3_radar_endpoints.csv` | ED 3 | One row per spoke with **both** sexes on it |
| `Extended_Data_Figure_4/EDFig4_keypoints_vs_gaitmae.csv` | ED 4 | Bar height, SD error bar and the +% gain printed above each pair |
| `Extended_Data_Figure_4/EDFig4_keypoints_vs_gaitmae_per_seed.csv` | ED 4 | The dots on those bars |
| `Extended_Data_Figure_5/EDFig5_liver_bars.csv` | ED 5 | Bar height (mean r over 15 seeds) + SD error bar |
| `Extended_Data_Figure_5/EDFig5_liver_bars_per_seed.csv` | ED 5 | The dots on those bars |
| `Extended_Data_Figure_5/EDFig5_significance_brackets.csv` | ED 5 | The exact P printed over each bracket |
| `Extended_Data_Figure_6/EDFig6a_c_auc_dumbbell.csv` | ED 6a, 6c | Both dots, the printed ΔAUC, and the P values |
| `Extended_Data_Figure_6/EDFig6a_c_auc_dumbbell_per_seed.csv` | ED 6a, 6c | The AUC seed clouds, raw **and** as drawn |
| `Extended_Data_Figure_6/EDFig6b_d_sensitivity_dumbbell.csv` | ED 6b, 6d | Both dots, the printed Δ in percentage points, and the P values |
| `Extended_Data_Figure_6/EDFig6b_d_sensitivity_dumbbell_per_seed.csv` | ED 6b, 6d | The sensitivity seed clouds |
| `Extended_Data_Figure_7/EDFig7_skeleton_joints.csv` | ED 7 | The 26 drawn node positions (mm, pelvis at the origin) |
| `Extended_Data_Figure_7/EDFig7_skeleton_bones.csv` | ED 7 | The segments drawn between those nodes |

## Column conventions

`panel` is the panel letter as printed. `label` is the raw endpoint name in
`results/*.csv`; `label_display` / `system_display` are the strings printed on the
figure. Ordering columns (`spoke_index`, `x_position`, `bar_order`, `row_order`,
`list_position`, `legend_order`) give the drawing order, clockwise from 12 o'clock for
the radars and top-to-bottom for the horizontal panels. Blank cells mean the value is
not available in the shipped result tables, never zero.

## Statistics

- **Tests.** `wilcox_pvalue` is the pipeline's **one-sided** Wilcoxon signed-rank test
  over seeds (`alternative='greater'`); `wilcox_pvalue_fdr` is its Benjamini–Hochberg
  adjustment. `score_pvalue` is the mean of the per-seed Pearson P values and is
  **uncorrected**. `score_pvalue_fdr` (Figure 3 / ED Fig 3 only) is the BH adjustment
  applied at plot time, across every endpoint tested in that panel.
- **Thresholds.** Figures 4, 5 and ED Fig 1 keep endpoints at `wilcox_pvalue_fdr < 0.1`
  (Figures 4 and ED 1 additionally require `score_pvalue < 0.05`, `score > 0`,
  `delta > 0`). Figures 3 and ED 3 use BH `q < 0.10` on `score_pvalue`.
- **Seed counts differ by panel**: 15 seeds for the embedding models, 10 for the
  Gait-Features baseline in Fig 2a, 10 for Figures 5 and 6, 5 seeds × 5 folds for
  Fig 2b. `n_seeds` states it per row.
- **Figure 2a** AUCs are computed once on the seed-averaged prediction per
  subject-visit — that pooled value is what the legend prints (`auc_plotted`), and the
  seed distribution behind it is given alongside (`auc_seed_mean`, `auc_seed_sd` and
  the `_per_seed` file).
- **Figure 5 seeds are unfloored.** The `seed_0_*` through `seed_9_*` columns in the
  points file are the raw values; the summary `baseline_auc` is the shipped aggregate.
  `n_in_panel_title` is the sex-specific unique-participant count printed in each panel
  title.
- **Figure 6 has no P value of its own.** The plotted importance is a min–max-normalised
  contrast between masked and unmasked models, and no test is attached to it
  (`wilcox_pvalue` is empty throughout `masking_ablation_pearson.csv`). The exact P of
  every model that goes into it is in `Fig6_masking_ablation_components.csv`, where
  `used_by_figure` marks the row the figure consumes. Two quirks are visible there and
  are faithful to the published panel: `snoring` is scored by AUC rather than Pearson r,
  and age / BMI / VAT appear under two systems with the first taken.
- **Extended Data Fig. 2's error bars are not all the same thing**: the continuous
  curves carry SD over seeds, the intermittent overlay markers carry a min–max whisker.
  Both are given explicitly. The 60 s duration exists in `results/` but is not drawn, so
  it is not exported.
- **Extended Data Fig. 4 has unequal arms**: normalized keypoints n=5 seeds, GaitMAE
  n=15.
- **Extended Data Fig. 5's brackets are two-sided** paired Wilcoxon signed-rank over the
  15 seeds, computed at plot time (nothing in `results/` carries them). With n=15 the
  smallest attainable two-sided P is 2/2¹⁵ = 6.1×10⁻⁵, and three of the four sit exactly
  on it. `printed_label` is the string the figure prints.
- **Extended Data Fig. 6 floors the baseline at 0.5**, both for the summary dot and for
  the seed cloud, so `baseline_auc_plotted` and `baseline_auc_raw` are both given. Its P
  values are a paired t-test across folds (`paired_t_pvalue`) with its BH adjustment;
  `fdr_significant_gain_q_lt_0p1` is the `*` printed after the endpoint name.
- **Extended Data Fig. 7 is a schematic**, not a measurement: one frame of one A-pose
  recording projected to the coronal plane, with no statistics to report. The
  coordinates are provided so the drawn geometry is reproducible.
- **Not included**, by design: the ROC curve points of Fig 2a and the UMAP coordinates
  of Fig 2b (both are in `results/roc_gender.csv` and `results/umap_embeddings.csv`).
