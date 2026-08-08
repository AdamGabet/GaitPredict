# GaitPredict

<p align="center">
  <img src="assets/skeleton_demo.gif" width="600"/>
</p>

Self-supervised gait representation learning from RGB-D skeleton sequences.
This repository accompanies the paper **"Self-supervised deep learning reveals Gait as a biomarker in the Human Phenotype Project"** and contains the model code, pre-computed downstream results, and all plotting scripts needed to reproduce the figures.

---

## Repository structure

```
GaitPredict-Paper/
├── model/
│   ├── architecture/          # DSTformer transformer backbone + ReconstructNet
│   ├── preprocessing/         # Dataset, augmentation, joint definitions
│   ├── training/              # Training loop, eval loop, configs, loss
│   └── inference/             # Embedding extraction scripts
├── plotting/
│   ├── publication_colors.py  # Shared color palettes and system renaming
│   ├── figure_1/              # Training loss curve + biomarker system infographic
│   ├── figure_2/              # UMAP activity embeddings + ROC curves + ASBV plots
│   ├── figure_3/              # Radar plot of Pearson r across body systems
│   ├── figure_4/              # Gait vs confounders boxplots (male & female)
│   ├── figure_4b/             # Selected biomarker bar plots with error bars
│   ├── figure_5_medical/      # Medical condition dumbbell plots
│   ├── figure_6/              # Joint masking ablation heatmap + body-part plot
│   └── extended_figure_1/     # Gait feature baseline boxplots
├── results/                   # Pre-computed CSV results (see below)
├── sample_data/               # One sample skeleton CSV for smoke testing
├── requirements.txt
└── LICENSE
```

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ recommended.

To train a new model rather than just reproduce the paper's figures, see
[`model/training/README.md`](model/training/README.md) — covers install on
both Mac (MPS) and Linux EC2 (CUDA), environment variables, and choosing a
data source/config.

---

## Reproducing figures

Each plotting script is self-contained and reads from `results/`. Run from the repo root:

```bash
python plotting/figure_1/loss_plot.py
python plotting/figure_1/gait_biomarker_infographic.py
python plotting/figure_2/umap_activity_embeddings.py
python plotting/figure_2/plot_gender_roc.py
python plotting/figure_2/plot_asbv.py
python plotting/figure_3/radar_gait_only.py
python plotting/figure_4/publication_figure_gm.py
python plotting/figure_4b/combined_pearson_barplot.py
python plotting/figure_5_medical/medical_conditions_dumbbell.py
python plotting/figure_6/plot_body_system_heatmap.py
python plotting/figure_6/plot_grouped_by_body_part.py --metric pearson
python plotting/extended_figure_1/publication_figure_gm.py
```

Outputs (PNG + PDF) are written to each figure's `output/` subdirectory.

---

## Results CSVs

| File | Description |
|------|-------------|
| `gait_only_pearson.csv` | Pearson r results: gait model vs no-confounder baseline, all body systems |
| `gait_vs_confounders_pearson.csv` | Pearson r results: gait model vs age/BMI/VAT confounders baseline |
| `medical_conditions_pearson.csv` | AUC results for medical condition classification |
| `masking_ablation_pearson.csv` | Joint group masking ablation Pearson r results (figure 6) |
| `asbv_metrics.csv` | Mean/std Pearson r for Age, BMI, VAT across activities |
| `movement_features_asbv.csv` | Movement feature baseline results for Age, BMI, VAT |
| `roc_gender.csv` | Pre-computed ROC curve points for gender classification by activity |
| `umap_embeddings.csv` | Pre-computed 2D UMAP coordinates of gait embeddings |
| `figure4b_seed_std.csv` | Seed-level std for selected biomarkers (figure 4b error bars) |
| `wandb_training_loss.csv` | W&B training loss export for figure 1 |

---

## Model

The model is a **DSTformer** (Dual-Stream Transformer) trained with a masked reconstruction objective on 3D skeleton sequences. Input: `(batch, time, joints, 3)` XYZ coordinates. Output: reconstructed joint positions used for self-supervised pre-training.

### Training (requires your own data)

```bash
python -m model.training.nonsweep_main
```

Configuration is in `model/training/default_config.py` (`LONG_CONFIG` is the main config used in the paper).

### Extracting embeddings

```bash
python -m model.inference.extract_and_eval
```

### Smoke test

```bash
python -m model.training.nonsweep_main --smoke_test
```

Uses the sample skeleton in `sample_data/`.

---

## License

MIT — see [LICENSE](LICENSE).
