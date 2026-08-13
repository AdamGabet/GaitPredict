6# CLAUDE.md — Collaboration Guide

> This file defines how Claude should operate as an embedded engineering and data science
> collaborator for BioPilot.ai. It is read at the start of every Claude Code session.

---

## Who We Are

BioPilot.ai is a small professional services company delivering AI/ML-powered analysis
of human movement data for clinical pharma trials. Our core work is the computational
backbone of clinical movement analysis pipelines — feature extraction, model training,
trial analytics, and the productionalization of research code into stable, client-deliverable
software.

Our clients are neurological drug startups running Phase 1–3 trials. The code we write
ends up in clinical contexts; quality and reproducibility are not optional.

**Current active workstreams:**
- Multi-camera recording software (Spaulding / TAU sites)
- Solid pharma trial analysis (stairs, lie-to-stand, 10-m walk)
- Nervegen pipeline productionalization & QA
- Foundational motion model migration & in-house training (JEPA-style architecture)

---

## How We Work

### Engagement Model

Work is scoped in **Parts** (~1 week / ~$5,000 each) under numbered POs (005–008 active).
Each Part has a defined deliverable: source code + documentation, or source code + summary
of findings. Claude should be aware of which Part a task belongs to so deliverables are
scoped correctly.

### Tone & Communication Style

- Be direct and concise. Skip filler phrases.
- Think out loud on hard problems — show reasoning before the solution.
- Flag trade-offs proactively, especially around clinical validity, reproducibility,
  and architectural debt that will surface at productionalization time.
- When uncertain about domain specifics (clinical protocols, test definitions,
  trial structure), say so and ask rather than assume.

### Collaboration Defaults

- Don't silently change file structure, rename outputs, or modify configs without noting it.
- Destructive operations (overwrite a results file, retrain a checkpoint) require
  explicit confirmation before proceeding.
- Prefer reversible over irreversible, especially in client-facing environments.
- When a research-phase shortcut would create productionalization debt, name it.

---

## Domain Knowledge

### Clinical Movement Tests

The core movement tests we analyze. Claude should understand what each measures:

| Test | What it measures | Key considerations |
|------|------------------|--------------------|
| **10-meter walk** | Gait speed, stride characteristics | Primary endpoint for many neurological trials; batch effect is a known issue |
| **Stairs (ascent/descent)** | Lower-limb strength, coordination, safety | Solid trial primary |
| **Lie-to-stand** | Functional transfer, postural control | Solid trial; high inter-subject variability |
| **Romberg** | Static balance (eyes open/closed) | Highly sensitive to room/recording environment (batch effect) |
| **Sit-to-stand** | Transfer function, leg power | Batch effect visible in UMAP embeddings |
| **Treadmill** | Continuous gait under controlled speed | Less visually affected by batch effect but still leaks |

### Batch Effect — Current Priority Technical Problem

The dominant technical headache. On ~100 patients recorded in a different room,
classifiers can identify the recording room better than almost any clinical signal.
Romberg and sit-to-stand are most affected (visible in UMAP); treadmill leaks too.

**Mitigation approaches under consideration:**
1. Suppress during embedding training (architecture-level)
2. Regress out in the ML head
3. Fix in preprocessing (normalization, domain adaptation)

When working on any model or embedding task, flag whether the approach is
vulnerable to batch effect. Never treat cross-site or cross-session results
as directly comparable without first checking for confound.

### Recording Infrastructure

| Site | Camera Setup | Notes |
|------|-------------|-------|
| TAU | Single 3D-USB camera | Single-camera site |
| Spaulding | Multiple cameras | Multi-camera, needs synchronization |

**Azure (Kinect) is deprecated** — cameras discontinued. Do not write new code
targeting the Azure pose pipeline. Migration target: open-source pose estimator
(OpenPose or equivalent).

### Motion Model

The foundational motion model is a co-authored, JEPA-style architecture nearing
publication. Migration into Newton's in-house environment is PO 008.

- Healthy subjects are already embedded in the model and serve as the shared baseline —
  pharma partners do not need their own healthy controls.
- IP is governed by §5.6 of the master agreement (jointly-owned "Training Software").
  Flag any code that is part of the model/training pipeline for IP review before sharing.
- Pharma-scope features applied to existing healthy videos are the "cheap first run"
  alternative if full model migration is not yet funded.

---

## Engineering Stack

### Languages

- **Python 3.11+** — everything: video processing, feature extraction, ML, pipelines
- **SQL** — for any structured data storage or trial metadata queries
- **Bash** — automation, scripting, environment setup

### Core Libraries

| Domain | Tools |
|--------|-------|
| Video processing | OpenCV, ffmpeg (via subprocess or ffmpeg-python) |
| Pose estimation | OpenPose (migration target); avoid Azure/Kinect SDK for new code |
| Numerical / signal | NumPy, SciPy |
| Data wrangling | pandas, polars |
| ML / modeling | scikit-learn, PyTorch, XGBoost / LightGBM |
| Dimensionality reduction / viz | UMAP, matplotlib, seaborn |
| Experiment tracking | MLflow |
| API / serving | FastAPI, Pydantic |
| Packaging / config | pyproject.toml, pydantic-settings |

### Repository Layout

```
project/
├── data/
│   ├── raw/            # raw video + sensor data (gitignored; never commit client data)
│   ├── interim/        # intermediate artifacts (pose keypoints, features)
│   └── processed/      # analysis-ready datasets
├── notebooks/          # EDA and exploration only — not production code
├── src/
│   ├── recording/      # camera capture, multi-camera sync (PO 005)
│   ├── pose/           # pose estimation pipeline (OpenPose migration)
│   ├── features/       # feature extraction per test type
│   ├── models/         # training, evaluation, checkpoints
│   ├── analysis/       # trial-level KPIs, longitudinal, responder logic
│   └── api/            # service layer if applicable
├── tests/
├── scripts/            # ops and one-off scripts
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Data Science Workflow

### Research Phase vs. Production Phase

This matters enormously for our work — we often build something in research mode
(notebooks, quick scripts) and then productionalize it for a client (PO 007 is
exactly this pattern).

- **Research phase:** notebooks in `notebooks/`, quick iteration, speed over structure.
  Name notebooks with a prefix: `01_solid_feature_exploration.ipynb`.
- **Production phase:** refactor into `src/` with type hints, docstrings, tests.
  No notebook code called from production. No hardcoded paths or patient IDs.

When starting a task, establish which phase it belongs to. Don't over-engineer
research code, and don't under-engineer production code.

### Feature Engineering Standards

- Features must be **reproducible**: same input → same output, always.
- Separate feature computation from model training. Features should be cacheable.
- **Document units**: meters/second vs. pixels/frame is not obvious. Always state units
  in variable names or docstrings.
- **Watch for target leakage** — especially when features are derived from clinical
  assessments (e.g., don't use rater scores to compute features used to predict rater scores).
- **Flag batch-effect-sensitive features** explicitly with a comment.

### Model Development Standards

- Establish a baseline (e.g., handcrafted features + logistic regression) before
  moving to complex architectures.
- Document the problem framing: what is being predicted, the unit of analysis (patient?
  visit? test?), the evaluation metric, and why that metric is appropriate for a pharma trial.
- Track all experiments in MLflow. Never lose a run.
- Report on a held-out test set. For longitudinal trial data, ensure the test split
  respects time — no future leakage.
- Include confidence intervals on all reported metrics.
- Document failure modes before handing off results to Newton.

### Clinical Validity Checks

Before finalizing any analysis deliverable:
- Confirm the cohort definition matches what Newton specified (which subjects, which
  time points, which tests).
- Verify that time points are labeled correctly (baseline, 3-month, 6-month, etc.).
- Confirm responder definitions match the trial protocol, not our own assumptions.
- Flag any data quality issues (missing visits, corrupted video, outlier subjects) explicitly.

---

## Deliverable Standards

Each PO Part has a defined deliverable. Default structure:

**Code deliverables:**
- Source code in `src/` with docstrings and type hints
- Tests in `tests/` — at minimum a smoke test and one behavioral test per module
- A short `README` or docstring in the relevant module explaining what it does,
  inputs/outputs, and any known limitations

**Summary of findings deliverables:**
- Plain language summary (Newton stakeholders are not always technical)
- Key figures (UMAP, trajectory plots, KPI tables) with axis labels and units
- Limitations and open questions section — never omit this

**Documentation deliverables:**
- Architecture overview: what the system does, how data flows through it,
  what each component is responsible for
- Operational notes: how to run it, environment requirements, known issues

---

## Security & Data Handling

- **Never commit patient data, video, or clinical records to version control.**
- Use anonymized or synthetic data in tests and examples.
- Patient IDs in code should be treated as PII: `# PII: use anonymized IDs in examples`.
- Secrets (API keys, credentials) live in environment variables or a secrets manager.
  Use `.env` locally with `python-dotenv`; `.env` in `.gitignore`.
- IP-sensitive code (motion model, training pipeline) — flag with `# IP: jointly-owned
  per §5.6` before sharing externally or committing to shared repos.

---

## When Helping With a Task

1. **Establish context** — which PO, which Part, research or production phase?
2. **Check for batch effect exposure** — does this task touch data across sites or sessions?
3. **Propose the approach** — especially for non-trivial tasks; get alignment before code.
4. **Implement** — with type hints, docstrings, units documented, and tests.
5. **Summarize what changed** — flag any architectural decisions, limitations, or
   open questions that Newton or the team should know about.

If a task is underspecified, ask one clarifying question rather than assuming.

---

## Active Engagement Notes

**Client:** Newton-Tech
**POs active:** 005 (recording), 006 (Solid), 007 (Nervegen), 008 (model migration)
**Pharma programs:** Solid (4 subjects × 4 time points × 3 tests), Nervegen (productionalization)
**Pending:** ~300 videos from a third company (on-funding)
**IP note:** PO 008 model/training code is jointly-owned — flag for review before external sharing.
**Key contacts:** Yuval Brodsky, CEO of Newton Tech
