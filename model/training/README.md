# Training Procedure

How to set up an environment and run MotionBERT (DSTformer + ReconstructNet)
training, on either a Mac (Apple Silicon, MPS) or a Linux EC2 instance (CUDA).

See the top-level [`README.md`](../../README.md) for reproducing the published
figures from pre-computed results — this doc is for training a new model.

---

## 1. Prerequisites

- Python 3.11 (matches the tested environment; 3.9+ per the top-level README's
  general requirement, but training specifically has only been exercised on 3.11)
- A virtual environment (`venv` or `conda`) — don't install into system Python
- A [Weights & Biases](https://wandb.ai) account, for run tracking (or run in
  `debug_mode: True`, which forces `wandb` into offline mode)

---

## 2. Install

Both platforms share the same `requirements.txt`; only the PyTorch install step differs.

### Mac (Apple Silicon, MPS)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The standard PyPI `torch` wheel already includes MPS support on Apple Silicon —
no separate install step. Training auto-detects the device (CUDA → MPS → CPU);
you don't need to set anything to use MPS.

Verify MPS is actually available before a real run:
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

### Linux EC2 (CUDA)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The default `pip install torch` (via `requirements.txt`) pulls a CUDA-enabled
build automatically on Linux — but confirm the CUDA version matches your
instance/driver. If you need a specific CUDA build (check your driver version
with `nvidia-smi` first), install torch explicitly before the rest of
`requirements.txt`, e.g. for CUDA 12.4:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the
current index URL matching your CUDA version.

Verify CUDA is visible:
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Both platforms

`requirements.txt` covers `model/`'s core training/inference dependencies.
Two things it does **not** include, needed only for specific workflows:

- **Data sync from S3** (`migrate_ntds_to_s3.py`, `rescrape_10k_metadata.py`, or
  pulling a local `.ntds` subset for the `ntds` data source below): `boto3`,
  `google-api-python-client`, `google-auth`. Install with:
  ```bash
  pip install boto3 google-api-python-client google-auth
  ```
- **`torch_geometric`**: only needed if you pass `graph_data=True` to
  `get_datasets()`, which no current config does. Skip unless you're
  specifically working on that code path.

---

## 3. Environment variables

Copy `.env.example` (or create `.env` from scratch) at the repo root — all
scripts call `load_dotenv()`. Never commit `.env` (already gitignored).

| Variable | Used by | Notes |
|---|---|---|
| `SKELETON_DATA_DIR` | `data_source="legacy_csv"` (default) | Root dir with `train/`, `test/`, `eval/` subfolders of front/back CSVs. Defaults to the original cluster path — override for any other machine. |
| `NTDS_MANIFEST` | `data_source="ntds"` | Path to the clips manifest, e.g. `data/manifests/clips_v1.parquet`. |
| `NTDS_LOCAL_DIR` | `data_source="ntds"` | Local dir of synced `.ntds` files. Defaults to `data/ntds/`. |
| `MODEL_SAVE_DIR` | all training | Where checkpoints (`epoch_N.pth`) get written. |
| `WANDB_PROJECT`, `WANDB_ENTITY` | all training | wandb run tracking. |
| `TRAIN_CONFIG` | `nonsweep_main.py` | Which named config to use — see §5. |
| `MODEL_FILE` | `TRAIN_CONFIG=finetune` | Checkpoint dir/file to fine-tune from. |
| `SERVICE_ACCOUNT_FILE`, `S3_BUCKET`, `AWS_*` | data sync scripts only | Not needed for training itself. |

---

## 4. Choosing a data source

Two interchangeable options, selected via the `data_source` config key
(`"legacy_csv"` default, or `"ntds"`) — set it directly on whichever config
dict you use, or via a new named config (see §5).

### `legacy_csv` (default) — `DualCameraDataset`

Reads paired `front.csv`/`back.csv` files from `SKELETON_DATA_DIR/{train,test,eval}/`.
This is the original, fully-featured pipeline: masking-curriculum augmentation,
research-stage lookup, the full clinical label set (`hr_bpm`, `eA1C`, `Anxiety`,
`Depression`, etc.). Requires access to that pre-split directory tree — nothing
to set up if you already have it mounted; otherwise this data source isn't usable.

### `ntds` — `NtdsChunkDataset`

Reads `.ntds` (feather) files directly from the S3-migrated corpus
(`migrate_ntds_to_s3.py`, `dataset_version=v1`), via the clips manifest. Lighter
weight, but **not a full replacement** for `legacy_csv` — see the limitations
below before relying on it for a real training run, not just a smoke test.

**Setup** (run once, or re-run any time to fetch more of the corpus — it's
resumable, see below):
```bash
# from repo root, with AWS creds set (see .env, or an IAM instance role on EC2)
aws s3 sync s3://<bucket>/manifests/clips/dataset_version=v1/ data/manifests/
python3 scripts/sync_ntds_corpus.py
```
`scripts/sync_ntds_corpus.py` downloads each manifest row to a **flat**
`NTDS_LOCAL_DIR/<basename>` — deliberately not `aws s3 sync` on the raw
prefix, because the files live under a partitioned S3 layout
(`interim/ntds/dataset_version=v1/...`) that doesn't match the flat directory
`NtdsChunkDataset` expects; a plain `aws s3 sync` would preserve that nesting
and silently yield zero usable files. It's idempotent (skips files already
present at the expected size) and parallelized (`--max-workers`, default 32).
Pass `--limit N` to pull just the first N manifest rows for a local smoke
test instead of the full ~127GB corpus.

`build_ntds_datasets()` (in `model/preprocessing/ntds_dataset.py`) only uses
rows whose file is actually present in `NTDS_LOCAL_DIR` — a partial sync just
yields a smaller dataset, not an error, so it's safe to start small and grow it.

Decoding is lazy — `NtdsChunkDataset` reads each file's frame count up front
(a cheap single-column read) to size itself, then decodes the actual pose data
per `__getitem__` call rather than holding the whole corpus in RAM (measured
~7ms/file, no caching needed — that's well under a forward/backward pass on
this model). This means it's safe to point it at the full corpus regardless of
instance RAM; the earlier eager-decode version wasn't (it held everything
decoded in memory for the life of the process, ~0.6x the raw corpus size).

**Known limitations** (documented in `ntds_dataset.py`'s module docstring too):
- No masking-curriculum augmentation (`group_masking`/`random_mask` configs are
  silently inert on this path — every chunk gets an all-ones mask).
- Only `age` and `gender` labels are populated from real data (from
  `clips_v1.parquet`'s demographics). Any other label name in `config['labels']`
  (`hr_bpm`, `eA1C`, `Anxiety`, `Depression`, `wearable_total_weekly_hours`, ...)
  silently becomes a `0.0` placeholder — harmless for the self-supervised
  training loss (which never reads labels), but **any probe metric evaluated
  against an unavailable label is meaningless, not just noisy**. Don't trust
  those specific numbers in `run_final_evaluation`'s output on this data source.
  (`long_lowmem_ntds`, below, restricts `labels` to just `["age", "gender"]`
  for exactly this reason.)
- Train/test/eval split is by `test_id` (participant) with a fixed seed —
  reasonable default to avoid leakage, but not configurable yet beyond the
  `test_size`/`eval_size`/`seed` args on `build_ntds_datasets()`. ~1.4% of rows
  have no linked participant demographics (age/gender/height/weight all null)
  — they still split safely by `test_id` and fall back to the same `0.0`
  placeholder as any other missing label, so no special handling is needed.
- Every recording in this corpus is Azure Kinect-sourced (`source ==
  "azure_kinect"` for 100% of `clips_v1.parquet`), and the model's joint
  schema (`KEEP_JOINTS` in `ntds_dataset.py`) is hardcoded to Kinect's 32-joint
  layout. There is no OpenPose (or other estimator) code or data anywhere in
  this repo yet, so this isn't a live incompatibility — but it does mean the
  foundational model's entire input contract is Kinect-shaped, worth knowing
  before treating results as representative of a future non-Kinect pipeline.

---

## 5. Choosing a config

Set `TRAIN_CONFIG` (env var) to one of:

| Name | What it is |
|---|---|
| `long` (default) | Full recipe matching the published `epoch_31.pth` — `size_seq=900`, `depth=8`, `batch_size=8`. **Needs ~200+GB peak memory as-is** — not runnable on a single Mac or most EC2 GPUs without `long_lowmem`. |
| `long_lowmem` | Identical recipe to `long` (same architecture/hyperparameters, so results are directly comparable), but `batch_size=1` + `grad_accum_steps=8` (mathematically equivalent effective batch of 8 — verified bit-identical/floating-point-exact) + `use_grad_checkpointing=True` (verified bit-identical gradients). Peak memory ~5GB instead of ~200+GB. **Use this one** unless you have a GPU large enough for `long` directly. Uses `legacy_csv` (needs `SKELETON_DATA_DIR`). |
| `long_lowmem_ntds` | Same as `long_lowmem`, but `data_source="ntds"` (needs the S3 corpus synced — see §4) and `labels=["age", "gender"]` (the only real labels on this data source). **Use this one for training against the S3 corpus** — `long_lowmem` alone won't touch it, since `data_source` isn't inherited from a config name. |
| `short` | Shorter sequences (`size_seq=128`), faster iteration, not directly comparable to `epoch_31.pth`. |
| `default` | The original baseline config. |
| `like_old` | An older recipe kept for reference. |
| `finetune` | Continue training from a checkpoint (`MODEL_FILE`) on new data — inherits `long`'s architecture so it loads the published checkpoint cleanly. |

```bash
TRAIN_CONFIG=long_lowmem python -m model.training.nonsweep_main
```

`use_grad_checkpointing` and `grad_accum_steps` aren't config-specific — set
them on any config dict if you want the same memory trick elsewhere (e.g. a
memory-constrained `finetune` run).

---

## 6. Running

```bash
TRAIN_CONFIG=long_lowmem python -m model.training.nonsweep_main
```

Checkpoints land in `$MODEL_SAVE_DIR/bert_<wandb_run_id>_<date>/epoch_N.pth`.
Set `debug_mode: True` on a config for a fast one-epoch/few-batch smoke test
before committing to a full run (forces wandb offline too).

To benchmark a newly trained checkpoint against `epoch_31.pth`, run both
through `run_final_evaluation()` (`model/training/eval_loop.py`) on the
identical held-out split — see the reconstruction-loss, ridge-probe, and UMAP
metrics it already produces. Confirm the eval split doesn't overlap with
whatever `epoch_31.pth` was originally trained on before treating the
comparison as fair.

---

## 7. Full pipeline on a fresh EC2 instance

`scripts/ec2_train_and_upload.sh` wraps install → S3 corpus sync → train →
checkpoint upload into one script for a fresh instance:

```bash
export S3_BUCKET=<bucket>          # required
export TRAIN_CONFIG=long_lowmem_ntds  # default; override for a different config
scripts/ec2_train_and_upload.sh
```

- Prefer an **IAM instance role** on the EC2 instance over `AWS_*` keys in
  `.env` — boto3/aws-cli pick it up automatically, and it avoids putting
  long-lived credentials on disk.
- Checkpoints stream to `s3://$S3_BUCKET/checkpoints/<run_tag>/` every 5
  minutes (`SYNC_INTERVAL_SECONDS`) during training, plus a final sync on
  exit — so a spot-instance interruption doesn't lose the run. Training itself
  never talks to S3; the sync is a background loop in the wrapper script.
- Corpus sync is resumable (`scripts/sync_ntds_corpus.py`) — re-running the
  script after an interruption just picks up where it left off instead of
  re-downloading everything.
- For a smoke test before committing to the full ~127GB corpus, sync a
  subset first (`python3 scripts/sync_ntds_corpus.py --limit 2000`) and set
  `debug_mode: True` on the config for a fast partial-epoch run.
