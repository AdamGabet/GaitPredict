"""Stream `.ntds` clips from S3 into per-clip embeddings **without keeping files locally**.

Plain Python — no Jupyter dependency, usable from a script or a notebook. For each
clip it downloads to a temp path, runs the proven `normalize_seq_fast` preprocessing
(the same path validated in the guide notebook, MPJPE ~0.007 — NOT `ntds_dataset.py`'s
center-only path), forward-passes the published DSTformer, pools one embedding per clip,
**deletes the temp file**, and appends to a resumable parquet store.

Typical use:
    from model.inference.ntds_embeddings import load_config, select_subset, load_model, stream_extract
    cfg = load_config()
    subset = select_subset(cfg, n_participants=250)
    model = load_model(cfg)
    emb_df = stream_extract(cfg, subset, model, out_parquet="results/ntds_gait_smoke_emb.parquet")
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message=".*read_feather is deprecated.*")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.architecture.motionBert_full import DSTformer, ReconstructNet
from model.preprocessing.args import PreprocessingArgs
from model.preprocessing.preprocessing import DualCameraDataset
from model.preprocessing.normalizing_time_and_space import normalize_cycle_and_space
from model.preprocessing.joints_file import noise_groups

GAIT_ACTIVITIES = ["self_selected_gait_speed", "tm_3kmh", "stationary_walk"]
# The paper's five standardized motor tasks (all activities except apose + romberg_open).
FIVE_TASKS = ["self_selected_gait_speed", "tm_3kmh", "stationary_walk",
              "romberg_closed", "sit_to_stand"]

_PUB = os.path.join(_ROOT, "model", "published")
_DEFAULT_CKPT = os.path.join(_PUB, "epoch_31.pth")
_DEFAULT_PREP = os.path.join(_PUB, "preprocessing_args.json")
_DEFAULT_MARGS = os.path.join(_PUB, "model_args.txt")


@dataclass
class Config:
    prep_args: PreprocessingArgs
    model_args: dict
    manifest: pd.DataFrame
    ckpt_path: str
    device: torch.device
    bucket: str

    @property
    def size_seq(self) -> int:
        return self.model_args["size_seq"]


def load_config(env_path: Optional[str] = None,
                manifest_path: str = "data/manifests/clips_v1.parquet",
                ckpt_path: str = _DEFAULT_CKPT) -> Config:
    """Load .env creds, the clips manifest, and the published model/preprocessing configs."""
    from dotenv import load_dotenv
    load_dotenv(env_path or os.path.join(_ROOT, ".env"))

    prep_args = PreprocessingArgs(**json.load(open(_DEFAULT_PREP)))
    prep_args.use_memmap = False
    prep_args.load_to_ram = False
    model_args = ast.literal_eval(open(_DEFAULT_MARGS).read())
    manifest = pd.read_parquet(os.path.join(_ROOT, manifest_path)
                               if not os.path.isabs(manifest_path) else manifest_path)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    bucket = os.getenv("S3_BUCKET", "")
    return Config(prep_args, model_args, manifest, ckpt_path, device, bucket)


def select_subset(cfg: Config, n_participants: Optional[int] = None,
                  activities: Sequence[str] = GAIT_ACTIVITIES,
                  front_only: bool = True, labeled_only: bool = True, seed: int = 0) -> pd.DataFrame:
    """Pick clips from the manifest: front-camera + labeled clips, sampled by participant.

    `front_only` keeps `is_front_facing==True` — the paper's/original model's frontal camera
    (in-distribution: reconstructs at MPJPE ~0.008; the other camera is off-distribution ~0.037).
    `n_participants=None` returns every matching clip. BMI is derived from height/weight.
    """
    m = cfg.manifest
    sel = m[m.activity.isin(activities)].copy()
    if front_only:
        sel = sel[sel.is_front_facing]
    if labeled_only:
        sel = sel[sel.age_at_session.notna() & sel.gender.notna()
                  & sel.height_cm.notna() & sel.weight_kg.notna()]
    if n_participants is not None:
        rng = np.random.RandomState(seed)
        parts = sel.participant_id.dropna().unique()
        pick = rng.choice(parts, size=min(n_participants, len(parts)), replace=False)
        sel = sel[sel.participant_id.isin(pick)]
    sel = sel.copy()
    sel["bmi"] = sel.weight_kg / (sel.height_cm / 100.0) ** 2
    sel["clip"] = sel.ntds_uri.apply(lambda u: os.path.basename(u))
    return sel.reset_index(drop=True)


def _detect_arch(sd: dict) -> dict:
    return dict(
        dim_feat=sd["model_backbone.pos_embed"].shape[-1],
        num_joints=sd["model_backbone.pos_embed"].shape[1],
        depth=sum(1 for k in sd if "blocks_st." in k and k.endswith(".norm1_s.weight")),
        dim_rep=sd["model_backbone.pre_logits.fc.weight"].shape[0],
        with_rope="model_backbone.temp_embed" not in sd,
        num_sink=sd["model_backbone.sink_tokens"].shape[1] if "model_backbone.sink_tokens" in sd else 0,
        dim_in=sd["joint_embd.weight"].shape[1],
        dim_out=sd["reconstruct_head.weight"].shape[0],
        koleo_dim=sd["koleo_head.weight"].shape[0] if "koleo_head.weight" in sd else None,
    )


def load_model(cfg: Config) -> ReconstructNet:
    """Rebuild DSTformer+ReconstructNet from the checkpoint (architecture auto-detected)."""
    sd = torch.load(cfg.ckpt_path, map_location="cpu")["model"]
    for pfx in ("_orig_mod.", "module."):
        if any(k.startswith(pfx) for k in sd):
            sd = {k[len(pfx):]: v for k, v in sd.items()}
    a = _detect_arch(sd)
    backbone = DSTformer(num_joints=a["num_joints"], dim_in=a["dim_in"], maxlen=cfg.size_seq,
                         depth=a["depth"], drop_rate=cfg.model_args["dropout_ratio"], use_rope=a["with_rope"],
                         use_flash_attn=False, dim_feat=a["dim_feat"], dim_rep=a["dim_rep"],
                         num_heads=cfg.model_args["num_heads"], num_sink_tokens=a["num_sink"])
    model = ReconstructNet(backbone, dim_in=a["dim_in"], dim_out=a["dim_out"], koleo_dim=a["koleo_dim"])
    model.load_state_dict(sd)
    return model.to(cfg.device).eval()


class _Shim:
    """Minimal stand-in so we can reuse DualCameraDataset's exact transforms without its
    cluster-only file discovery."""
    def __init__(self, args):
        self.args = args
        self.per_joint_amount = 4 if args.use_confidence else 3
        self.centroid = None


def preprocess_ntds(cfg: Config, path: str, max_chunks: Optional[int] = None) -> np.ndarray:
    """`.ntds` -> [n_chunks, size_seq, 26, 4] via the proven full normalization."""
    shim = _Shim(cfg.prep_args)
    df = pd.read_feather(path)
    if "body_id" in df.columns:                                  # keep the primary tracked body
        df = df[df["body_id"] == df["body_id"].value_counts().idxmax()].reset_index(drop=True).drop(columns=["body_id"])
    raw = DualCameraDataset._get_np_from_df(shim, df)             # [T, 32, 4]
    nrm = DualCameraDataset.normalize_seq_fast(shim, raw, front=True)  # [T, 26, 4]
    nc = normalize_cycle_and_space(nrm, activity="x", normalize_space=cfg.prep_args.normalize_space,
                                   normalize_time=cfg.prep_args.normalize_time, num_frames=cfg.prep_args.cycle_len)
    T = cfg.size_seq
    chunks = []
    for i in range(max(1, nc.shape[0] // T)):
        seg = nc[i * T:(i + 1) * T]
        if seg.shape[0] < T:
            seg = np.pad(seg, ((0, T - seg.shape[0]), (0, 0), (0, 0)), mode="edge")
        chunks.append(seg)
        if max_chunks and len(chunks) >= max_chunks:
            break
    return np.stack(chunks)


# Variant-5 pooling (the paper's choice): bilaterally-merged anatomical groups × (mean, 99th
# percentile) over time, on the frozen encoder's block_embeddings -> 1024-d. Same recipe as
# _pool_v5_batch in model/inference/extract_and_eval.py.
_V5_MERGED = {"legs": ["L_leg", "R_leg"], "arms": ["L_arm", "R_arm"],
              "torso": ["torso"], "head": ["head_full"]}


def _pool_v5(block_emb: torch.Tensor) -> torch.Tensor:
    """block_emb [1, T, J, D] -> [1, 8*D]: 4 merged groups × {mean, p99} over the time axis."""
    groups = {g: block_emb[:, :, idx, :].mean(dim=2) for g, idx in noise_groups.items()}  # [1,T,D]
    merged = {m: torch.stack([groups[g] for g in src], 0).mean(0)                          # [1,T,D]
              for m, src in _V5_MERGED.items()}
    parts = []
    for m in ("legs", "arms", "torso", "head"):
        e = merged[m]
        parts.append(e.mean(dim=1))                        # [1, D]  steady-state
        parts.append(torch.quantile(e, 0.99, dim=1))       # [1, D]  transient peaks
    return torch.cat(parts, dim=1)                          # [1, 8D = 1024]


@torch.no_grad()
def embed_chunks(cfg: Config, model: ReconstructNet, chunks: np.ndarray) -> np.ndarray:
    """Forward each 900-frame window, Variant-5-pool block_embeddings, mean over windows -> [1024]."""
    x = torch.tensor(chunks, dtype=torch.float32).to(cfg.device)
    vecs = []
    for i in range(x.shape[0]):
        _, bundle = model(x[i:i + 1])
        be = bundle["block_embeddings"].detach().to("cpu")   # [1,T,J,D]; pool on CPU (torch.quantile)
        vecs.append(_pool_v5(be).numpy())
    return np.concatenate(vecs, 0).mean(0)                    # [1024]


def stream_extract(cfg: Config, subset: pd.DataFrame, model: ReconstructNet,
                   out_parquet: str, max_chunks: Optional[int] = 1,
                   keep_files: bool = False, flush_every: int = 100,
                   download_workers: int = 8, prefetch: int = 24,
                   progress: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
    """Prefetch-download → embed → delete; append to a resumable parquet store.

    `download_workers` S3 downloads run concurrently, overlapping the (single-threaded, GPU-bound)
    embedding so network I/O hides behind compute. At most `prefetch` clips sit on disk at once
    (bounds temp usage). Returns the full embeddings DataFrame (labels + emb_0..emb_{D-1}); clips
    already present in `out_parquet` are skipped, so a re-run resumes / grows the store.
    """
    import boto3
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    s3 = boto3.client("s3")                                       # boto3 clients are thread-safe

    done: set[str] = set()
    existing = None
    if os.path.exists(out_parquet):
        existing = pd.read_parquet(out_parquet)
        done = set(existing.get("clip", pd.Series(dtype=str)).tolist())

    tmpdir = None if keep_files else tempfile.mkdtemp(prefix="ntds_")
    local_dir = os.path.join(_ROOT, "data", "ntds") if keep_files else tmpdir
    os.makedirs(local_dir, exist_ok=True)

    todo = subset[~subset["clip"].isin(done)].reset_index(drop=True)
    n = len(todo)
    sem = threading.Semaphore(prefetch)                          # bound clips-on-disk
    out_q: "queue.Queue" = queue.Queue()
    SENTINEL = object()

    def download_one(row):
        clip = row["clip"]
        local = os.path.join(local_dir, clip)
        err = None
        try:
            if not os.path.exists(local):
                bucket, key = row["ntds_uri"].replace("s3://", "").split("/", 1)
                s3.download_file(bucket, key, local)
        except Exception as exc:                                 # noqa: BLE001
            err, local = exc, None
        out_q.put((row, local, err))

    def feeder():
        with ThreadPoolExecutor(max_workers=download_workers) as ex:
            for _, row in todo.iterrows():
                sem.acquire()                                    # block until a slot frees
                ex.submit(download_one, row)
        out_q.put(SENTINEL)

    threading.Thread(target=feeder, daemon=True).start()

    rows, processed = [], 0
    while True:
        item = out_q.get()
        if item is SENTINEL:
            break
        row, local, err = item
        clip = row["clip"]
        try:
            if err is not None or local is None:
                raise err or RuntimeError("download failed")
            emb = embed_chunks(cfg, model, preprocess_ntds(cfg, local, max_chunks))
            rows.append({
                "clip": clip, "participant_id": row["participant_id"], "test_id": row["test_id"],
                "activity": row["activity"], "age": float(row["age_at_session"]),
                "gender_male": 1.0 if row["gender"] == "male" else 0.0, "bmi": float(row["bmi"]),
                **{f"emb_{j}": float(v) for j, v in enumerate(emb)},
            })
        except Exception as exc:                                 # skip a bad clip, keep going
            if progress:
                progress(processed + 1, n, f"skip {clip}: {exc}")
        finally:
            if not keep_files and local and os.path.exists(local):
                os.remove(local)                                 # <-- never keep the .ntds
            sem.release()                                        # free a prefetch slot
        processed += 1
        if progress and processed % 10 == 0:
            progress(processed, n, clip)
        if flush_every and rows and len(rows) % flush_every == 0:
            _append(out_parquet, existing, rows); existing = pd.read_parquet(out_parquet); rows = []

    if rows:
        _append(out_parquet, existing, rows)
    if tmpdir and os.path.isdir(tmpdir) and not os.listdir(tmpdir):
        os.rmdir(tmpdir)
    return pd.read_parquet(out_parquet)


def _append(out_parquet: str, existing: Optional[pd.DataFrame], rows: list) -> None:
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True) if existing is not None else new
    combined = combined.drop_duplicates(subset="clip", keep="last").reset_index(drop=True)
    combined.to_parquet(out_parquet)
