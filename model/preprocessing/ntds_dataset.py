"""Lightweight `.ntds`-native Dataset for training/evaluating directly against
the S3 pose corpus (dataset_version=v1, see migrate_ntds_to_s3.py).

Scope note: this is NOT a full replacement for DualCameraDataset. It reuses the
same joint-selection/centering preprocessing already proven in
model/inference/extract_and_eval.py, but does not implement DualCameraDataset's
masking-curriculum augmentation, research-stage lookup, or full clinical label
set (those labels — hr_bpm, eA1C, Anxiety, Depression, etc. — live in a richer
clinical dataset that was never part of the S3 ntds migration; only
age/gender/height/weight/dominant_hand are available per clips_v1.parquet; any
other requested label name silently becomes a 0.0 placeholder — harmless for
the self-supervised training loss, which never reads labels, but any probe
metric evaluated against an unavailable label is meaningless, not just noisy).

Selected via the `data_source` config key in get_datasets() ("legacy_csv"
[default, DualCameraDataset] vs "ntds" [this module]) — see README.md.
"""
from __future__ import annotations

import enum
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from model.preprocessing.joints_file import noise_joints

CENTROID_JOINT = 0  # PELVIS


class k4abt_joints(enum.IntEnum):
    PELVIS = 0; SPINE_NAVEL = 1; SPINE_CHEST = 2; NECK = 3
    CLAVICLE_LEFT = 4; SHOULDER_LEFT = 5; ELBOW_LEFT = 6; WRIST_LEFT = 7
    HAND_LEFT = 8; HANDTIP_LEFT = 9; THUMB_LEFT = 10
    CLAVICLE_RIGHT = 11; SHOULDER_RIGHT = 12; ELBOW_RIGHT = 13; WRIST_RIGHT = 14
    HAND_RIGHT = 15; HANDTIP_RIGHT = 16; THUMB_RIGHT = 17
    HIP_LEFT = 18; KNEE_LEFT = 19; ANKLE_LEFT = 20; FOOT_LEFT = 21
    HIP_RIGHT = 22; KNEE_RIGHT = 23; ANKLE_RIGHT = 24; FOOT_RIGHT = 25
    HEAD = 26; NOSE = 27; EYE_LEFT = 28; EAR_LEFT = 29; EYE_RIGHT = 30; EAR_RIGHT = 31


KEEP_JOINTS = [j for j in range(32) if j not in noise_joints]
NUM_JOINTS = len(KEEP_JOINTS)  # 26


def extract_pose_array(df: pd.DataFrame) -> np.ndarray:
    """[T, 32, 4] array of (x, y, z, confidence) per k4abt joint."""
    T = len(df)
    arr = np.zeros((T, 32, 4), dtype=np.float32)
    for joint in k4abt_joints:
        idx, name = joint.value, joint.name
        arr[:, idx, 0] = df[f"{idx}_{name}_x"].values
        arr[:, idx, 1] = df[f"{idx}_{name}_y"].values
        arr[:, idx, 2] = df[f"{idx}_{name}_z"].values
        arr[:, idx, 3] = df[f"{idx}_{name}_c"].values
    return arr


def chunk_sequence(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Split [T, J, C] into non-overlapping [seq_len]-frame chunks, zero-padding the tail."""
    T, J, C = arr.shape
    n_full = T // seq_len
    remainder = T % seq_len
    chunks = [arr[i * seq_len:(i + 1) * seq_len] for i in range(n_full)]
    if remainder > 0:
        pad = np.zeros((seq_len - remainder, J, C), dtype=arr.dtype)
        chunks.append(np.concatenate([arr[n_full * seq_len:], pad], axis=0))
    return np.stack(chunks, axis=0)


class NtdsChunkDataset(Dataset):
    """One item = one [size_seq, NUM_JOINTS, 4] chunk from one .ntds file.

    `rows` is a list of dicts with at least: local_path (downloaded .ntds file),
    test_id, activity, age_at_session (optional), gender (optional) -- i.e. rows
    pulled from clips_v1.parquet plus a resolved local file path per clip.
    """

    def __init__(self, rows: Sequence[dict], size_seq: int = 900, label_names: Sequence[str] = ("age",)):
        self.size_seq = size_seq
        self.label_names = list(label_names)
        self._chunks: list[dict] = []
        for row in rows:
            df = pd.read_feather(row["local_path"])
            pose_all = extract_pose_array(df)
            pose_kept = pose_all[:, KEEP_JOINTS, :]
            pelvis_idx = KEEP_JOINTS.index(CENTROID_JOINT)
            centroid = pose_kept[:, pelvis_idx:pelvis_idx + 1, :3]
            pose_kept[:, :, :3] -= centroid

            for chunk in chunk_sequence(pose_kept, size_seq):
                self._chunks.append({
                    "pose": chunk,  # [size_seq, NUM_JOINTS, 4]
                    "activity": row.get("activity", "unknown"),
                    "id": row.get("test_id", "unknown"),
                    "labels": [self._label_value(row, name) for name in self.label_names],
                })

    @staticmethod
    def _label_value(row: dict, name: str) -> float:
        if name == "age":
            v = row.get("age_at_session")
        elif name == "gender":
            v = row.get("gender")
            v = 1.0 if v == "male" else 0.0 if v == "female" else 0.0
        else:
            v = row.get(name)
        try:
            return float(v) if v is not None and not pd.isna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict:
        c = self._chunks[idx]
        pose = torch.tensor(c["pose"], dtype=torch.float32)  # [F, J, 4]
        return {
            "data": pose,
            "original": pose,
            "mask": torch.ones_like(pose, dtype=torch.bool),
            "label_list": c["labels"],
            "activity": c["activity"],
            "id": c["id"],
        }


def build_ntds_datasets(
    size_seq: int = 900,
    labels: Optional[Sequence[str]] = None,
    manifest_path: Optional[str] = None,
    local_dir: Optional[str] = None,
    test_size: float = 0.15,
    eval_size: float = 0.15,
    seed: int = 42,
) -> tuple["NtdsChunkDataset", "NtdsChunkDataset", "NtdsChunkDataset"]:
    """Build (train, test, eval) NtdsChunkDatasets from the S3 clips manifest.

    Requires the .ntds files themselves already synced to `local_dir` (this
    does NOT download from S3 on demand -- see README.md's sync step). Rows
    whose file isn't present locally are silently skipped, so a partial local
    sync just yields a smaller dataset rather than an error.

    Split is by test_id (participant), not by row/clip -- a participant's
    clips never span train/test/eval, matching how DualCameraDataset's
    pre-split directories are expected to avoid the same leakage.
    """
    manifest_path = manifest_path or os.getenv("NTDS_MANIFEST", "data/manifests/clips_v1.parquet")
    local_dir = local_dir or os.getenv("NTDS_LOCAL_DIR", "data/ntds")
    labels = list(labels) if labels else ["age"]

    manifest = pd.read_parquet(manifest_path)
    manifest = manifest.copy()
    manifest["local_path"] = manifest["ntds_uri"].apply(
        lambda uri: os.path.join(local_dir, os.path.basename(uri))
    )
    available = manifest[manifest["local_path"].apply(os.path.exists)]
    if available.empty:
        raise FileNotFoundError(
            f"No .ntds files found in {local_dir!r} matching {manifest_path!r}. "
            f"Sync a subset from S3 first (see README.md 'Using the ntds data source')."
        )

    # .tolist() avoids an Arrow-backed array type sklearn's indexing doesn't handle
    test_ids = available["test_id"].unique().tolist()
    train_ids, holdout_ids = train_test_split(test_ids, test_size=test_size + eval_size, random_state=seed)
    relative_eval_size = eval_size / (test_size + eval_size)
    test_ids_split, eval_ids_split = train_test_split(holdout_ids, test_size=relative_eval_size, random_state=seed)

    def _rows_for(ids) -> list[dict]:
        subset = available[available["test_id"].isin(ids)]
        return subset.to_dict(orient="records")

    train_dataset = NtdsChunkDataset(_rows_for(train_ids), size_seq=size_seq, label_names=labels)
    test_dataset = NtdsChunkDataset(_rows_for(test_ids_split), size_seq=size_seq, label_names=labels)
    eval_dataset = NtdsChunkDataset(_rows_for(eval_ids_split), size_seq=size_seq, label_names=labels)

    # per_joint_amount mirrors DualCameraDataset's attribute of the same name,
    # since train_loop.py reads it directly off whatever dataset get_datasets() returns.
    for ds in (train_dataset, test_dataset, eval_dataset):
        ds.per_joint_amount = 4

    return train_dataset, test_dataset, eval_dataset
