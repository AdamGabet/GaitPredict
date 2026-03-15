"""
Embedding extraction and evaluation script.
Loads a trained model from a given epoch and extracts embeddings with 4 pooling variants.
Pools per-batch to avoid OOM on large datasets.
"""
import copy
import os
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
import wandb

from model.architecture.motionBert_full import DSTformer, ReconstructNet
from model.preprocessing import args
from model.preprocessing.preprocessing import get_datasets
from model.preprocessing.joints_file import noise_groups
from model.training.utils.training_helper import get_latest_dir, find_model_dir_for_run

load_dotenv(interpolate=True)

# Joint groups for variant 3 pooling
JOINT_GROUPS = noise_groups  # {'L_leg': [12,13,14,15], 'R_leg': [...], ...}

# Mapping from noise joint index (0-25) to actual joint name
NOISE_JOINT_NAMES = {
    0: "pelvis",
    1: "spine_navel",
    2: "spine_chest",
    3: "neck",
    4: "clavicle_L",
    5: "shoulder_L",
    6: "elbow_L",
    7: "wrist_L",
    8: "clavicle_R",
    9: "shoulder_R",
    10: "elbow_R",
    11: "wrist_R",
    12: "hip_L",
    13: "knee_L",
    14: "ankle_L",
    15: "foot_L",
    16: "hip_R",
    17: "knee_R",
    18: "ankle_R",
    19: "foot_R",
    20: "head",
    21: "nose",
    22: "eye_L",
    23: "ear_L",
    24: "eye_R",
    25: "ear_R",
}

VARIANT_NAMES = {
    'v1': 'variant1_mean_max',
    'v2': 'variant2_joint_stats',
    'v3': 'variant3_joint_groups',
    'v4': 'variant4_prelogits_joints',
    'v5': 'variant5_merged_groups_percentile',
    'v6': 'variant6_sink_cls',
}

# Merged joint groups: combine L/R arms and L/R legs
MERGED_GROUPS = {
    'legs': ['L_leg', 'R_leg'],
    'arms': ['L_arm', 'R_arm'],
    'torso': ['torso'],
    'head': ['head_full'],
}


def load_run_config_and_args(run_id: str) -> Tuple[dict, args.PreprocessingArgs, str]:
    """Fetch wandb config, preprocessing args, and model directory for a run."""
    if not run_id:
        raise ValueError("run_id must be provided")

    api = wandb.Api()
    project = os.getenv("WANDB_PROJECT", "GaitPredict")
    entity = os.getenv("WANDB_ENTITY", "adam-gabet-weizmann-institute-of-science")
    run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = api.run(run_path)
    config = dict(run.config)

    root_dir = os.getenv("MODEL_SAVE_DIR")
    if not root_dir:
        raise ValueError("MODEL_SAVE_DIR is not set")

    model_dir = find_model_dir_for_run(run_id, root_dir)
    if model_dir is None:
        raise FileNotFoundError(f"Could not locate model directory for run_id={run_id}")

    try:
        loaded_args = args.load_args_from_path(model_dir)
        args_cfg = copy.deepcopy(args.apply_args(loaded_args))
        print(f"Args config: {args_cfg}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Unable to load preprocessing args from {model_dir}") from exc

    return config, args_cfg, model_dir


def _sanitize_for_inference(args_cfg: args.PreprocessingArgs) -> args.PreprocessingArgs:
    """Disable training-only augmentations and masks before evaluation."""
    args_cfg = copy.deepcopy(args_cfg)
    args_cfg.random_mask = 0.0
    args_cfg.random_mask_frames = 0.0
    args_cfg.augment_eval = False
    args_cfg.group_masking = 0.0
    args_cfg.seq_per_person = -1
    args_cfg.limit_users = -1
    if args_cfg.centroid_type == 'second':
        args_cfg.centroid_type = 'second'
    else:
        args_cfg.centroid_type = 'first'
    args_cfg.augments = dict(getattr(args_cfg, "augments", {}))
    args_cfg.equal_seq_per_activity = False
    for key in list(args_cfg.augments.keys()):
        args_cfg.augments[key] = False
    args_cfg.augments['None'] = True
    return args_cfg


def _resolve_checkpoint_path(model_dir: str, epoch: Optional[int]) -> str:
    """Pick the checkpoint file either by epoch or by latest available."""
    if epoch is not None:
        candidate = os.path.join(model_dir, f"epoch_{epoch}.pth")
        if not os.path.exists(candidate):
            print(f"Requested checkpoint {candidate} does not exist, using latest")
            return get_latest_dir(model_dir)
        return candidate
    return get_latest_dir(model_dir)


def _build_model_from_config(config: dict, args_cfg: args.PreprocessingArgs, device: torch.device) -> ReconstructNet:
    """Instantiate the reconstruction model exactly like the training loop."""
    num_joints = 26
    if args_cfg.remove_noise_joints:
        num_joints = 26
    if args_cfg.motionbert_format:
        num_joints = 17

    dim_in = config.get('dim_in')
    if dim_in is None:
        raise ValueError("Config must include dim_in to build the model")

    model_backbone = DSTformer(
        num_joints=num_joints,
        dim_in=dim_in,
        maxlen=config['size_seq'],
        depth=config.get('depth', 5),
        drop_rate=config['dropout_ratio'],
        use_rope=config['with_rope'],
        use_flash_attn=config.get('use_flash_attn', False),
        dim_feat=config.get('dim_feat', 256),
        dim_rep=config['dim_representation'],
        num_heads=config['num_heads'],
        num_sink_tokens=config.get('num_sink_tokens', 0),
    )

    dim_out = 7 if dim_in > 4 else 3
    model = ReconstructNet(
        model_backbone=model_backbone,
        dim_in=dim_in,
        dim_out=dim_out,
        modality_dropout_prob=0.0,
    )
    return model.to(device)


def _load_checkpoint_for_inference(model: ReconstructNet, checkpoint_path: str, device: torch.device) -> ReconstructNet:
    """Load checkpoint weights, stripping DataParallel/torch.compile prefixes."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model']

    # Strip torch.compile wrapper prefix
    if any(k.startswith('_orig_mod.') for k in model_state):
        model_state = {k[len('_orig_mod.'):]: v for k, v in model_state.items()}

    # Strip DataParallel prefix
    if any(k.startswith('module.') for k in model_state):
        model_state = {k[len('module.'):]: v for k, v in model_state.items()}
    
    # ---- Drop unwanted heads/layers here ----
    EXCLUDE_PREFIXES = ('koleo_head',)  # add more prefixes if needed

    model_state = {
        k: v for k, v in model_state.items()
        if not k.startswith(EXCLUDE_PREFIXES)
    }

    model.load_state_dict(model_state)
    model.eval()
    return model


def load_model_and_loaders(
    config: dict,
    args_cfg: args.PreprocessingArgs,
    model_dir: str,
    epoch: Optional[int],
    device: Optional[torch.device] = None,
) -> Tuple[ReconstructNet, Dict[str, torch.utils.data.DataLoader], args.PreprocessingArgs]:
    """Load trained weights and create loaders for train/eval/test splits."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clean_args = _sanitize_for_inference(args_cfg)

    print(f"Clean args: {clean_args}")

    train_dataset, test_dataset, eval_dataset = get_datasets(
        size_seq=config['size_seq'],
        labels=config['labels'],
        graph_data=False,
        overlap_sequence=0,
        args_cfg=clean_args,
    )

    dim_in = train_dataset.per_joint_amount
    config['dim_in'] = dim_in

    batch_size = config.get('eval_batch_size') or config.get('batch_size', 8)
    common_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 6,
        "pin_memory": True,
        "drop_last": False,
    }

    # Order: eval first so we can verify it works quickly
    loaders = {
        "eval": torch.utils.data.DataLoader(eval_dataset, **common_loader_kwargs),
        "test": torch.utils.data.DataLoader(test_dataset, **common_loader_kwargs),
        "train": torch.utils.data.DataLoader(train_dataset, **common_loader_kwargs),
    }

    model = _build_model_from_config(config, clean_args, device)
    checkpoint_path = _resolve_checkpoint_path(model_dir, epoch)
    model = _load_checkpoint_for_inference(model, checkpoint_path, device)

    return model, loaders, clean_args


# =============================================================================
# Per-Batch Pooling Functions (operate on tensors, return numpy)
# =============================================================================

def _pool_v1_batch(block_emb: torch.Tensor) -> np.ndarray:
    """Variant 1: Mean+Max across time and joints. [B,T,J,D] -> [B, 2D]"""
    B, T, J, D = block_emb.shape
    flat = block_emb.view(B, T * J, D)
    mean_pool = flat.mean(dim=1)
    max_pool = flat.max(dim=1).values
    return torch.cat([mean_pool, max_pool], dim=1).cpu().numpy()


def _pool_v2_batch(block_emb: torch.Tensor) -> np.ndarray:
    """Variant 2: Mean/Max/Std across joints, then mean over time. [B,T,J,D] -> [B, 3D]"""
    mean_joint = block_emb.mean(dim=2)
    max_joint = block_emb.max(dim=2).values
    std_joint = block_emb.std(dim=2)
    mean_time = mean_joint.mean(dim=1)
    max_time = max_joint.mean(dim=1)
    std_time = std_joint.mean(dim=1)
    return torch.cat([mean_time, max_time, std_time], dim=1).cpu().numpy()


def _pool_v3_batch(block_emb: torch.Tensor, joint_groups: Dict[str, List[int]]) -> np.ndarray:
    """Variant 3: Mean per joint group, mean over time. [B,T,J,D] -> [B, num_groups*D]"""
    parts = []
    for group_name, indices in joint_groups.items():
        group_emb = block_emb[:, :, indices, :]
        group_pooled = group_emb.mean(dim=2).mean(dim=1)
        parts.append(group_pooled)
    return torch.cat(parts, dim=1).cpu().numpy()


def _pool_v4_batch(prelogits_emb: torch.Tensor) -> np.ndarray:
    """Variant 4: Mean over time per joint, concat all joints. [B,T,J,D] -> [B, J*D]"""
    B, T, J, D = prelogits_emb.shape
    time_pooled = prelogits_emb.mean(dim=1)
    return time_pooled.reshape(B, J * D).cpu().numpy()


def _pool_v5_batch(block_emb: torch.Tensor, joint_groups: Dict[str, List[int]]) -> np.ndarray:
    """
    Variant 5: Mean over joints per group, merge L/R limbs, then mean + 90th percentile over time.
    [B,T,J,D] -> [B, 4 merged groups * 2 stats * D] = [B, 8D]
    """
    B, T, J, D = block_emb.shape
    
    # Step 1: Mean over joints in each original group -> [B, T, D] per group
    group_embeddings = {}
    for group_name, indices in joint_groups.items():
        group_embeddings[group_name] = block_emb[:, :, indices, :].mean(dim=2)  # [B, T, D]
    
    # Step 2: Merge L/R arms and L/R legs by averaging
    merged_embeddings = {}
    for merged_name, source_groups in MERGED_GROUPS.items():
        tensors = [group_embeddings[g] for g in source_groups if g in group_embeddings]
        merged_embeddings[merged_name] = torch.stack(tensors, dim=0).mean(dim=0)  # [B, T, D]
    
    # Step 3: Mean and 90th percentile over time for each merged group
    parts = []
    for merged_name in ['legs', 'arms', 'torso', 'head']:
        emb = merged_embeddings[merged_name]  # [B, T, D]
        mean_pool = emb.mean(dim=1)  # [B, D]
        pct99_pool = torch.quantile(emb, 0.99, dim=1)  # [B, D]
        parts.extend([mean_pool, pct99_pool])
    
    return torch.cat(parts, dim=1).cpu().numpy()


def _pool_v6_batch(sink_emb: torch.Tensor) -> np.ndarray:
    """
    Variant 6: CLS-like pooling from sink tokens.
    Mean + 90th percentile over joints for each sink token, then concat across sink tokens.
    [B, S, J, D] -> [B, S * 2 * D]
    """
    B, S, J, D = sink_emb.shape
    parts = []
    for s in range(S):
        emb = sink_emb[:, s, :, :]  # [B, J, D]
        mean_pool = emb.mean(dim=1)  # [B, D]
        pct90_pool = torch.quantile(emb, 0.90, dim=1)  # [B, D]
        parts.extend([mean_pool, pct90_pool])
    return torch.cat(parts, dim=1).cpu().numpy()


def _get_variant_columns(dim_feat: int, dim_rep: int, num_joints: int, joint_groups: Dict[str, List[int]], num_sink_tokens: int = 0) -> Dict[str, List[str]]:
    """Generate column names for all variants with meaningful joint names.
    
    Args:
        dim_feat: dimension of block_embeddings (used for v1, v2, v3, v5, v6)
        dim_rep: dimension of prelogits_embeddings (used for v4)
        num_sink_tokens: number of sink tokens (for v6)
    """
    # v4 uses prelogits which has dim_rep
    v4_cols = []
    for j in range(num_joints):
        joint_name = NOISE_JOINT_NAMES.get(j, f"joint{j}")
        v4_cols.extend([f"emb_{joint_name}_{i}" for i in range(dim_rep)])

    # v5: 4 merged groups * 2 stats (mean, p90) * dim_feat
    v5_cols = []
    for merged_name in ['legs', 'arms', 'torso', 'head']:
        v5_cols.extend([f"emb_{merged_name}_mean_{i}" for i in range(dim_feat)])
        v5_cols.extend([f"emb_{merged_name}_p90_{i}" for i in range(dim_feat)])

    # v6: sink tokens CLS pooling - S sinks * 2 stats (mean, p90) * dim_feat
    v6_cols = []
    for s in range(num_sink_tokens):
        v6_cols.extend([f"emb_sink{s}_mean_{i}" for i in range(dim_feat)])
        v6_cols.extend([f"emb_sink{s}_p90_{i}" for i in range(dim_feat)])

    return {
        'v1': [f"emb_mean_{i}" for i in range(dim_feat)] + [f"emb_max_{i}" for i in range(dim_feat)],
        'v2': (
            [f"emb_mean_joint_mean_time_{i}" for i in range(dim_feat)] +
            [f"emb_max_joint_mean_time_{i}" for i in range(dim_feat)] +
            [f"emb_std_joint_mean_time_{i}" for i in range(dim_feat)]
        ),
        'v3': [f"emb_group_{g}_{i}" for g in joint_groups.keys() for i in range(dim_feat)],
        'v4': v4_cols,
        'v5': v5_cols,
        'v6': v6_cols,
    }


# =============================================================================
# Extraction and Saving (per-split, saves immediately)
# =============================================================================

def _extract_and_save_split(
    model: ReconstructNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_name: str,
    output_dir: str,
    joint_groups: Dict[str, List[int]],
    columns: Optional[Dict[str, List[str]]] = None,
) -> Tuple[int, int, int, int, int]:
    """
    Extract embeddings for one split and save CSVs immediately.
    Returns (dim_feat, dim_rep, num_joints, num_sink_tokens, num_samples) for column generation.
    """
    model.eval()
    v1_chunks, v2_chunks, v3_chunks, v4_chunks, v5_chunks, v6_chunks = [], [], [], [], [], []
    ids, activities, research_stages, sequence_indices = [], [], [], []
    dim_feat, dim_rep, num_joints, num_sink_tokens = None, None, None, 0

    print(f"\n{'='*50}")
    print(f"Extracting {split_name.upper()} split ({len(loader)} batches)")
    print(f"{'='*50}")

    with torch.no_grad():
        for batch_idx, skeleton_data in enumerate(tqdm(loader, desc=f"{split_name}")):
            data = skeleton_data['data'].to(device)
            _, embeddings_bundle = model(data, mask=None)

            block_emb = embeddings_bundle['block_embeddings']
            prelogits_emb = embeddings_bundle['pre_logits_embeddings']
            sink_emb = embeddings_bundle.get('sink_embeddings')

            if dim_feat is None:
                dim_feat = block_emb.shape[-1]
                dim_rep = prelogits_emb.shape[-1]
                num_joints = block_emb.shape[2]
                if sink_emb is not None:
                    num_sink_tokens = sink_emb.shape[1]

            v1_chunks.append(_pool_v1_batch(block_emb))
            v2_chunks.append(_pool_v2_batch(block_emb))
            v3_chunks.append(_pool_v3_batch(block_emb, joint_groups))
            v4_chunks.append(_pool_v4_batch(prelogits_emb))
            v5_chunks.append(_pool_v5_batch(block_emb, joint_groups))
            if sink_emb is not None:
                v6_chunks.append(_pool_v6_batch(sink_emb))

            ids.extend(skeleton_data['id'])
            activities.extend(skeleton_data['activity'])
            research_stages.extend(skeleton_data['research_stage'])
            if 'sequence_idx' in skeleton_data:
                # Convert tensor to list of ints
                seq_idx = skeleton_data['sequence_idx']
                if hasattr(seq_idx, 'tolist'):
                    seq_idx = seq_idx.tolist()
                sequence_indices.extend(seq_idx)
            else:
                batch_size = len(skeleton_data['id'])
                start_idx = batch_idx * loader.batch_size
                sequence_indices.extend(range(start_idx, start_idx + batch_size))

    if not v1_chunks:
        print(f"No data in {split_name} split")
        return dim_feat, dim_rep, num_joints, num_sink_tokens, 0

    # Concatenate all chunks
    pooled_data = {
        'v1': np.concatenate(v1_chunks, axis=0),
        'v2': np.concatenate(v2_chunks, axis=0),
        'v3': np.concatenate(v3_chunks, axis=0),
        'v4': np.concatenate(v4_chunks, axis=0),
        'v5': np.concatenate(v5_chunks, axis=0),
    }
    if v6_chunks:
        pooled_data['v6'] = np.concatenate(v6_chunks, axis=0)

    # Generate columns if not provided
    if columns is None:
        columns = _get_variant_columns(dim_feat, dim_rep, num_joints, joint_groups, num_sink_tokens)

    # Save each variant immediately
    print(f"\nSaving {split_name} CSVs...")
    for vkey, vname in VARIANT_NAMES.items():
        if vkey not in pooled_data:
            continue  # Skip v6 if no sink tokens
        variant_dir = os.path.join(output_dir, vname)
        os.makedirs(variant_dir, exist_ok=True)

        df = pd.DataFrame(pooled_data[vkey], columns=columns[vkey])
        df.insert(0, 'seq_idx', sequence_indices)
        df.insert(0, 'activity', activities)
        df.insert(0, 'research_stage', research_stages)
        df.insert(0, 'RegistrationCode', ids)

        csv_path = os.path.join(variant_dir, f"{split_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved {vname}/{split_name}.csv ({len(df)} rows)")

    return dim_feat, dim_rep, num_joints, num_sink_tokens, len(ids)


def _combine_split_csvs(output_dir: str) -> None:
    """Read all split CSVs and create combined 'all.csv' for each variant."""
    print(f"\n{'='*50}")
    print("Creating combined 'all.csv' files...")
    print(f"{'='*50}")

    for vkey, vname in VARIANT_NAMES.items():
        variant_dir = os.path.join(output_dir, vname)
        split_files = []

        for split_name in ['eval', 'test', 'train']:
            csv_path = os.path.join(variant_dir, f"{split_name}.csv")
            if os.path.exists(csv_path):
                split_files.append(csv_path)

        if split_files:
            dfs = [pd.read_csv(f) for f in split_files]
            combined = pd.concat(dfs, ignore_index=True)
            all_path = os.path.join(variant_dir, "all.csv")
            combined.to_csv(all_path, index=False)
            print(f"  Saved {vname}/all.csv ({len(combined)} rows)")


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_embeddings(
    run_id: str,
    epoch: Optional[int] = None,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Main entry point: load model, extract embeddings, save all 4 variants.
    Processes eval first, saves each split immediately, combines at end.
    
    Args:
        run_id: wandb run ID
        epoch: specific epoch to load (None = latest)
        output_dir: where to save CSVs (default: {model_dir}/embeddings/)
        device: torch device
        
    Returns:
        Path to output directory
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading run {run_id}...")
    config, args_cfg, model_dir = load_run_config_and_args(run_id)
    model, loaders, clean_args = load_model_and_loaders(config, args_cfg, model_dir, epoch, device)

    output_dir = output_dir or os.path.join(model_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Extract and save each split (eval first)
    columns = None
    total_samples = 0

    num_sink_tokens = 0
    for split_name, loader in loaders.items():
        dim_feat, dim_rep, num_joints, num_sink, n_samples = _extract_and_save_split(
            model, loader, device, split_name, output_dir, JOINT_GROUPS, columns
        )
        total_samples += n_samples
        if num_sink > 0:
            num_sink_tokens = num_sink

        # Generate columns from first split with data
        if columns is None and dim_feat is not None:
            columns = _get_variant_columns(dim_feat, dim_rep, num_joints, JOINT_GROUPS, num_sink_tokens)

    # Combine all splits into 'all.csv' by reading back the saved CSVs
    _combine_split_csvs(output_dir)

    print(f"\n{'='*50}")
    print(f"DONE! Extracted {total_samples} total samples")
    print(f"Output: {output_dir}")
    print(f"{'='*50}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings from a trained model")
    parser.add_argument("run_id", type=str, help="WandB run ID")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load (default: latest)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    cli_args = parser.parse_args()
    extract_embeddings(
        run_id=cli_args.run_id,
        epoch=cli_args.epoch,
        output_dir=cli_args.output_dir,
        device=torch.device(cli_args.device),
    )
