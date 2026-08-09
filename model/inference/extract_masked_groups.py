"""
Embedding extraction script with joint group masking.
Extracts embeddings with specific joint groups masked (zeroed out).
Saves v1 and v5 pooling variants only.
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
from model.training.utils.training_helper import get_latest_dir, find_model_dir_for_run, select_device

load_dotenv(interpolate=True)

# Valid research stages for cleaned_pooled.csv
VALID_STAGES = ['baseline', '02_00_visit', '04_00_visit', '06_00_visit']

# Joint groups for masking - each will be zeroed out separately
MASK_GROUPS = {
    'head': [20, 21, 22, 23, 24, 25],           # head_full
    'arms': [4, 5, 6, 7, 8, 9, 10, 11],         # L_arm + R_arm
    'legs': [12, 13, 14, 15, 16, 17, 18, 19],   # L_leg + R_leg
    'torso': [0, 1, 2, 3],                      # torso
}

# Merged joint groups for v5 pooling
MERGED_GROUPS = {
    'legs': ['L_leg', 'R_leg'],
    'arms': ['L_arm', 'R_arm'],
    'torso': ['torso'],
    'head': ['head_full'],
}

VARIANT_NAMES = {
    'v1': 'variant1_mean_max',
    'v5': 'variant5_merged_groups_percentile',
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

    loaded_args = args.load_args_from_path(model_dir)
    args_cfg = copy.deepcopy(args.apply_args(loaded_args))
    print(f"Args config: {args_cfg}")

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
    
    # Drop unwanted heads/layers
    EXCLUDE_PREFIXES = ('koleo_head',)
    model_state = {
        k: v for k, v in model_state.items()
        if not k.startswith(EXCLUDE_PREFIXES)
    }

    # load_state_dict copies values into the model's *existing* parameter
    # tensors, preserving whatever device those were already on -- it does
    # NOT move the model to match map_location. Move explicitly first.
    model = model.to(device)
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
    device = device or select_device("auto")

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
# Per-Batch Pooling Functions (v1 and v5 only)
# =============================================================================

def _pool_v1_batch(block_emb: torch.Tensor) -> np.ndarray:
    """Variant 1: Mean+Max across time and joints. [B,T,J,D] -> [B, 2D]"""
    B, T, J, D = block_emb.shape
    flat = block_emb.view(B, T * J, D)
    mean_pool = flat.mean(dim=1)
    max_pool = flat.max(dim=1).values
    return torch.cat([mean_pool, max_pool], dim=1).cpu().numpy()


def _pool_v5_batch(block_emb: torch.Tensor, joint_groups: Dict[str, List[int]]) -> np.ndarray:
    """
    Variant 5: Mean over joints per group, merge L/R limbs, then mean + 99th percentile over time.
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
    
    # Step 3: Mean and 99th percentile over time for each merged group
    parts = []
    for merged_name in ['legs', 'arms', 'torso', 'head']:
        emb = merged_embeddings[merged_name]  # [B, T, D]
        mean_pool = emb.mean(dim=1)  # [B, D]
        pct99_pool = torch.quantile(emb, 0.99, dim=1)  # [B, D]
        parts.extend([mean_pool, pct99_pool])
    
    return torch.cat(parts, dim=1).cpu().numpy()


def _get_variant_columns(dim_feat: int) -> Dict[str, List[str]]:
    """Generate column names for v1 and v5 variants."""
    # v5: 4 merged groups * 2 stats (mean, p99) * dim_feat
    v5_cols = []
    for merged_name in ['legs', 'arms', 'torso', 'head']:
        v5_cols.extend([f"emb_{merged_name}_mean_{i}" for i in range(dim_feat)])
        v5_cols.extend([f"emb_{merged_name}_p99_{i}" for i in range(dim_feat)])

    return {
        'v1': [f"emb_mean_{i}" for i in range(dim_feat)] + [f"emb_max_{i}" for i in range(dim_feat)],
        'v5': v5_cols,
    }


# =============================================================================
# Extraction with Masking
# =============================================================================

def _extract_and_save_split_with_mask(
    model: ReconstructNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    split_name: str,
    output_dir: str,
    mask_group_name: str,
    mask_indices: List[int],
    joint_groups: Dict[str, List[int]],
    columns: Optional[Dict[str, List[str]]] = None,
) -> Tuple[int, int]:
    """
    Extract embeddings for one split with specific joints masked.
    Returns (dim_feat, num_samples) for column generation.
    """
    model.eval()
    v1_chunks, v5_chunks = [], []
    ids, activities, research_stages, sequence_indices = [], [], [], []
    dim_feat = None

    print(f"\n{'='*50}")
    print(f"Extracting {split_name.upper()} split with {mask_group_name} masked ({len(loader)} batches)")
    print(f"{'='*50}")

    with torch.no_grad():
        for batch_idx, skeleton_data in enumerate(tqdm(loader, desc=f"{split_name}")):
            data = skeleton_data['data'].to(device)
            
            # Apply masking: zero out the specified joint group
            # data shape: [B, T, J, D] where J is joints dimension
            data[:, :, mask_indices, :] = 0
            
            _, embeddings_bundle = model(data, mask=None)

            block_emb = embeddings_bundle['block_embeddings']

            if dim_feat is None:
                dim_feat = block_emb.shape[-1]

            v1_chunks.append(_pool_v1_batch(block_emb))
            v5_chunks.append(_pool_v5_batch(block_emb, joint_groups))

            ids.extend(skeleton_data['id'])
            activities.extend(skeleton_data['activity'])
            research_stages.extend(skeleton_data['research_stage'])
            if 'sequence_idx' in skeleton_data:
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
        return dim_feat, 0

    # Concatenate all chunks
    pooled_data = {
        'v1': np.concatenate(v1_chunks, axis=0),
        'v5': np.concatenate(v5_chunks, axis=0),
    }

    # Generate columns if not provided
    if columns is None:
        columns = _get_variant_columns(dim_feat)

    # Save each variant immediately
    print(f"\nSaving {split_name} CSVs...")
    for vkey, vname in VARIANT_NAMES.items():
        variant_dir = os.path.join(output_dir, mask_group_name, vname)
        os.makedirs(variant_dir, exist_ok=True)

        df = pd.DataFrame(pooled_data[vkey], columns=columns[vkey])
        df.insert(0, 'seq_idx', sequence_indices)
        df.insert(0, 'activity', activities)
        df.insert(0, 'research_stage', research_stages)
        df.insert(0, 'RegistrationCode', ids)

        csv_path = os.path.join(variant_dir, f"{split_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved {mask_group_name}/{vname}/{split_name}.csv ({len(df)} rows)")

    return dim_feat, len(ids)


def _combine_split_csvs(output_dir: str, mask_group_name: str) -> None:
    """Read all split CSVs and create combined 'all.csv' for each variant."""
    print(f"\n{'='*50}")
    print(f"Creating combined 'all.csv' files for {mask_group_name}...")
    print(f"{'='*50}")

    for vkey, vname in VARIANT_NAMES.items():
        variant_dir = os.path.join(output_dir, mask_group_name, vname)
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
            print(f"  Saved {mask_group_name}/{vname}/all.csv ({len(combined)} rows)")


def _create_pooled_csvs(output_dir: str, mask_group_name: str) -> None:
    """Create pooled.csv and cleaned_pooled.csv for each variant."""
    print(f"\n{'='*50}")
    print(f"Creating pooled CSVs for {mask_group_name}...")
    print(f"{'='*50}")

    for vkey, vname in VARIANT_NAMES.items():
        variant_dir = os.path.join(output_dir, mask_group_name, vname)
        all_path = os.path.join(variant_dir, "all.csv")
        
        if not os.path.exists(all_path):
            print(f"  Warning: {all_path} not found, skipping...")
            continue
        
        df = pd.read_csv(all_path)
        
        # Get embedding columns
        emb_cols = [c for c in df.columns if c.startswith('emb_')]
        
        # Create pooled.csv (all stages, no seq_idx)
        print(f"  Creating pooled.csv for {mask_group_name}/{vname}...")
        agg_dict = {col: 'median' for col in emb_cols}
        pooled = df.groupby(['RegistrationCode', 'research_stage', 'activity']).agg(agg_dict).reset_index()
        pooled_path = os.path.join(variant_dir, "pooled.csv")
        pooled.to_csv(pooled_path, index=False)
        print(f"    Saved pooled.csv ({len(pooled)} rows)")
        
        # Create cleaned_pooled.csv (filtered stages, no seq_idx)
        print(f"  Creating cleaned_pooled.csv for {mask_group_name}/{vname}...")
        df_filtered = df[df['research_stage'].isin(VALID_STAGES)].copy()
        cleaned_pooled = df_filtered.groupby(['RegistrationCode', 'research_stage', 'activity']).agg(agg_dict).reset_index()
        cleaned_pooled_path = os.path.join(variant_dir, "cleaned_pooled.csv")
        cleaned_pooled.to_csv(cleaned_pooled_path, index=False)
        print(f"    Saved cleaned_pooled.csv ({len(cleaned_pooled)} rows, filtered to {VALID_STAGES})")


# =============================================================================
# Main Extraction Function
# =============================================================================

def extract_masked_embeddings(
    run_id: str,
    epoch: Optional[int] = None,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    Main entry point: load model, extract embeddings with each group masked.
    
    Args:
        run_id: wandb run ID
        epoch: specific epoch to load (None = latest)
        output_dir: where to save CSVs (default: {model_dir}/embeddings_masked/)
        device: torch device
        
    Returns:
        Path to output directory
    """
    device = device or select_device("auto")

    print(f"Loading run {run_id}...")
    config, args_cfg, model_dir = load_run_config_and_args(run_id)
    model, loaders, clean_args = load_model_and_loaders(config, args_cfg, model_dir, epoch, device)

    output_dir = output_dir or os.path.join(model_dir, "embeddings_masked")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Process each mask group
    for mask_group_name, mask_indices in MASK_GROUPS.items():
        print(f"\n{'='*70}")
        print(f"PROCESSING MASK GROUP: {mask_group_name.upper()}")
        print(f"Masking joints: {mask_indices}")
        print(f"{'='*70}")
        
        columns = None
        total_samples = 0

        # Extract and save each split
        for split_name, loader in loaders.items():
            dim_feat, n_samples = _extract_and_save_split_with_mask(
                model, loader, device, split_name, output_dir, 
                mask_group_name, mask_indices, noise_groups, columns
            )
            total_samples += n_samples

            # Generate columns from first split with data
            if columns is None and dim_feat is not None:
                columns = _get_variant_columns(dim_feat)

        # Combine all splits into 'all.csv'
        _combine_split_csvs(output_dir, mask_group_name)
        
        # Create pooled and cleaned_pooled CSVs
        _create_pooled_csvs(output_dir, mask_group_name)

        print(f"\n{'='*70}")
        print(f"DONE with {mask_group_name}! Extracted {total_samples} total samples")
        print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"ALL MASK GROUPS COMPLETE!")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract embeddings with joint groups masked")
    parser.add_argument("run_id", type=str, help="WandB run ID")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch to load (default: latest)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/mps/cpu)")

    cli_args = parser.parse_args()
    extract_masked_embeddings(
        run_id=cli_args.run_id,
        epoch=cli_args.epoch,
        output_dir=cli_args.output_dir,
        device=select_device(cli_args.device),
    )



