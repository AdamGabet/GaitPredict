import os
import numpy as np
import torch
import torch.nn as nn
from functools import partial

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, SequentialLR, ConstantLR, LinearLR
from torch.utils.data import DataLoader
import re
from model.training.utils.loss import loss_mpjpe, n_mpjpe, loss_velocity, loss_quat_geodesic

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, scaler=None):
    """
    Loads a checkpoint containing model, optimizer, scheduler and scaler states.
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        scheduler: The scheduler to load state into
        scaler: The GradScaler to load state into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        model, optimizer, scheduler, scaler, start_epoch
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Fix state_dict for model - handle DataParallel prefix
    model_state = checkpoint['model']
    
     # Next, strip any TorchDynamo I wrap first with DataParallel so remove it second
    if any(k.startswith('_orig_mod.') for k in model_state):
        print("Stripping Dynamo wrapper prefix")
        model_state = {k[len('_orig_mod.'):]: v for k, v in model_state.items()}

    # First, strip any DataParallel
    if any(k.startswith('module.') for k in model_state):
        print("Stripping DataParallel prefix")
        model_state = {k[len('module.'):]: v for k, v in model_state.items()}

    # model_state now has clean keys like 'model_backbone.temp_embed', etc.
    print("Final state_dict keys sample:", list(model_state.keys())[:5])
    
    # Load state dict
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Load scaler state if it exists in the checkpoint
    if 'scaler' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    
    start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
    
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    
    return model, optimizer, scheduler, scaler, start_epoch



def load_pretrained_weights(model, checkpoint):
    """Load pretrianed weights to model
    Incompatible layers (unmatched in name or size) will be ignored
    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)
    print('load_weight', len(matched_layers))
    return model

import os

def get_latest_dir(path):
    """
    Returns the latest .pth file in a directory.
    Accepts either:
      - a model file path (e.g., ".../checkpoints/model_12.pth"), or
      - a direct directory path (e.g., ".../checkpoints/")

    Sorting is based on the numeric epoch suffix before ".pth".
    """
    # If input is a file, get its directory
    if os.path.isfile(path):
        model_dir = os.path.dirname(path)
    else:
        model_dir = path.rstrip('/')

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    # List all .pth files
    files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")

    # Sort by epoch number if present, fallback to lexicographic
    try:
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    except ValueError:
        files.sort()

    latest_file = files[-1]
    return os.path.join(model_dir, latest_file)



def find_model_dir_for_run(run_id: str, base_model_dir: str):
    """
    Find the model directory for a given run ID.

    Args:
        run_id: WandB run ID
        base_model_dir: Base directory where models are saved

    Returns:
        Path to model directory if found, None otherwise
    """
    if not os.path.isdir(base_model_dir):
        return None

    # Regex pattern equivalent to "bert_<run_id>_*"
    pattern = re.compile(rf"^bert_{re.escape(run_id)}_.+")

    # Collect matching subdirectories
    matching_dirs = [
        os.path.join(base_model_dir, d)
        for d in os.listdir(base_model_dir)
        if os.path.isdir(os.path.join(base_model_dir, d)) and pattern.match(d)
    ]

    if not matching_dirs:
        return None

    # Return the most recently modified one
    matching_dirs.sort(key=os.path.getmtime)
    return str(matching_dirs[-1]) + "/"


def get_scheduler(optimizer, config):
    steps = config.get('steps_per_epoch', 1)  # default = per-epoch
    # allows for fractional warmup epochs
    warmup_ratio = config.get('warmup_ratio', None)
    # backward compatibility
    if warmup_ratio is not None:
        warmup_iters = int(warmup_ratio * steps * config['num_epochs'])
    else:
        warmup_iters = int(config['warmup_epochs'] * steps)

    T_max = config['num_epochs'] * steps - warmup_iters

    min_lr = config.get('learning_rate', None) 
    if min_lr is not None:
        min_lr = min_lr * 0.01

    if config['scheduler_type'] == 'Annealing':
        return SequentialLR(
            optimizer,
            schedulers=[
                ConstantLR(optimizer, factor=1.0, total_iters=warmup_iters),
                CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr)    
            ],
            milestones=[warmup_iters]
        )
    else:
        return StepLR(optimizer, step_size=1, gamma=0.99)


class MaskingScheduler:
    def __init__(self, warmup_epochs, total_epochs, start_ratio, end_ratio, group_masking=False):
        """
        scheduler for masking ratio
        will have a function that linearly increases the masking ratio
        :param warmup_epochs:
        :param total_epochs:
        :param start_ratio:
        :param end_ratio:
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.group_masking = group_masking

    def step_epoch(self, epoch, loader, dataset, batch_size, args):
        if epoch < self.warmup_epochs or not self.group_masking:
            return loader
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            ratio = self.start_ratio
            span = 16
            random_mask = 0.15
            random_frame_mask = 0.05
            dataset.args.random_mask = random_mask
            dataset.args.random_mask_frames = random_frame_mask
            dataset.args.group_masking = ratio
            dataset.args.span_masking = span
            # also update args
            args.random_mask = random_mask
            args.random_mask_frames = random_frame_mask
            args.group_masking = ratio
            args.span_masking = span


            loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=3,
                                      pin_memory=True,
                                      persistent_workers=True)

            return loader, args


def _pool_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Pool transformer embeddings across time and joints."""
    # flatten time and joint dimensions before pooling
    B, F, J, D = embeddings.shape
    flat = embeddings.view(B, F * J, D)
    mean_pool = flat.mean(dim=1)
    max_pool = flat.max(dim=1).values
    return torch.cat([mean_pool, max_pool], dim=1)


def _collect_model_outputs(model, loader, device, collect_losses=True, debug_mode=False, track_progress=False, num_sink_tokens=0):
    """Run the model and gather pooled embeddings plus optional reconstruction metrics."""
    was_training = model.training
    model.eval()

    metric_keys = [
        'eval_loss_3d_reconstruction',
        'eval_loss_3d_scale',
        'eval_loss_velocity',
    ]
    metrics_accumulator = {key: 0.0 for key in metric_keys} if collect_losses else {}
    activity_loss_totals = {} if collect_losses else None
    activity_counts = {} if collect_losses else None
    embedding_chunks = []
    block_embeddings_chunks = []
    label_buckets = []
    activities = []
    ids = []
    research_stages = []
    sequence_indices = []

    with torch.no_grad():
        for skel_idx, skeleton_data in enumerate(loader):
            if track_progress and (skel_idx % 50 == 0):
                print(f"Processing batch {skel_idx}/{len(loader)}", end=",")
            data = skeleton_data['data'].to(device)

            original_full = skeleton_data['original'].to(device)
            original_data = original_full[:, :, :, :3]

            # Truncate last frames to make room for sink tokens
            if num_sink_tokens > 0:
                data = data[:, :-num_sink_tokens, :, :]
                original_data = original_data[:, :-num_sink_tokens, :, :]
                original_full = original_full[:, :-num_sink_tokens, :, :]

            reconstructed, embeddings_bundle = model(data, mask=None)
            predicted_3d_pos = reconstructed[:, :, :, :3]

            quaternion_data = None
            predicted_3d_quat = None
            if reconstructed.shape[-1] > 3 and original_full.shape[-1] > 5:
                predicted_3d_quat = reconstructed[:, :, :, 3:]
                quaternion_data = original_full[:, :, :, 4:8]
                if collect_losses and 'eval_loss_quaternion' not in metrics_accumulator:
                    metrics_accumulator['eval_loss_quaternion'] = 0.0
                    metric_keys.append('eval_loss_quaternion')
                    if activity_loss_totals is not None:
                        for totals in activity_loss_totals.values():
                            totals['eval_loss_quaternion'] = 0.0

            pooled_embeddings = embeddings_bundle['pooled_pre_logits_embeddings']
            embedding_chunks.append(pooled_embeddings.cpu().numpy())
            pooled_block_embeddings = embeddings_bundle['pooled_block_embeddings']
            block_embeddings_chunks.append(pooled_block_embeddings.cpu().numpy())

            batch_labels = skeleton_data['label_list']
            for idx, label_tensor in enumerate(batch_labels):
                if idx >= len(label_buckets):
                    label_buckets.append([])
                label_buckets[idx].extend(label_tensor.tolist())

            activities.extend(skeleton_data['activity'])
            ids.extend(skeleton_data['id'])
            research_stages.extend(skeleton_data['research_stage'])
            if 'sequence_idx' in skeleton_data:
                sequence_indices.extend(skeleton_data['sequence_idx'])
            if collect_losses:
                # keep per-activity accumulators ready
                if activity_loss_totals is None:
                    activity_loss_totals = {}
                if activity_counts is None:
                    activity_counts = {}

                batch_activities = skeleton_data['activity']
                unique_batch_activities = sorted(set(batch_activities))

                for activity in unique_batch_activities:
                    indices = [idx for idx, act in enumerate(batch_activities) if act == activity]
                    if not indices:
                        continue
                    idx_tensor = torch.tensor(indices, device=device)
                    act_pred_pos = predicted_3d_pos.index_select(0, idx_tensor)
                    act_target_pos = original_data.index_select(0, idx_tensor)

                    activity_counts[activity] = activity_counts.get(activity, 0) + len(indices)
                    activity_loss_totals.setdefault(activity, {key: 0.0 for key in metric_keys})

                    # MPJPE-style reconstruction loss
                    loss_recon = loss_mpjpe(act_pred_pos, act_target_pos).item()
                    metrics_accumulator['eval_loss_3d_reconstruction'] += loss_recon * len(indices)
                    activity_loss_totals[activity]['eval_loss_3d_reconstruction'] += loss_recon * len(indices)

                    # Normalized MPJPE
                    loss_scale = n_mpjpe(act_pred_pos, act_target_pos).item()
                    metrics_accumulator['eval_loss_3d_scale'] += loss_scale * len(indices)
                    # activity_loss_totals[activity]['eval_loss_3d_scale'] += loss_scale * len(indices)

                    # Velocity loss
                    loss_vel = loss_velocity(act_pred_pos, act_target_pos).item()
                    metrics_accumulator['eval_loss_velocity'] += loss_vel * len(indices)
                    # activity_loss_totals[activity]['eval_loss_velocity'] += loss_vel * len(indices)

                    # Quaternion loss if present
                    if predicted_3d_quat is not None and quaternion_data is not None:
                        act_pred_quat = predicted_3d_quat.index_select(0, idx_tensor)
                        act_target_quat = quaternion_data.index_select(0, idx_tensor)
                        loss_quat = loss_quat_geodesic(act_pred_quat, act_target_quat).item()
                        metrics_accumulator['eval_loss_quaternion'] += loss_quat * len(indices)
                        activity_loss_totals[activity]['eval_loss_quaternion'] += loss_quat * len(indices)
            if debug_mode and skel_idx > 80:
                print("Debug mode active - breaking after 200 batches")
                break


    if collect_losses and len(loader) > 0:
        for activity, totals in activity_loss_totals.items():
            count = max(activity_counts.get(activity, 0), 1)
            for key in totals:
                totals[key] = totals[key] / count
        total_count = sum(activity_counts.values())
        if total_count > 0:
            for key in metrics_accumulator:
                metrics_accumulator[key] = metrics_accumulator[key] / total_count

    if was_training:
        model.train()

    embedding_array = np.concatenate(embedding_chunks, axis=0) if embedding_chunks else np.empty((0, 0))
    block_embeddings_array = np.concatenate(block_embeddings_chunks, axis=0) if block_embeddings_chunks else np.empty((0, 0))

    return {
        'metrics': metrics_accumulator if collect_losses else {},
        'activity_losses': activity_loss_totals if collect_losses else {},
        'activity_counts': activity_counts if collect_losses else {},
        "pre_logits_embeddings": embedding_array,
        "block_embeddings": block_embeddings_array,
        'labels': label_buckets,
        'activities': activities,
        'ids': ids,
        'research_stage': research_stages,
        'sequence_indices': sequence_indices
    }
