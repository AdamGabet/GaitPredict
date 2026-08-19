import os
import warnings

import torch
import wandb
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

from model.training.utils.loss import loss_mpjpe, n_mpjpe, loss_velocity, loss_quat_geodesic
from model.training.utils.linear_probe import run_cross_validated_ridge_probe
from model.training.utils.linear_probe import run_cross_validated_ridge_probe_with_ensemble, probe_eval
from model.training.utils.training_helper import _collect_model_outputs
import wandb

warnings.filterwarnings("ignore", category=FutureWarning)


def _unwrap_model(model, device):
    """Strip model of torch.compile and DataParallel wrappers to get clean model for eval."""
    unwrapped = model
    
    # Strip torch.compile wrapper (_orig_mod)
    if hasattr(unwrapped, '_orig_mod'):
        unwrapped = unwrapped._orig_mod
    
    # Strip DataParallel/DDP wrapper (module)
    if hasattr(unwrapped, 'module'):
        unwrapped = unwrapped.module
    
    # Ensure model is on the correct device
    unwrapped = unwrapped.to(device)
    
    return unwrapped


def _collect_attention_samples(model, loader, device, activities, attention_cfg):
    """Capture one attention bundle per activity and build visualization payloads.
    
    Note: Expects model to already be unwrapped (no torch.compile or DataParallel wrappers).
    """
    if not activities or not attention_cfg.get('enabled', False):
        return {}

    unique_activities = sorted(set(activities))
    max_per_activity = max(attention_cfg.get('max_per_activity', 1), 1)
    remaining = {activity: 0 for activity in unique_activities}
    log_payload = {}

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for skeleton_data in loader:
            data = skeleton_data['data'].to(device)

            batch_activities = skeleton_data['activity']
            for idx, activity in enumerate(batch_activities):
                if activity not in remaining:
                    continue
                if remaining[activity] >= max_per_activity:
                    continue

                sample_data = data[idx:idx + 1]

                # Model is already clean (unwrapped), safe to capture attention
                reconstructed, outputs = model(sample_data, mask=None, capture_attention=True)
                # ReconstructNet returns different tuples depending on flags
                attention_bundle = outputs['attention_maps']

                if attention_bundle:
                    pass  # attention visualization not included in this release



                remaining[activity] += 1
                if all(count >= max_per_activity for count in remaining.values()):
                    break

            if all(count >= max_per_activity for count in remaining.values()):
                break

    if was_training:
        model.train()

    return log_payload

def _save_embeddings_to_csv(embedding_array, ids, activities, research_stage, save_dir, filename, sequence_indices=None):
    """Persist pooled embeddings alongside metadata."""
    if embedding_array.size == 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(embedding_array)
    df.columns = [f'embedding_{i + 1}' for i in range(df.shape[1])]
    if sequence_indices:
        df['sequence_idx'] = sequence_indices
    if ids:
        df['subject_id'] = ids
    if activities:
        df['activity'] = activities
    if research_stage:
        df['research_stage'] = research_stage
    cols = list(df.columns)
    if 'subject_id' in cols:
        cols.insert(0, cols.pop(cols.index('subject_id')))
    if 'research_stage' in cols:
        idx = 1 if 'subject_id' in cols else 0
        cols.insert(idx, cols.pop(cols.index('research_stage')))
    if 'activity' in cols:
        idx = 2 if 'subject_id' in cols else 0
        cols.insert(idx, cols.pop(cols.index('activity')))
    if 'sequence_idx' in cols:
        idx = 3 if 'subject_id' in cols else 0
        cols.insert(idx, cols.pop(cols.index('sequence_idx')))
    # rename subject_id to RegistrationCode for consistency
    df = df[cols]
    df.rename(columns={'subject_id': 'RegistrationCode'}, inplace=True)

    df.to_csv(os.path.join(save_dir, filename), index=False)


def _create_umap_plot(embedding_array, activities, save_path=None, tag="epoch"):
    """Draw and optionally persist a UMAP projection of the embeddings."""
    if embedding_array.shape[0] <= 10:
        return
    reducer = umap.UMAP()
    embedding_umap = reducer.fit_transform(embedding_array)

    plt.figure(figsize=(12, 10))
    if activities:
        unique_activities = list(set(activities))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_activities)))
        for color, activity in zip(colors, unique_activities):
            mask = [a == activity for a in activities]
            plt.scatter(embedding_umap[mask, 0], embedding_umap[mask, 1], color=color, label=activity, alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(embedding_umap[:, 0], embedding_umap[:, 1], alpha=0.7)

    plt.title(f"UMAP Projection of Evaluation Motion Embeddings - {tag}")
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"umap_{tag}.png"), dpi=300)
    wandb.log({f"visualizations/umap_{tag}": wandb.Image(plt)})
    plt.close()


def run_quick_eval(model,
                   eval_loader,
                   device,
                   epoch=None,
                   log_to_wandb=True,
                   attention_config=None,
                   debug_mode=False,
                   run_labels=None,
                   label_names=None,
                   task_types=None,
                   all_metrics=False,
                   masked_eval_loader=None,
                   num_sink_tokens=0,
                   ):
    """Quick per-epoch reconstruction and embedding-health evaluation."""
    # Create clean model without torch.compile or DataParallel wrappers for evaluation
    clean_model = _unwrap_model(model, device)
    
    # gather embeddings and reconstruction stats
    print("Running quick evaluation... debug_mode=", debug_mode)
    outputs = _collect_model_outputs(clean_model, eval_loader, device, collect_losses=True, debug_mode=debug_mode, num_sink_tokens=num_sink_tokens)
    if masked_eval_loader is not None:
        masked_outputs = _collect_model_outputs(clean_model, masked_eval_loader, device, collect_losses=True, debug_mode=debug_mode, num_sink_tokens=num_sink_tokens)
        outputs['masked_metrics'] = masked_outputs['metrics']
        outputs['masked_activity_losses'] = masked_outputs['activity_losses']

    attention_payload = {}
    if attention_config and attention_config.get('enabled', False):
        attn_cfg = dict(attention_config)
        if epoch is not None:
            attn_cfg['epoch'] = epoch
        attention_payload = _collect_attention_samples(
            model=clean_model,
            loader=eval_loader,
            device=device,
            activities=outputs['activities'],
            attention_cfg=attn_cfg,
        )

    # Always refresh the UMAP projection each quick evaluation.
    umap_tag = f"quick_epoch_{epoch}" if epoch is not None else "quick_eval"
    _create_umap_plot(outputs['block_embeddings'], outputs['activities'], save_path=None, tag=umap_tag)
    

    # fit lightweight readouts to quantify correlations
    label_names = eval_loader.dataset.label_names
    probe_metrics = {}
    if run_labels is not None:
        probe_metrics = probe_eval(label_names, task_types, run_labels, outputs, all_metrics=all_metrics, wandb_prefix="probe_metrics")


    if log_to_wandb and epoch is not None:
        # log quick metrics to wandb for tracking
        log_payload = {f"eval_loss/{k}": v for k, v in outputs['metrics'].items()}
        if outputs.get('activity_losses'):
            for activity, losses in outputs['activity_losses'].items():
                for loss_name, value in losses.items():
                    log_payload[f"eval_loss_activity/{activity}/{loss_name}"] = value
        if outputs.get('masked_metrics'):
            for metric_name, value in outputs['masked_metrics'].items():
                log_payload[f"masked_eval_loss/{metric_name}"] = value
        if outputs.get('masked_activity_losses'):
            for activity, losses in outputs['masked_activity_losses'].items():
                for loss_name, value in losses.items():
                    log_payload[f"masked_eval_loss_activity/{activity}/{loss_name}"] = value
        log_payload['epoch'] = epoch
        if attention_payload:
            log_payload.update(attention_payload)
        if probe_metrics:
            log_payload.update(probe_metrics)
        wandb.log(log_payload)
    elif log_to_wandb and attention_payload:
        wandb.log(attention_payload)

    return probe_metrics


def run_final_evaluation(model, train_loader, eval_loader, test_loader, device, label_names, task_types, track_progress=False,
                         save_path=None, umap_view=False, debug_mode=False, num_sink_tokens=0):
    """Full post-training evaluation with embedding probes and correlations."""
    print("Running final evaluation... debug_mode=", debug_mode)
    
    # Create clean model without torch.compile or DataParallel wrappers for evaluation
    clean_model = _unwrap_model(model, device)
    
    # collect embeddings for both splits
    eval_outputs = _collect_model_outputs(clean_model, eval_loader, device, collect_losses=True, debug_mode=debug_mode, track_progress=track_progress, num_sink_tokens=num_sink_tokens)
    train_outputs = _collect_model_outputs(clean_model, train_loader, device, collect_losses=False, debug_mode=debug_mode, track_progress=track_progress, num_sink_tokens=num_sink_tokens)
    test_outputs = _collect_model_outputs(clean_model, test_loader, device, collect_losses=False, debug_mode=debug_mode, track_progress=track_progress, num_sink_tokens=num_sink_tokens)
    if save_path:
        for embeddings_type in [a for a in train_outputs.keys() if "embedding" in a]:
            embeddings_dir = os.path.join(save_path, embeddings_type)
            os.makedirs(embeddings_dir, exist_ok=True)
            if eval_outputs[embeddings_type].size > 0:
                _save_embeddings_to_csv(
                    eval_outputs[embeddings_type],
                    eval_outputs['ids'],
                    eval_outputs['activities'],
                    eval_outputs['research_stage'],
                    embeddings_dir,
                    "eval_embeddings.csv",
                    sequence_indices=eval_outputs['sequence_indices'],
                )
            if train_outputs[embeddings_type].size > 0:
                _save_embeddings_to_csv(
                    train_outputs[embeddings_type],
                    train_outputs['ids'],
                    train_outputs['activities'],
                    train_outputs['research_stage'],
                    embeddings_dir,
                    "train_embeddings.csv",
                    sequence_indices=train_outputs['sequence_indices'],
                )
            if test_outputs[embeddings_type].size > 0:
                _save_embeddings_to_csv(
                    test_outputs[embeddings_type],
                    test_outputs['ids'],
                    test_outputs['activities'],
                    test_outputs['research_stage'],
                    embeddings_dir,
                    "test_embeddings.csv",
                    sequence_indices=test_outputs['sequence_indices'],
                )

    metrics = {}
    core_eval_metrics = {f"final_eval_loss/{k}": v for k, v in eval_outputs['metrics'].items()}
    metrics.update(core_eval_metrics)
    if core_eval_metrics:
        wandb.log(core_eval_metrics)

    if eval_outputs.get('activity_losses'):
        for activity, losses in eval_outputs['activity_losses'].items():
            activity_payload = {
                f"final_eval_loss_activity/{activity}/{loss_name}": value for loss_name, value in losses.items()
            }
            metrics.update(activity_payload)
            wandb.log(activity_payload)

    # fit lightweight readouts to quantify correlations
    run_labels = label_names
    probe_metrics = {}
    if run_labels is not None:
        probe_metrics = probe_eval(label_names, task_types, run_labels, train_outputs, all_metrics=False, wandb_prefix="final_eval_probe_metrics")

    metrics.update(probe_metrics)
    if probe_metrics:
        wandb.log(probe_metrics)

    if umap_view:
        _create_umap_plot(eval_outputs['block_embeddings'], eval_outputs['activities'], save_path, tag="final_eval_umap")

    return metrics
