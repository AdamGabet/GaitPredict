import copy
import gc
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.preprocessing import args, joints_file
from model.preprocessing.preprocessing import get_datasets
import wandb
from model.architecture.motionBert_full import DSTformer, ReconstructNet
from model.training.utils.training_helper import *
from model.training.eval_loop import run_quick_eval, run_final_evaluation
from model.training.utils.loss import *
from torch.amp import autocast, GradScaler
import warnings
import pandas as pd
import numpy as np
import math
import os
from dotenv import load_dotenv
load_dotenv(interpolate=True)

warnings.filterwarnings("ignore", category=FutureWarning)

def set_args(config: dict, args_cfg):
    args_cfg.use_angular = config['use_angular']

        # Materialize mutable copies before modifications
    args_cfg.augments = dict(args_cfg.augments)
    args_cfg.augment_args = dict(args_cfg.augment_args)

    # Should I Add Noise
    if not config['with_noise']:
        args_cfg.augments['jittering'] = False
        args_cfg.augments['None'] = True
    else:
        args_cfg.augments['jittering'] = True
        args_cfg.augments['None'] = True

    args_cfg.normalize_time = config['time_normalize']
    args_cfg.normalize_space = config['space_normalize']
    noise_std = config.get('noise_std', 0.05)
    args_cfg.augment_args['jitter_sigma'] = noise_std
    if config.get('group_masking') is not None:
        args_cfg.group_masking = config['group_masking']
    if config.get('random_masking') is not None:
        args_cfg.random_mask = config['random_masking']
    if config.get('random_masking_frames') is not None:
        args_cfg.random_mask_frames = config['random_masking_frames']
    if config.get("span_group_masking", None) is not None:
        args_cfg.span_masking = config['span_group_masking']
    if config.get('len_cycle', None) is not None:
        args_cfg.cycle_len = config['len_cycle']
    if config.get("len_sts_cycle", None) is not None:
        args_cfg.sts_cycle_len = config['len_sts_cycle']
    if config.get('seq_per_person', None) is not None:
        args_cfg.seq_per_person = config['seq_per_person']
    if config.get('limit_users', None) is not None:
        args_cfg.limit_users = config['limit_users']
    if config.get('equal_seq_per_activity', None) is not None:
        args_cfg.equal_seq_per_activity = config['equal_seq_per_activity']

    args_cfg.centroid_type = config.get('centroid_type', "first")
    
    return args_cfg

def train(run_final_eval: bool = True, config: dict = None, sweep=False, only_eval=False) -> dict:
    device = select_device((config or wandb.config).get('device', 'auto'))
    print(f"Using device: {device}")

    if config is None:
        config = wandb.config

    if config.get('use_matmul', None) is not None:
        torch.set_float32_matmul_precision('high')
    args_cfg = copy.deepcopy(args.active_args())
    args_cfg = set_args(config, args_cfg)

    now = datetime.now()
    date_str = now.strftime("%m_%d_%H_%M")
    # this is model dir unless continuing training
    if sweep:
        save_model_dir = os.getenv("SWEEP_MODEL_SAVE_DIR")
    else:
        save_model_dir = os.getenv("MODEL_SAVE_DIR", f'/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/models/motionBert/')

    save_model_dir += f"bert_{wandb.run.id}_{date_str}"

    visualization_config = {
        "animate": True if not config['debug_mode'] else False,
        "draw_skeleton": True if not config['debug_mode'] else False,
        "draw_skeleton_times_an_epoch": 3,
        "draw_skeleton_activity": "tm_3kmh",
        "animate_times_an_epoch": 5,
        "animate_activity": "tm_3kmh"
    }



    # clear memory

    num_epochs = config['epochs']
    size_seq = config['size_seq']
    batch_size = config['batch_size']
    depth = config.get('depth', 5)

    dim_out = 7 if args_cfg.use_angular else 3


    lambda_scale = config['lambda_scale']
    lambda_3d_velocity = config['lambda_3d_velocity']
    lambda_ko_leo_loss = config.get('lambda_ko_leo_loss', 0.0)
    if config['mask_loss']:
        lambda_scale = 0.0
        lambda_3d_velocity = 0.0

    # Micro-batch + accumulate, to hit a target effective batch size without the
    # memory spike of one large batch. Verified (see dev notes) that this gives
    # gradients equivalent to a true large batch to floating-point precision --
    # EXCEPT koleo loss, which compares embeddings *within* a batch and is not
    # accumulation-safe (a micro-batch's koleo loss isn't the same quantity as
    # the full effective batch's). Refuse rather than silently give wrong grads.
    grad_accum_steps = config.get('grad_accum_steps', 1)
    if grad_accum_steps > 1 and lambda_ko_leo_loss > 0:
        raise ValueError(
            "grad_accum_steps > 1 is not valid with lambda_ko_leo_loss > 0: "
            "koleo loss depends on batch composition and cannot be correctly "
            "accumulated across micro-batches."
        )

    modality_dropout_prob = config.get('modality_dropout_prob', 0.0)


    if config['training_type'] == 'continue':
        if config['load_model_epoch'] == -1:
            model_file = get_latest_dir(config['model_file'])
        else:
            model_file = config['model_file']

        model_dir = '/'.join(config['model_file'].split('/')[:-1]) + f'/'
        try:
            loaded_args = args.load_args_from_path(model_dir)
            args_cfg = copy.deepcopy(args.apply_args(loaded_args))

            lambda_quat = config.get('lambda_quat', 0.5) if args_cfg.use_angular else None
        except FileNotFoundError as e:
            print(f"Error loading preprocessing args from {model_dir}: {e}")
            raise e
    
        save_model_dir = model_dir

    lambda_quat = config.get('lambda_quat', 0.5) if args_cfg.use_angular else None
    num_joints = 26
    if args_cfg.remove_noise_joints:
        num_joints = 26
    if args_cfg.motionbert_format:
        num_joints = 17

    # cudnn tuning only applies to the CUDA backend
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # pin_memory speeds up host->CUDA transfers specifically; meaningless (and
    # occasionally noisy) on MPS/CPU
    pin_memory = device.type == 'cuda'

    # Initialize Loaders
    print(f"Seq overlap: {config.get('seq_overlap', 0)}")
    train_dataset, test_dataset, eval_dataset = get_datasets(size_seq=size_seq,
                                                             labels=config['labels'],
                                                             graph_data=False,
                                                             overlap_sequence=config.get('seq_overlap', 0),
                                                             args_cfg=args_cfg,
                                                             data_source=config.get('data_source', 'legacy_csv'))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6,
                              pin_memory=pin_memory,
                              persistent_workers=True)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=6,
                             pin_memory=pin_memory,
                             persistent_workers=True)

    masked_eval_dataset = copy.deepcopy(eval_dataset)
    masked_eval_dataset.override_mask_set = True
    masked_eval_loader = DataLoader(masked_eval_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=6,
                                    pin_memory=pin_memory,
                                    persistent_workers=True)

    print(f"TrainLoader len {len(train_loader)}")
    print(f"EvalLoader len {len(eval_loader)}")
    print(f"MaskedEvalLoader len {len(masked_eval_loader)}")

    print(
        f"per Joint amount: {train_dataset.per_joint_amount} with {num_joints} joints with angular {args_cfg.use_angular}, debug mode {config['debug_mode']}")
    print(
        f"time normalize: {config['time_normalize']}, space normalize: {config['space_normalize']}, with noise: {config['with_noise']}")
    # Define model
    model_backbone = DSTformer(num_joints=num_joints,
                               dim_in=train_dataset.per_joint_amount,
                               maxlen=size_seq,
                               depth=depth,
                               drop_rate=config['dropout_ratio'],
                               use_rope=config['with_rope'],
                               use_flash_attn=config.get('use_flash_attn', False),
                               dim_feat=config.get('dim_feat', 256),
                               dim_rep=config['dim_representation'],
                               num_heads=config['num_heads'],
                               num_sink_tokens=config.get('num_sink_tokens', 0),
                               use_grad_checkpointing=config.get('use_grad_checkpointing', False))

    if config.get('lambda_ko_leo_loss', None) is not None:
        koleo_dim = config.get('koleo_dim', 128)
    else:
        koleo_dim = None

    model = ReconstructNet(model_backbone,
                           dim_in=train_dataset.per_joint_amount,
                           dim_out=dim_out,
                           modality_dropout_prob=modality_dropout_prob,
                           zero_out_masked_joints=args_cfg.zero_out_masked_joints,
                           koleo_dim=koleo_dim)

    model = model.to(device)
    
    # Sink tokens: truncate input to keep memory footprint same (128 frames -> 127 + 1 sink = 128)
    num_sink_tokens = config.get('num_sink_tokens', 0)

    # Initialize Optimizer, Scheduler, Scaler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=config["learning_rate"],
                                  fused=(device.type == 'cuda'),  # fused kernels are CUDA-only
                                  weight_decay=config["adamw_weight_decay"])

    schedular_config = {
        "scheduler_type": config['scheduler_type'],
        "num_epochs": num_epochs,
        "warmup_epochs": config.get('warmup_epochs', None),
        # scheduler.step() only fires at accumulation boundaries now, not every
        # micro-batch -- steps_per_epoch must match or the schedule (e.g. cosine
        # decay/warmup) finishes early or never completes within num_epochs.
        "steps_per_epoch": math.ceil(len(train_loader) / grad_accum_steps),
        "warmup_ratio": config.get('warmup_ratio', None),
        "learning_rate": config.get('learning_rate', None),
    }
    scheduler = get_scheduler(optimizer, schedular_config)

    # GradScaler's loss-scaling only makes sense (and is only reliably
    # supported) on CUDA; disabling it on MPS/CPU makes it a documented no-op
    # rather than a device mismatch crash.
    use_amp_scaler = config['use_scaler'] and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp_scaler)

    # Load a model if needed
    if config['training_type'] == 'continue':
        checkpoint_path = model_file
        if checkpoint_path and os.path.exists(checkpoint_path):
            model, optimizer, scheduler, new_scaler, start_epoch = load_checkpoint(
                model, optimizer, scheduler, checkpoint_path, device, scaler=scaler)
            scaler = new_scaler if new_scaler is not None else scaler
        else:
            print("No checkpoint found or path not specified. Starting from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    # Compile model if enabled (only for single GPU - incompatible with DataParallel)
    if hasattr(torch, 'compile') and config.get('use_compile', True) and config['amount_device'] == 1:
        print(f"Compiling model for performance optimization (backend device: {device.type})...")
        try:
            model = torch.compile(model)  # Significant speedup with compilation
        except Exception as exc:
            # torch.compile's non-CUDA backends (esp. MPS) are less mature; fall
            # back to eager rather than aborting the whole run over a compile error.
            print(f"torch.compile failed on device={device.type}, falling back to eager mode: {exc}")
    elif config['amount_device'] > 1:
        print(f"Multi-GPU mode ({config['amount_device']} GPUs): skipping torch.compile (incompatible with DataParallel)")

    # 2 Device training (CUDA-only concept)
    if config['amount_device'] > 1:
        assert device.type == 'cuda', "amount_device > 1 (DataParallel) requires CUDA"
        model = nn.DataParallel(model, dim=0)

    
    
    wait_flags = {"viz": 0, "anim": 0}
    visualizer = None
    animator = None
    animation_queue = None

    # To save the args only once
    args_flag = 0

    attention_save_dir = os.path.join(save_model_dir, "attention_maps")
    attention_config = {
        "enabled": config.get('visualize_attention_maps', True),
        "output_dir": attention_save_dir,
        "max_per_activity": config.get('attention_samples_per_activity', 1),
        "log_to_wandb": True,
        "wandb_prefix": "eval_attention",
    }


    print('Starting Training Loop')
    for epoch in range(start_epoch, num_epochs):
        if only_eval:
            break
        model.train()
        train_loss = 0.0

        # collect all data in an epoch to calculate metrics at the end
        all_labels = []
        all_activities = []
        all_ids = []

        for batch_idx, skeleton_data in enumerate(train_loader):
            data = skeleton_data['data'].to(device)
            raw_mask = skeleton_data.get('mask')
            model_mask = None
            mask_tensor = None
            if raw_mask is not None:
                mask_tensor = raw_mask.to(device)
                if mask_tensor.dim() == 3:
                    mask_tensor = mask_tensor.unsqueeze(-1)
                model_mask = mask_tensor.any(dim=-1)

            # Truncate last frames to make room for sink tokens (keeps memory footprint same)
            if num_sink_tokens > 0:
                data = data[:, :-num_sink_tokens, :, :]
                if mask_tensor is not None:
                    mask_tensor = mask_tensor[:, :-num_sink_tokens, :, :]
                    model_mask = model_mask[:, :-num_sink_tokens, :]

            # remove the confidence
            original_data = skeleton_data['original'][:, :, :, :3].to(device)
            if num_sink_tokens > 0:
                original_data = original_data[:, :-num_sink_tokens, :, :]
            if args_cfg.use_angular:
                # Quaternion are 4 dimensions starting at 4
                quaternion_data = skeleton_data['original'][:, :, :, 4:].to(device)
                if num_sink_tokens > 0:
                    quaternion_data = quaternion_data[:, :-num_sink_tokens, :, :]

            is_accum_start = batch_idx % grad_accum_steps == 0
            is_accum_boundary = (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(train_loader) - 1
            if is_accum_start:
                optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=(device.type != 'cpu')):
                predicted_pos, embeddings = model(data, mask=model_mask)
                predicted_3d_pos = predicted_pos[:, :, :, :3]

                if args_cfg.use_angular:
                    predicted_3d_quat = predicted_pos[:, :, :, 3:]

                loss_mask = None
                quat_mask = None
                if config['mask_loss'] and mask_tensor is not None:
                    loss_mask = mask_tensor[..., :predicted_3d_pos.shape[-1]].to(predicted_3d_pos.dtype)
                    if args_cfg.use_angular:
                        quat_channels = predicted_3d_quat.shape[-1]
                        quat_mask = mask_tensor[..., predicted_3d_pos.shape[-1]:predicted_3d_pos.shape[-1] + quat_channels]
                        quat_mask = quat_mask.to(predicted_3d_quat.dtype)

                labels = skeleton_data['label_list']
                activities = skeleton_data['activity']
                ids = skeleton_data['id']

                for i, label_list in enumerate(labels):
                    if i >= len(all_labels):
                        all_labels.append([])
                    all_labels[i].extend(label_list.tolist())
                all_activities.extend(activities)
                all_ids.extend(ids)

                loss_3d_pos = loss_mpjpe(predicted_3d_pos, original_data, mask=loss_mask)
                loss_total = loss_3d_pos
                loss_3d_scale = None
                loss_3d_velocity = None
                loss_4d_quat = None

                if not config['mask_loss']:
                    loss_3d_scale = n_mpjpe(predicted_3d_pos, original_data, mask=loss_mask)
                    velocity_pred = predicted_3d_pos
                    velocity_target = original_data
                    if loss_mask is not None:
                        velocity_pred = velocity_pred * loss_mask
                        velocity_target = velocity_target * loss_mask
                    loss_3d_velocity = loss_velocity(velocity_pred, velocity_target)
                    loss_total = loss_total + lambda_scale * loss_3d_scale + \
                                 lambda_3d_velocity * loss_3d_velocity

                if args_cfg.use_angular:
                    loss_4d_quat = loss_quat_geodesic(predicted_3d_quat, quaternion_data, mask=quat_mask)
                    loss_total = loss_total + lambda_quat * loss_4d_quat

                # Koleo loss per activity class
                loss_koleo = None
                if lambda_ko_leo_loss > 0:
                    koleo_emb = embeddings.get('koleo_embeddings')
                    if koleo_emb is not None:
                        loss_koleo = koleo_loss_per_class(koleo_emb, activities)
                        loss_total = loss_total + lambda_ko_leo_loss * loss_koleo

            current_lr = optimizer.param_groups[0]['lr']
            # log granular loss metrics and learning rate for monitoring
            log_payload = {
                "loss": loss_total.item(),
                "epoch": epoch,
                "batch": batch_idx,
                "loss_3d_pos": loss_3d_pos.item(),
                "learning_rate": current_lr,
            }
            if loss_3d_scale is not None:
                log_payload["loss_3d_scale"] = loss_3d_scale.item()
            if loss_3d_velocity is not None:
                log_payload["loss_3d_velocity"] = loss_3d_velocity.item()
            if loss_4d_quat is not None:
                log_payload["loss_4d_quat"] = loss_4d_quat.item()
                log_payload['loss_quat_degrees'] = loss_4d_quat.item() * 180.0 / math.pi
            if loss_koleo is not None:
                log_payload["loss_koleo"] = loss_koleo.item()

            wandb.log(log_payload)
            # Scale for accumulation (average, not sum, across micro-batches), then
            # backward every step but only step optimizer/scheduler at the boundary.
            loss_for_backward = loss_total / grad_accum_steps
            if use_amp_scaler:
                scaler.scale(loss_for_backward).backward()
                if is_accum_boundary:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                loss_for_backward.backward()
                if is_accum_boundary:
                    optimizer.step()
            train_loss += loss_total.item()
            if is_accum_boundary:
                scheduler.step()

            # Log sink attention mass periodically (every 100 batches)
            sink_log_freq = config.get('sink_attention_log_freq', 100)
            if num_sink_tokens > 0 and batch_idx % sink_log_freq == 0:
                # Run a single forward pass with attention capture to measure sink mass
                with torch.no_grad():
                    model.eval()
                    # Get backbone (unwrap DataParallel/compile if needed)
                    backbone = model
                    if hasattr(backbone, '_orig_mod'):
                        backbone = backbone._orig_mod
                    if hasattr(backbone, 'module'):
                        backbone = backbone.module
                    backbone = backbone.model_backbone
                    
                    # Forward with attention capture on first sample
                    sample_data = data[:1]
                    backbone._set_attention_recording(True)
                    _ = model(sample_data, mask=None)
                    sink_stats = backbone.get_sink_attention_stats()
                    backbone._set_attention_recording(False)
                    
                    if sink_stats:
                        wandb.log({
                            "attention/sink_mass_mean": sink_stats['sink_mass_mean'],
                            "attention/sink_mass_max": sink_stats['sink_mass_max'],
                            "attention/sink_mass_median": sink_stats['sink_mass_median'],
                            "attention/sink_mass_p80": sink_stats['sink_mass_p80'],
                            "attention/sink_mass_std": sink_stats['sink_mass_std'],
                            "epoch": epoch,
                            "batch": batch_idx,
                        })
                    model.train()

            # Skeleton visualization disabled in this release

            if config['debug_mode'] and batch_idx >= 10:
                print("Debug mode active, breaking after 500 batches.")
                break

        if config['debug_mode'] and epoch >= 0:
            print("Debug mode active, breaking after 1 epochs.")
            quick_metrics = run_quick_eval(
                model=model,
                eval_loader=eval_loader,
                device=device,
                epoch=epoch,
                log_to_wandb=True,
                attention_config=attention_config,
                debug_mode=config['debug_mode'],
                run_labels=['age', 'hr_bpm', 'bmi', "Anxiety", "Depression"],
                label_names=config['labels'],
                task_types=config['task_types'],
                num_sink_tokens=num_sink_tokens,
            )
            if 'eval_loss_3d_reconstruction' in quick_metrics and 'eval_loss_3d_scale' in quick_metrics:
                print(
                    f"Quick eval epoch {epoch}: "
                    f"reconstruction_loss={quick_metrics['eval_loss_3d_reconstruction']:.4f}, "
                    f"scale_loss={quick_metrics['eval_loss_3d_scale']:.4f}"
                )
            break

        
        print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {train_loss:.4f}")
        if epoch % 1 == 0:
            if args_flag == 0:
                os.makedirs(save_model_dir, exist_ok=True)
                args.save_args_to_path(args_cfg, save_model_dir)
                with open(f'{save_model_dir}/model_args.txt', 'a') as f:
                    model_args = {"model": model.__class__.__name__,
                                  "dim_in": train_dataset.per_joint_amount,
                                  "dim_out": 7 if train_dataset.per_joint_amount > 4 else 3,
                                  "dropout_ratio": config['dropout_ratio'],
                                  "batch_size": batch_size,
                                  "learning_rate": config['learning_rate'],
                                  "adamw_weight_decay": config['adamw_weight_decay'],
                                  "scheduler_type": config['scheduler_type'],
                                  "num_epochs": num_epochs,
                                  "warmup_epochs": config.get("warmup_epochs", None),
                                  "warmup_ratio": config.get('warmup_ratio', None),
                                  "size_seq": size_seq,
                                  "num_joints": num_joints,
                                  "lambda_scale": lambda_scale,
                                  "lambda_3d_velocity": lambda_3d_velocity,
                                  "mask_loss": config['mask_loss'],
                                  "with_noise": config['with_noise'],
                                  "with_rope": config['with_rope'],
                                  "dim_representation": config['dim_representation'],
                                  "num_heads": config['num_heads'],
                                  }
                    f.write(f'{str(model_args)}')
                args_flag = 1
            # Save the model checkpoint also the optimizer state and scheduler state
            model_to_save = model
            if hasattr(model_to_save, "_orig_mod"): model_to_save = model_to_save._orig_mod  # unwrap torch.compile
            if hasattr(model_to_save, "module"):    model_to_save = model_to_save.module  # unwrap DataParallel/DDP

            torch.save({
                'model': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
            }, os.path.join(save_model_dir, f'epoch_{epoch}.pth'))

        # Quick eval keeps an eye on reconstruction and embedding health each epoch
        if epoch % 1 == 0:
            quick_metrics = run_quick_eval(
                model=model,
                eval_loader=eval_loader,
                device=device,
                epoch=epoch,
                log_to_wandb=True,
                attention_config=attention_config,
                debug_mode=config['debug_mode'],
                run_labels=['age', 'hr_bpm', 'bmi', "Anxiety", "Depression"],
                label_names=config['labels'],
                task_types=config['task_types'],
                masked_eval_loader=masked_eval_loader,
                num_sink_tokens=num_sink_tokens,
            )

    # Skeleton visualization disabled in this release

    # Run the comprehensive embedding evaluation once training finishes
    final_metrics = None
    if run_final_eval:
        args_cfg.augment_eval = False
        # remove masking and augmentation for final evaluation
        args_cfg.random_mask = 0
        args_cfg.random_mask_frames = 0
        args_cfg.group_masking = 0
        args_cfg.augments = {
            'None': True,
            'rotation': False,
            'scaling': False,
            'translation': False,
            'jittering': False,
            'mirroring': False,
        }
        train_dataset, test_dataset, eval_dataset = get_datasets(size_seq=size_seq,
                                                                 labels=config['labels'],
                                                                 graph_data=False,
                                                                 overlap_sequence=0,
                                                                 args_cfg=args_cfg,
                                                                 data_source=config.get('data_source', 'legacy_csv'))

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=3,
                                  pin_memory=pin_memory,
                                  persistent_workers=True)

        eval_loader = DataLoader(eval_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=3,
                                 pin_memory=pin_memory,
                                 persistent_workers=True)

        test_loader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=3,
                                 pin_memory=pin_memory,
                                 persistent_workers=True)

        final_metrics = run_final_evaluation(
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            test_loader=test_loader,
            device=device,
            label_names=config['labels'],
            task_types=config['task_types'],
            save_path=save_model_dir,
            umap_view=True,
            debug_mode=config['debug_mode'],
            track_progress=True,
            num_sink_tokens=num_sink_tokens,
        )

    
    return {
        'model': model,
        'train_loader': train_loader,
        'eval_loader': eval_loader,
        'device': device,
        'config': config,
        'save_model_dir': save_model_dir,
        'final_metrics': final_metrics,
    }
if __name__ == "__main__":
    train()
