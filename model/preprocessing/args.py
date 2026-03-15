import json
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class PreprocessingArgs:
    """Container that holds preprocessing parameters for a single sweep or experiment."""

    # Sequence length configuration per activity
    videos_lens: Dict[str, float] = field(default_factory=lambda: {
        'self_selected_gait_speed': 1800.0,
        'tm_3kmh': 5400.0,
        'romberg_open': 900.0,
        'apose': 300.0,
        'sit_to_stand': 900.0,
        'romberg_closed': 900.0,
        'stationary_walk': 1800.0,
    })


    # Augmentation toggles and parameters
    augments: Dict[str, bool] = field(default_factory=lambda: {
        'None': True,
        'rotation': False,
        'scaling': False,
        'translation': False,
        'jittering': True,
        'mirroring': False,
        'temporal_downsample_slicing0': False,
        'temporal_downsample_slicing1': False,
        'temporal_downsample_interpole': False,
        'temporal_upsample': False,
    })
    augment_eval: bool = False
    cutout: bool = False
    shuffle_activities: bool = False
    augment_args: Dict[str, float] = field(default_factory=lambda: {
        'rotation_max_angle': 2 * np.pi,
        'scaling_max': 1.3,
        'scaling_min': 0.7,
        'translation_max': 0.01,
        'jitter_sigma': 0.05,
        'down_frame_size': 100,
        'up_frame_size': 100,
        'skip': 2,  # frames step
    })


    # Masking configuration
    random_mask: float = 0.15
    random_mask_frames: float = 0.1
    group_masking: float = 2  # 0-1 float, 0 means no masking, 1 means full masking
    span_masking: int = 32  # expands masked frames by up to this many steps on each side (0 disables)

    specific_joint_keep_mask: Optional[List[int]] = None  # pass a list of all joints you want to keep

    zero_out_masked_joints: bool = True  # if True, zero out the masked joints in the data and then we dont need the mask in the model

    normalize_time: bool = False  # normalize by gait cycle time
    normalize_space: bool = False  # normalize by subject height
    cycle_len: int = 128  # number of frames in one gait cycle for time normalization
    sit_to_stand_frames: int = 128  # number of frames to use for sit to stand sequences
    skip_n_cycles: int = 0  # skip n cycles from the beginning of the sequence

    # Viewpoint and sensor fusion controls
    two_viewpoints: bool = False
    side_views: bool = False
    use_kalman: bool = False
    kalman_front: bool = False
    kalman_side: bool = False
    centroid_type: str = 'second' # first, second, or both


    # Signal formatting and feature selection
    motionbert_format: bool = False
    remove_noise_joints: bool = True  # if True, remove the noise joints
    use_confidence: bool = True
    use_angular: bool = False
    overlap_sequence: int = 0  # if > 0, sequences can overlap for continuity
    global_centroid: bool = False  # normalize with the centroid of only the first frame
    distances: bool = True  # provide the distances between the joints
    add_features: bool = False

    # Data handling preferences
    load_to_ram: bool = False
    use_memmap: bool = True
    memmap_dir: Optional[str] = "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/memmap_cache/"
    save_as_float16: bool = False  # if True, save preprocessed data as float16 to save disk space

    # Dataset filtering options
    seq_per_person: int = 2 # how many sequences to keep per person -1 means all
    equal_seq_per_activity: bool = True # if True, keep the same number of sequences per activity by duplicating the sequences
    limit_users: int = -1  # limit to this many users -1 means no limit
    exclude_list: List[str] = field(default_factory=lambda: [
        'apose',
        'romberg_open',
    ])

    def update_from_config(self, config: Dict) -> 'PreprocessingArgs':
        """Update parameters from a training config dictionary.
        
        This allows you to set parameters in one place (e.g., training config)
        and have them automatically applied to the preprocessing args.
        
        Args:
            config: Dictionary containing parameter updates
            
        Returns:
            Updated PreprocessingArgs instance
        """
        # Handle use_angular parameter
        if 'use_angular' in config:
            self.use_angular = config['use_angular']
        
        # Handle noise parameters
        if 'with_noise' in config:
            if not config['with_noise']:
                self.augments = dict(self.augments)  # Make mutable copy
                self.augments['jittering'] = False
                self.augments['None'] = True
            else:
                self.augments = dict(self.augments)  # Make mutable copy
                self.augments['jittering'] = True
                self.augments['None'] = True
        
        # Handle noise_std parameter
        if 'noise_std' in config and config['noise_std'] is not None:
            self.augment_args = dict(self.augment_args)  # Make mutable copy
            self.augment_args['jitter_sigma'] = config['noise_std']
        
        # Handle mask_span_ratio parameter - Not Applied
        if 'mask_span_ratio' in config and config['mask_span_ratio'] is not None and False:
            self.random_mask = float(config['mask_span_ratio'])
            self.random_mask_frames = float(config['mask_span_ratio']) / 2.0
            # Note: span_frames calculation would need size_seq, so we'll handle this separately
        
        return self

    def get_num_joints(self) -> int:
        """Calculate the number of joints based on current configuration."""
        if self.remove_noise_joints:
            return 26
        elif self.motionbert_format:
            return 17
        else:
            return 32


# Default instance keeps backwards compatibility for existing imports.
_DEFAULT_ARGS = PreprocessingArgs()
_ARGS_FILENAME = 'preprocessing_args.json'


def default_args() -> PreprocessingArgs:
    """Return a fresh copy of the default preprocessing arguments."""

    return PreprocessingArgs(**asdict(_DEFAULT_ARGS))


def active_args() -> PreprocessingArgs:
    """Expose the shared default instance for callers still relying on module globals."""

    return _DEFAULT_ARGS


def save_args_to_path(args_cfg: PreprocessingArgs,
                      model_path: Union[str, Path],
                      filename: str = _ARGS_FILENAME) -> Path:
    """Persist a preprocessing args configuration next to a model checkpoint."""

    target_dir = Path(model_path)
    if target_dir.suffix:  # if a file path is provided, use its directory
        target_dir = target_dir.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    target_path.write_text(json.dumps(asdict(args_cfg), indent=2, sort_keys=True))
    return target_path


def load_args_from_path(model_path: Union[str, Path], filename: str = _ARGS_FILENAME) -> PreprocessingArgs:
    """Load a preprocessing args configuration from a model directory or file."""

    candidate = Path(model_path)
    if candidate.is_dir():
        candidate = candidate / filename
    elif candidate.suffix:
        candidate = candidate.parent / filename

    with candidate.open('r') as fh:
        loaded = json.load(fh)

    return PreprocessingArgs(**loaded)


    
def apply_args(args_cfg: PreprocessingArgs) -> PreprocessingArgs:
    """Set the provided configuration as the active module-level state.
    
    Note: This function is deprecated. Use PreprocessingArgs instances directly
    instead of relying on module-level globals.
    """
    global _DEFAULT_ARGS
    _DEFAULT_ARGS = args_cfg
    return _DEFAULT_ARGS



if __name__ == '__main__':
    pass
