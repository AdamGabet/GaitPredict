import numpy as np
import pandas as pd
from model.preprocessing import args, joints_file
import matplotlib.pyplot as plt

# Build a lookup so we know where each joint ends up after noise joints are removed
_KEPT_JOINTS = [j for j in range(joints_file.k4abt_joints.len) if j not in joints_file.noise_joints]
_JOINT_INDEX_MAP = {joint: idx for idx, joint in enumerate(_KEPT_JOINTS)}

# Pre-cached indices for the joints we care about
HIP_RIGHT_IDX = int(_JOINT_INDEX_MAP[joints_file.k4abt_joints.HIP_RIGHT])
KNEE_RIGHT_IDX = int(_JOINT_INDEX_MAP[joints_file.k4abt_joints.KNEE_RIGHT])
ANKLE_RIGHT_IDX = int(_JOINT_INDEX_MAP[joints_file.k4abt_joints.ANKLE_RIGHT])
FOOT_RIGHT_IDX = int(_JOINT_INDEX_MAP[joints_file.k4abt_joints.FOOT_RIGHT])

# same as above but for left side for calculting average and finding outliers
HIP_LEFT_IDX = int(_JOINT_INDEX_MAP[joints_file.k4abt_joints.HIP_LEFT])
KNEE_LEFT_IDX = int(_JOINT_INDEX_MAP[joints_file.k4abt_joints.KNEE_LEFT])
ANKLE_LEFT_IDX = int(_JOINT_INDEX_MAP[joints_file.k4abt_joints.ANKLE_LEFT])


def _get_np_from_df(skel_df):
    skel_df = skel_df.drop(columns=[col for col in skel_df.columns if '_c' in col or 'angle' in col])
    np_coords_front = skel_df.iloc[:, 2:].values.reshape(len(skel_df), -1, 3)  # 4 for confidence
    return np_coords_front


def normalize_seq_fast(array_3d: np.ndarray, global_centroid: bool = True, front: bool = True) -> np.ndarray:
    """
    Normalize a 3D array of joint coordinates.

    Args:
        array_3d: numpy array of shape (frames, joints, 3)
        global_centroid: if True, use first frame centroid for all frames
        front: if True, store centroid for later use

    Returns:
        numpy array of normalized joint coordinates
    """
    coords = array_3d.copy()

    # remove noise joints
    coords = coords[:, _KEPT_JOINTS, :]

    # Define indices for centroid joints
    centroid_indices = [joint for joint in range(joints_file.k4abt_joints.len)
                        if joint in joints_file.joints_centroid]

    # Calculate centroid - Global centroid means normalize using only the first frame
    if not global_centroid:
        centroid = np.mean(coords[:, centroid_indices], axis=1, keepdims=True)
    else:
        centroid = np.mean(coords[0, centroid_indices], axis=0, keepdims=True)

    # Subtract centroid
    coords_centered = coords - centroid

    # Calculate furthest distance
    furthest_distance = np.max(np.sqrt(np.sum(coords_centered ** 2, axis=2)), axis=1, keepdims=True)

    # Normalize
    normalized_coords = coords_centered / furthest_distance[:, np.newaxis]

    # Replace NaNs with 0
    normalized_coords = np.nan_to_num(normalized_coords)

    return normalized_coords


def rescale_skeleton(coords: np.ndarray, idx_a: int, idx_b: int, target_length: float = None) -> np.ndarray:
    """
    Rescale a single skeleton to a target scale based on the distance between two reference joints.

    Args:
        coords: numpy array of shape (frames, joints, channels)
        idx_a: index of first reference joint
        idx_b: index of second reference joint
        target_length: target distance between joints (if None, returns unscaled)

    Returns:
        Scaled coordinates
    """
    vector = coords[:, idx_a, :3] - coords[:, idx_b, :3]
    # Compute distances more efficiently
    distances_sq = np.sum(vector * vector, axis=1)
    # Filter in one step
    valid_mask = np.isfinite(distances_sq) & (distances_sq > 0)

    if np.any(valid_mask):
        mean_length = np.sqrt(np.mean(distances_sq[valid_mask]))
    else:
        mean_length = 1.0

    # If no target length provided, just return the original
    if target_length is None:
        target_length = mean_length

    # Apply scaling
    scale = target_length / mean_length if mean_length > 0 else 1.0
    scaled_coords = coords.copy()
    scaled_coords[:, :, :3] *= scale

    return scaled_coords


def compute_knee_flexion_both_sides(normalized_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute knee flexion angles for both sides simultaneously."""
    # Extract all joints at once
    hip_right = normalized_coords[:, HIP_RIGHT_IDX, :3]
    knee_right = normalized_coords[:, KNEE_RIGHT_IDX, :3]
    ankle_right = normalized_coords[:, ANKLE_RIGHT_IDX, :3]

    hip_left = normalized_coords[:, HIP_LEFT_IDX, :3]
    knee_left = normalized_coords[:, KNEE_LEFT_IDX, :3]
    ankle_left = normalized_coords[:, ANKLE_LEFT_IDX, :3]

    def _compute_angles(hip, knee, ankle):
        thigh_vec = hip - knee
        shank_vec = ankle - knee

        dot_products = np.einsum("ij,ij->i", thigh_vec, shank_vec)
        thigh_norm_sq = np.sum(thigh_vec * thigh_vec, axis=1)
        shank_norm_sq = np.sum(shank_vec * shank_vec, axis=1)
        norm_products_sq = thigh_norm_sq * shank_norm_sq

        valid_mask = norm_products_sq > 0
        cos_angles = np.zeros_like(dot_products)
        norm_products = np.sqrt(norm_products_sq[valid_mask])
        cos_angles[valid_mask] = np.clip(dot_products[valid_mask] / norm_products, -1.0, 1.0)

        angles = np.degrees(np.arccos(cos_angles))
        angles[~valid_mask] = 0.0
        return angles

    right_angles = _compute_angles(hip_right, knee_right, ankle_right)
    left_angles = _compute_angles(hip_left, knee_left, ankle_left)

    return right_angles, left_angles


from scipy.signal import argrelextrema


def detect_peaks_custom(signal: np.ndarray, k: int = 5, start_K=5, mode="lows") -> np.ndarray:
    """Detect lows/highs using scipy's optimized peak detection."""
    if signal.size < 2 * k + 1:
        return np.array([], dtype=int)

    # Use scipy's argrelextrema which is highly optimized
    if mode == "lows":
        peaks = argrelextrema(signal, np.less, order=k)[0]
    else:  # highs
        peaks = argrelextrema(signal, np.greater, order=k)[0]

    # Filter by start_K boundary
    peaks = peaks[(peaks >= start_K) & (peaks < signal.size - start_K)]

    return peaks


def build_cycle_phase(num_frames: int, landmarks: np.ndarray) -> np.ndarray:
    """Assign a normalized gait phase (0-1) to each frame based on landmarks."""
    phase = np.full(num_frames, np.nan, dtype=float)
    if landmarks.size < 2:
        return phase

    for start, end in zip(landmarks[:-1], landmarks[1:]):
        frame_count = end - start + 1
        phase[start: end + 1] = np.linspace(0.0, 1.0, num=frame_count)

    return phase


from scipy.interpolate import interp1d


def resample_coordinates_with_landmarks(
        coords: np.ndarray, landmarks: np.ndarray, target_points: int = 100
) -> np.ndarray:
    """Resample the full coordinate array for each cycle."""
    if landmarks.size < 2:
        return np.empty((0, target_points, coords.shape[1], coords.shape[2]))

    cycles = []
    target_positions = np.linspace(0.0, 1.0, num=target_points)

    for start, end in zip(landmarks[:-1], landmarks[1:]):
        segment = coords[start: end + 1]
        num_frames = segment.shape[0]
        frame_positions = np.linspace(0.0, 1.0, num=num_frames)

        # scipy's interp1d can handle multiple dimensions efficiently
        # axis=0 means interpolate along the frame dimension
        interpolator = interp1d(
            frame_positions,
            segment,
            axis=0,
            kind='linear',
            assume_sorted=True,
            copy=False
        )

        resampled = interpolator(target_positions)
        cycles.append(resampled)

    return np.stack(cycles, axis=0)


def turn_into_cycle(coords: np.ndarray, activity: str, num_frames: int = 64) -> tuple:
    """Turn coordinates into gait cycles with automatic cleaning."""
    knee_angles_right, knee_angles_left = compute_knee_flexion_both_sides(coords)

    # Set parameters based on activity
    if activity == "sit_to_stand":
        k, start_K, mode = 30, 10, "highs"
    else:
        k, start_K, mode = 10, 8, "lows"

    # Initial detection
    landmarks_right = detect_peaks_custom(knee_angles_right, k=k, start_K=start_K, mode=mode)
    landmarks_left = detect_peaks_custom(knee_angles_left, k=k, start_K=start_K, mode=mode)

    # Clean and fix landmarks
    landmarks_right = clean_and_fix_landmarks(
        knee_angles_right, landmarks_right, k=k, start_K=start_K, mode=mode, redetect_k_factor=0.7
    )
    landmarks_left = clean_and_fix_landmarks(
        knee_angles_left, landmarks_left, k=k, start_K=start_K, mode=mode, redetect_k_factor=0.7
    )

    # Resample with cleaned landmarks
    cycles_right = resample_coordinates_with_landmarks(coords, landmarks_right, target_points=num_frames)
    cycles_left = resample_coordinates_with_landmarks(coords, landmarks_left, target_points=num_frames)

    return cycles_right, cycles_left


def clean_and_fix_landmarks(signal: np.ndarray,
                            landmarks: np.ndarray,
                            k: int = 10,
                            start_K: int = 8,
                            mode: str = "lows",
                            mad_threshold: float = 3.5,
                            redetect_k_factor: float = 0.5,
                            max_iterations: int = 3) -> np.ndarray:
    """
    Clean landmarks by removing too-short cycles and re-detecting in too-long cycles.
    Iterates until no more outliers or max_iterations reached.

    Args:
        signal: The signal used for peak detection (e.g., knee angles)
        landmarks: Detected landmarks to clean
        k: Parameter for peak detection
        start_K: Parameter for peak detection
        mode: 'lows' or 'highs' for peak detection
        mad_threshold: MAD threshold for outlier detection (default 3.5)
        redetect_k_factor: Factor to multiply k by for re-detection (default 0.5 = half)
        max_iterations: Maximum number of cleaning iterations (default 3)

    Returns:
        Cleaned landmarks array
    """
    if landmarks.size < 3:
        return landmarks

    current_landmarks = landmarks.copy()

    for iteration in range(max_iterations):
        # Calculate cycle lengths and detect outliers using MAD
        cycle_lengths = np.diff(current_landmarks)
        median_length = np.median(cycle_lengths)
        mad = np.median(np.abs(cycle_lengths - median_length))

        if mad == 0:
            break  # All cycles are identical, stop

        # Modified Z-scores
        modified_z_scores = 0.6745 * (cycle_lengths - median_length) / mad
        is_outlier = np.abs(modified_z_scores) > mad_threshold

        # If no outliers, we're done
        if not np.any(is_outlier):
            break

        # Categorize outliers
        too_short = (cycle_lengths < 0.5 * median_length) & is_outlier
        too_long = (cycle_lengths > 1.7 * median_length) & is_outlier

        # Step 1: Remove landmarks creating too-short cycles
        landmarks_to_remove = []
        for i in np.where(too_short)[0]:
            if 0 < (i + 1) < len(current_landmarks):
                landmarks_to_remove.append(i + 1)

        landmarks_to_remove = list(set(landmarks_to_remove))
        keep_mask = np.ones(len(current_landmarks), dtype=bool)
        keep_mask[landmarks_to_remove] = False
        cleaned_landmarks = current_landmarks[keep_mask]

        # Step 2: Re-detect in too-long cycles with LOWER k for higher sensitivity
        redetect_k = max(3, int(k * redetect_k_factor))
        redetect_start_K = max(2, int(start_K * redetect_k_factor))

        new_landmarks = []

        for i in np.where(too_long)[0]:
            # Skip if this index is out of bounds after removal
            if i >= len(cleaned_landmarks) - 1:
                continue

            # Define region: 3 landmarks before to 3 landmarks after
            start_landmark_idx = max(0, i - 2)
            end_landmark_idx = min(len(cleaned_landmarks) - 1, i + 4)

            region_start = cleaned_landmarks[start_landmark_idx]
            region_end = cleaned_landmarks[end_landmark_idx]

            # Extract signal region
            signal_region = signal[region_start:region_end + 1]

            # Re-detect with LOWER k (more sensitive)
            local_landmarks = detect_peaks_custom(
                signal_region,
                k=redetect_k,
                start_K=redetect_start_K,
                mode=mode
            )

            # Convert local landmarks to global frame indices
            global_landmarks = local_landmarks + region_start

            # Keep only landmarks that fall in the problematic cycle
            cycle_start = cleaned_landmarks[i]
            cycle_end = cleaned_landmarks[i + 1]
            in_cycle = (global_landmarks > cycle_start) & (global_landmarks < cycle_end)
            new_in_cycle = global_landmarks[in_cycle]

            new_landmarks.extend(new_in_cycle.tolist())

        # Step 3: Merge original cleaned landmarks with newly detected ones
        if new_landmarks:
            all_landmarks = np.concatenate([cleaned_landmarks, new_landmarks])
            all_landmarks = np.unique(all_landmarks)
            all_landmarks = np.sort(all_landmarks)
        else:
            all_landmarks = cleaned_landmarks

        # Check if anything changed
        if len(all_landmarks) == len(current_landmarks) and np.all(all_landmarks == current_landmarks):
            break  # Converged, no changes

        current_landmarks = all_landmarks

    return current_landmarks


def normalize_cycle_and_space(coords: np.ndarray, activity: str, normalize_space: bool = True, normalize_time: bool = True,
                              num_frames: int = 64, skip_n_cycles=0, debug=False, sit_to_stand_frames=128) -> np.ndarray:

    if normalize_space:
        coords = rescale_skeleton(coords, HIP_RIGHT_IDX, HIP_LEFT_IDX)

    if normalize_time and "romberg" in activity:
        # interpolate the rombergs data to 2x the original length
        coords = interp1d(np.arange(coords.shape[0]), coords, axis=0, kind='linear', assume_sorted=True, copy=False)(np.linspace(0, coords.shape[0] - 1, 2 * coords.shape[0]))

    if normalize_time and "romberg" not in activity:
        if "sit_to_stand" in activity:
            num_frames = sit_to_stand_frames
        coords, coords_left = turn_into_cycle(coords, activity, num_frames=num_frames)
        num_cycles = coords.shape[0]
        num_cycles2 = coords_left.shape[0]

        # skip initial cycles if needed
        if skip_n_cycles > 0 and "sit_to_stand" not in activity:
            coords = coords[skip_n_cycles:, :, :]
            coords_left = coords_left[skip_n_cycles:, :, :]
            num_cycles -= skip_n_cycles
            num_cycles2 -= skip_n_cycles

        # concatenate coords into a continuous sequence by reshaping frames
        coords = coords.reshape(-1, coords.shape[2], coords.shape[3])
        coords_left = coords_left.reshape(-1, coords_left.shape[2], coords_left.shape[3])
        if debug:
            plot_both_views(coords, coords_left, num_frames, num_cycles, num_cycles2)

    return coords



#plotting for debug

def plot_both_views(coords_continuous, coords2_continuous, frames_per_cycle, num_cycles1, num_cycles2):
    """
    Create a figure with both time-domain and cycle-overlay views.
    """
    knee_angles_right, knee_angles_left = compute_knee_flexion_both_sides(coords_continuous)
    knee_angles_right2, knee_angles_left2 = compute_knee_flexion_both_sides(coords2_continuous)
    knee_cycles1 = knee_angles_right.reshape(num_cycles1, frames_per_cycle)
    knee_cycles2 = knee_angles_left.reshape(num_cycles1, frames_per_cycle)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Normalized cycle axis (0 to 1)
    cycle_axis = np.linspace(0, 1, frames_per_cycle)

    colors = plt.cm.tab10.colors

    # ----- Subject 1 -----
    # Time domain (left column, top)
    axes[0, 0].plot(knee_angles_right, color=colors[0],
                    marker='o', markersize=2, markeredgecolor='red',
                    markerfacecolor='red', label='Subject 1')
    # Add vertical lines at cycle boundaries
    for i in range(1, knee_cycles1.shape[0]):
        axes[0, 0].axvline(i * frames_per_cycle, color=colors[0], alpha=0.08)
    axes[0, 0].set_ylabel('Knee Flexion (deg)')
    axes[0, 0].set_xlabel('Normalized Frame')
    axes[0, 0].set_title('Subject 1 - Time Domain')
    axes[0, 0].legend()

    # Cycle overlay (right column, top)
    for cycle_idx in range(knee_cycles1.shape[0]):
        axes[0, 1].plot(cycle_axis, knee_cycles1[cycle_idx],
                        color=colors[0], alpha=0.6,
                        marker='o', markersize=2,
                        markeredgecolor='red', markerfacecolor='red')
    # Add mean cycle
    mean_cycle1 = np.mean(knee_cycles1, axis=0)
    axes[0, 1].plot(cycle_axis, mean_cycle1,
                    color='black', linewidth=3,
                    label=f'Mean (n={knee_cycles1.shape[0]} cycles)',
                    linestyle='--', zorder=10)
    axes[0, 1].set_ylabel('Knee Flexion (deg)')
    axes[0, 1].set_xlabel('Gait Cycle (0-1)')
    axes[0, 1].set_title('Subject 1 - Cycle-Normalized (All Cycles Overlaid)')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 1)

    # ----- Subject 2 -----
    # Time domain (left column, bottom)
    axes[1, 0].plot(knee_angles_right2, color=colors[1],
                    marker='o', markersize=2, markeredgecolor='red',
                    markerfacecolor='red', label='Subject 2')
    # Add vertical lines at cycle boundaries
    for i in range(1, knee_cycles2.shape[0]):
        axes[1, 0].axvline(i * frames_per_cycle, color=colors[1], alpha=0.08)
    axes[1, 0].set_ylabel('Knee Flexion (deg)')
    axes[1, 0].set_xlabel('Normalized Frame')
    axes[1, 0].set_title('Subject 2 - Time Domain')
    axes[1, 0].legend()

    # Cycle overlay (right column, bottom)
    for cycle_idx in range(knee_cycles2.shape[0]):
        axes[1, 1].plot(cycle_axis, knee_cycles2[cycle_idx],
                        color=colors[1], alpha=0.6,
                        marker='o', markersize=2,
                        markeredgecolor='red', markerfacecolor='red')
    # Add mean cycle
    mean_cycle2 = np.mean(knee_cycles2, axis=0)
    axes[1, 1].plot(cycle_axis, mean_cycle2,
                    color='black', linewidth=3,
                    label=f'Mean (n={knee_cycles2.shape[0]} cycles)',
                    linestyle='--', zorder=10)
    axes[1, 1].set_ylabel('Knee Flexion (deg)')
    axes[1, 1].set_xlabel('Gait Cycle (0-1)')
    axes[1, 1].set_title('Subject 2 - Cycle-Normalized (All Cycles Overlaid)')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig('knee_angles_comparison.png', dpi=150, bbox_inches='tight')
    print("Figure saved as 'knee_angles_comparison.png'")
    plt.show()

