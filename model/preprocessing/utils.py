import numpy as np
import pandas as pd
import torch
import model.preprocessing.joints_file as joints_file
import os
import random
from scipy.ndimage import median_filter

def add_features(positions):
    # Assuming input shape is (F, 32, 3)
    F = positions.shape[0]  # number of frames
    num_points = positions.shape[1]  # number of 3D points
    # Initialize arrays for speed and acceleration
    speed = np.zeros((F, num_points))
    acceleration = np.zeros((F, num_points))

    # Calculate speed (velocity magnitude)
    # Speed at frame i is the distance between positions at frame i+1 and i
    velocity = np.zeros_like(positions)
    velocity[:-1] = positions[1:] - positions[:-1]
    speed[:-1] = np.linalg.norm(velocity[:-1], axis=2)
    # Copy last frame speed from second-to-last frame
    speed[-1] = speed[-2]

    # Calculate acceleration
    # Acceleration at frame i is the difference between speeds at frame i+1 and i
    acceleration[:-1] = speed[1:] - speed[:-1]
    # Copy last frame acceleration from second-to-last frame
    acceleration[-1] = acceleration[-2]

    # concat speed and acceleration and indices
    result = np.zeros((F, num_points * 3))  # 3 = 1 (indices) + 1(speed) + 1(acceleration)

    indices = np.array([np.arange(num_points)] * F)
    for i in range(num_points):
        result[:, i * 3:(i + 1) * 3] = np.column_stack([
            indices[:, i],
            speed[:, i],
            acceleration[:, i]
        ])
    result = result.reshape(F, num_points, 3)

    positions_features = np.concatenate([positions, result], axis=2)

    return positions_features

def calculate_distances(np_data, edge_index):
    """
    Calculate the average distances and angles between the joints for each edge in the edge tensor
    :param np_data: numpy array of shape (3, F, 32)
    :return: tuple of two lists, each containing tensors of shape edge_index.shape[1] or None
    """
    distance_list = []
    angle_list = []

    for skel in np_data:
        if skel is not None:
            # Reshape from (3, F, 32) to (F, 32, 3)
            skel = np.transpose(skel, (1, 2, 0))

            # Calculate all distances and angles at once
            start_coords = skel[:, edge_index[0]]
            end_coords = skel[:, edge_index[1]]

            # Distances
            vectors = end_coords - start_coords
            distances = np.linalg.norm(vectors, axis=2)
            #avg_distances = np.mean(distances, axis=0)

            distance_list.append(torch.from_numpy(distances))
        else:
            distance_list.append(None)

    return distance_list

def apply_joint_subsampling(data, edge_index):
    """
    :param: original_data: torch.tensor of shape (3, F, 32)
    Subsample the data by removing joints specified in joint_cutout
    Will return subsampled data and updated edge index
    :return:
    """
    joint_cutout = joints_file.joints_cutout
    subsampled_data = data.clone()
    num_joints = subsampled_data.shape[2]
    # keep joints that are not in joint_cutout as a tensor index list
    keep_joints = torch.tensor([i for i in range(num_joints) if i not in joint_cutout])
    subsampled_data = subsampled_data[:, :, keep_joints]

    # Update edge index
    new_edge_index = []
    for i in range(len(edge_index[0])):
        if edge_index[0][i].tolist() in keep_joints and edge_index[1][i].tolist() in keep_joints:
            amount1 = len([j for j in joint_cutout if j < edge_index[0][i].tolist()])
            amount2 = len([j for j in joint_cutout if j < edge_index[1][i].tolist()])
            new_edge_index.append((edge_index[0][i] - amount1, edge_index[1][i] - amount2))

    return subsampled_data, torch.tensor(new_edge_index).t().contiguous()

def resolve_memmap_path(csv_path, base_dir):
        """
        Final path = base_dir / dir2above / dir1above / filename
        Example:
        csv_path = /data/dataset1/subsetA/front/file123.csv
        base_dir = /mnt/memmaps
        -> /mnt/memmaps/subsetA/front/file123.csv
        """
        source_dir, source_name = os.path.split(csv_path)
        dir1 = os.path.basename(source_dir)                  # 'front'
        dir2 = os.path.basename(os.path.dirname(source_dir)) # 'subsetA'

        final_dir = os.path.join(base_dir, dir2, dir1)
        os.makedirs(final_dir, exist_ok=True)

        return os.path.join(final_dir, source_name)

def group_span_masking(mask_groups, mask: torch.Tensor, span_masking: int = 0, group_masking: float = 0.0) -> torch.Tensor:
    """
    Masks joint groups with a random temporal offset to prevent phase-locking.
    """
    T, J = mask.shape
    span_frames = max(1, int(span_masking))
    k = int(group_masking)

    if k == 0:
        return mask
    if k >= len(mask_groups):
        raise ValueError(f"k={k} must be less than groups {len(mask_groups)}")

    span_mask = mask.clone()

    # 1. Generate a random offset (0 to span_frames)
    # This determines where the "grid" starts relative to Frame 0
    offset = random.randint(0, span_frames - 1)

    # 2. Start the loop from negative offset
    # This simulates the mask grid sliding in from the left
    # e.g., if offset is 10, we start at -10.
    # Block 1: -10 to 22 (We only apply 0 to 22)
    # Block 2: 22 to 54 ...
    for start in range(-offset, T, span_frames):
        # Clamp the start/end to the valid video range [0, T]
        valid_start = max(0, start)
        valid_end = min(T, start + span_frames)
        
        # If the block is entirely off-screen (negative), skip it
        if valid_start >= valid_end:
            continue

        # Randomly choose k groups for this specific time block
        chosen = random.sample(mask_groups, k)
        
        for grp in chosen:
            span_mask[valid_start:valid_end, grp] = True

    return span_mask


import os
import json
import fcntl


def save_index_cache(cache_path, data_file_label, index_mapping, normalized_lengths=None):
    """
    Save index mapping to cache file.

    Args:
        cache_path: Full path to cache file
        data_file_label: List of (file_pair, activity) tuples
        index_mapping: List of index tuples
        normalized_lengths: Optional dict of {file_idx: length}
    """
    cache_data = {
        'data_file_label': data_file_label,
        'index_mapping': index_mapping,
    }

    if normalized_lengths is not None:
        cache_data['normalized_lengths'] = {str(k): v for k, v in normalized_lengths.items()}

    lock_path = cache_path + '.lock'
    os.makedirs(os.path.dirname(lock_path) or '.', exist_ok=True)

    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    try:
        os.unlink(lock_path)
    except OSError:
        pass


def load_index_cache(cache_path):
    """
    Load index mapping from cache file.

    Args:
        cache_path: Full path to cache file

    Returns:
        tuple: (data_file_label, index_mapping, normalized_lengths) or None if doesn't exist
    """
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        data_file_label = cache_data['data_file_label']
        index_mapping = [
            tuple(x) if not isinstance(x, list) or not isinstance(x[0], list)
            else [tuple(y) for y in x]
            for x in cache_data['index_mapping']
        ]

        normalized_lengths = None
        if 'normalized_lengths' in cache_data:
            normalized_lengths = {int(k): v for k, v in cache_data['normalized_lengths'].items()}

        return (data_file_label, index_mapping, normalized_lengths)

    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def limit_users(data_file_label, limit_users, test_set, eval_set):
    if limit_users == -1 or test_set or eval_set:
        return data_file_label

    # Single pass: collect unique users and build result
    seen_users = set()
    filtered_data_file_label = []

    for file_pair, activity in data_file_label:
        subject_id = file_pair[0].split('/')[-1]

        if subject_id not in seen_users:
            if len(seen_users) >= limit_users:
                continue
            seen_users.add(subject_id)

        filtered_data_file_label.append((file_pair, activity))

    return filtered_data_file_label

def id_date2research_stage(subject_id, date, research_stage_df=None):
    """
    Map subject ID and recording date to research stage.
    """
    if research_stage_df is None and os.path.exists("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/id_date_long.csv"):
        research_stage_df = pd.read_csv("/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/id_date_long.csv")

    record = research_stage_df[
        (research_stage_df['subject_id'] == subject_id) &
        (research_stage_df['date'] == date)
    ]
    if not record.empty:
        return record.iloc[0]['research_stage']

    return None


def duplicate_sequences(mapping, data_file_label):
    """
    Duplicate the sequences to make it equal, with preference for underrepresented files
    """
    activity_seq_count = {}

    # check how many sequences are there for each activity then duplicate the sequences to make it equal
    for (file_idx, sequence_idx, augment_name) in mapping:
        activity = data_file_label[file_idx][1]
        if activity not in activity_seq_count:
            activity_seq_count[activity] = 0
        activity_seq_count[activity] += 1

    max_seq_count = max(activity_seq_count.values())

    for activity, seq_count in activity_seq_count.items():
        # if the sequence count is less than 80% of the maximum sequence count across all activities, 
        # duplicate sequences until it reaches the threshold
        target_count = max_seq_count - 0.1 * max_seq_count
        if seq_count < target_count:
            needed = int(target_count - seq_count)
            # Collect sequences for this activity
            activity_sequences = [(file_idx, sequence_idx, augment_name) 
                                for (file_idx, sequence_idx, augment_name) in mapping 
                                if data_file_label[file_idx][1] == activity]
            
            # Count sequences per file within this activity
            file_seq_count = {}
            for (file_idx, sequence_idx, augment_name) in activity_sequences:
                if file_idx not in file_seq_count:
                    file_seq_count[file_idx] = 0
                file_seq_count[file_idx] += 1
            
            # Create weights inversely proportional to file frequency (underrepresented files get higher weight)
            max_file_count = max(file_seq_count.values())
            weights = []
            for (file_idx, sequence_idx, augment_name) in activity_sequences:
                # Inverse weighting: files with fewer sequences get higher probability
                weight = max_file_count / file_seq_count[file_idx]
                weights.append(weight)
            
            # Normalize weights to probabilities
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            
            # Sample with replacement, weighted by rarity
            duplicated_indices = random.choices(range(len(activity_sequences)), 
                                               weights=probabilities, 
                                               k=needed)
            for idx in duplicated_indices:
                mapping.append(activity_sequences[idx])
                
    return mapping

def apply_median_filter(sequence, kernel_size=3):
    """
    Applies a median filter along the temporal axis to remove shot noise.
    
    Args:
        sequence (np.ndarray): Shape [Frames, Joints, 3] or [Frames, Joints, 4]
        kernel_size (int): Window size. Must be odd (3 or 5 recommended).
        
    Returns:
        np.ndarray: The despiked sequence.
    """
    # Define the footprint: (Time_Window, Joint_Window, Coord_Window)
    # We use (kernel_size, 1, 1) to ensure we filter ONLY across time.
    # We do NOT want to average the Knee with the Ankle (spatial).
    footprint = (kernel_size, 1, 1)
    
    # mode='nearest': Repeats the edge values. 
    # Prevents "zero padding" artifacts at the start/end of the clip.
    filtered_sequence = median_filter(sequence, size=footprint, mode='nearest')
    
    return filtered_sequence

