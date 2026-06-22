import random
from time import sleep

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import fcntl

from model.preprocessing.utils import *

from model.preprocessing import args
import model.preprocessing.joints_file as joints_file
from model.preprocessing.augmentation import Augmentation
from model.preprocessing.normalizing_time_and_space import normalize_cycle_and_space


def get_labels(directory, labels, **kwargs):
    """Stub: label loading from proprietary phenotype data is not included in this release.
    Returns None; pass labels=None to DualCameraDataset to run without supervised labels.
    """
    return None
from scipy.ndimage import median_filter

class DualCameraDataset(Dataset):
    """
    Dataset class for loading and preprocessing 3D joint data from CSV files
    """

    def __init__(self, root_dir, graph_data=False, data_dict=None, size_seq=100, labels=None, test_set=False,
                 eval_set=False,
                 edge_directional=False, overlap_sequence=0, single_activity=None, eval_unique_ids=False, five_seq_together=False,
                 args_cfg=None):
        self.graph_data = graph_data
        self.five_seq_together = five_seq_together
        self.centroid = None
        self.root_dir = root_dir
        self.args = args_cfg or args.active_args()

        if self.args.use_memmap and self.args.load_to_ram:
            raise ValueError("use_memmap and load_to_ram are mutually exclusive")
        if self.args.use_memmap and self.args.save_as_float16:
            print("Warning: memmap uses float32; overriding save_as_float16 to False")
            self.args.save_as_float16 = False
        self.augment_list = [key for key, value in self.args.augments.items() if value]
        self.test_set = test_set
        self.eval_set = eval_set
        self.override_mask_set = False # if True will use masking on this dataset even if its eval or test
        self.overlap_sequence = overlap_sequence
        self.eval_unique_ids = eval_unique_ids

        self.per_joint_amount = 3
        if self.args.use_confidence:
            self.per_joint_amount = 4
        if self.args.use_angular:
            self.per_joint_amount = 8

        if size_seq:
            self.size_seq = size_seq
            self.len_videos = self.args.videos_lens

        self.exclude_list = self.args.exclude_list

        if data_dict:
            self.data_dict = data_dict
        else:
            self.data_dict = self._get_data_dict(self.root_dir)

        self.edge_index = self._get_edge_tensor(edge_directional, self.args)


        # Path is environment-configurable so the pipeline runs off the original cluster.
        # Falls back to the historical cluster path for backward compatibility.
        research_stage_date_file = os.getenv(
            "RESEARCH_STAGE_DATE_FILE",
            "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/id_date_long.csv",
        )
        if os.path.exists(research_stage_date_file):
            self.research_stage_date_df = pd.read_csv(research_stage_date_file)

        if single_activity is not None:
            self.classes = [single_activity]
        else:
            self.classes = [a for a in self.data_dict.keys() if a not in self.exclude_list]

        self.label_names = labels
        if labels:
            # label can be any data in subject loader
            self.labels = get_labels(directory=self.root_dir, labels=labels, with_nan=True, old_cache=False, use_research_stage=True)
        else:
            self.labels = None
            print('will have labels as None')
        self.data_file_label, self.mapping = self._create_index_mapping()


        if self.args.equal_seq_per_activity:
            self.mapping = duplicate_sequences(self.mapping, self.data_file_label)

    def _get_data_dict(self, dir):
        # retrieves all file in a directory which is the root_dir
        files_dict = {}
        # Walk through the directory structure starting from the base_directory
        for root, dirs, files in os.walk(dir):
            # Exclude the base directory itself, only process subdirectories
            if root != dir:
                # Prepare a key for the dictionary from the directory name
                subdirectory = os.path.basename(root)
                # Prepare the list of file paths in this subdirectory
                if self.args.two_viewpoints:
                    # save a list for each file 0 is the file path 1 is None: add loaded pandas here
                    file_paths = [[os.path.join(root, file), None, None] for file in files if
                                  'front' in file]
                else:
                    if self.args.side_views:
                        file_paths = [[os.path.join(root, file), None] for file in files]
                    else:
                        file_paths = [[os.path.join(root, file), None] for file in files if 'front' in file]
                # Assign the list to the subdirectory key in the dictionary
                files_dict[subdirectory] = file_paths
        return files_dict

    def _create_index_mapping(self):
        """
        create an index map for iterating over fixed size sequences from each subject
        :return: a tuple
        """
        data_file_label = []
        index_mapping = []
        for class_name in self.classes:
            for file_path in self.data_dict[class_name]:
                file_idx = len(data_file_label)
                data_file_label.append((file_path, class_name))

        data_file_label = limit_users(data_file_label, self.args.limit_users, self.test_set, self.eval_set)
        print(f"data_file_label length: {len(data_file_label)} with limit_users: {self.args.limit_users}")
        if self.five_seq_together:
            # Dictionary to track available sequences by subject and activity
            # Structure: {subject_id: {activity: [(file_idx, seq_idx, augment), ...]}}
            subject_activity_sequences = {}

            # Step 1: Build data_file_label and organize sequences by subject and activity
            for file_idx, (file_pair, activity) in enumerate(data_file_label):
                class_name = activity
                file_path = file_pair

                # Extract subject ID
                subject_id = "_".join(file_path[0].split('/')[-1].split('_')[:2])

                # Initialize subject's entry if needed
                if subject_id not in subject_activity_sequences:
                    subject_activity_sequences[subject_id] = {}

                # Initialize activity's entry for this subject if needed
                if class_name not in subject_activity_sequences[subject_id]:
                    subject_activity_sequences[subject_id][class_name] = []

                # Calculate number of sequences in this file
                num_frames = self.args.videos_lens[class_name]
                num_sequences = int((num_frames - self.overlap_sequence) // (self.size_seq - self.overlap_sequence))

                if self.args.seq_per_person != -1:
                    num_sequences = min(num_sequences, self.args.seq_per_person)

                # Determine augmentation
                if (self.test_set or self.eval_set) and not self.args.augment_eval:
                    augment_list = ["None"]
                else:
                    augment_list = self.augment_list

                # Add all sequences for each augmentation to the subject's activity list
                for seq_idx in range(num_sequences):
                    for augment_name in augment_list:
                        subject_activity_sequences[subject_id][class_name].append((file_idx, seq_idx, augment_name))

            # Step 2: Create multi-activity groups for each subject
            for subject_id, activities in subject_activity_sequences.items():
                # Only process subjects who have sequences for all activity classes
                if len(activities) == len(self.classes):
                    # Find the minimum number of sequences available for any activity
                    min_sequences = min(len(sequences) for sequences in activities.values())

                    # Create as many multi-activity groups as possible
                    for i in range(min_sequences):
                        group_sequences = []

                        # For each activity, add one sequence to the group
                        for activity in self.classes:
                            sequences = activities[activity]
                            # Use modulo to cycle through sequences if needed
                            sequence = sequences[i % len(sequences)]
                            group_sequences.append(sequence)

                        # Add all sequences in this group to the index mapping
                        index_mapping.append(group_sequences)

            print(
                f"Created index mapping with {len(index_mapping)} entries from {len(subject_activity_sequences)} subjects")
            return data_file_label, index_mapping
        elif self.args.normalize_time or self.args.normalize_space:
            # Generate cache key for normalized time mode
            base_dir = self._get_base_mmap()
            # get the parent dir of base dir
            parent = os.path.dirname(os.path.abspath(base_dir))
            # add index_time{self.args.cycle_len}_space{self.args.normalize_space}_perjoint{self.per_joint_amount}
            add_prefix = "train" if not (self.test_set or self.eval_set) else "eval" if self.eval_set else "test"
            if self.args.centroid_type == 'second':
                add_centroidprefix = "centroid2_"
            else:
                add_centroidprefix = ""
            cache_filename = f"newusers_{add_centroidprefix}despiked_seq{self.size_seq}_users{self.args.limit_users}_{add_prefix}_index_time{self.args.cycle_len}_space{int(self.args.normalize_space)}_perjoint{self.per_joint_amount}_noise{int(self.args.augments['jittering'])}_none{int(self.args.augments['None'])}_seqper{self.args.seq_per_person}.json"

            cache_path = os.path.join(parent, cache_filename)

            # Try to load from cache
            cached_data = load_index_cache(cache_path)
            if cached_data is not None:
                print(f"✓ Loaded normalized index mapping from cache: {cache_path}")
                data_file_label, index_mapping, normalized_lengths = cached_data
                return data_file_label, index_mapping

            print("Cache miss - creating normalized index mapping...")

            normalized_lengths = {}
            # Pass 1: Normalize all sequences and record actual lengths
            for file_idx, (file_pair, activity) in enumerate(data_file_label):
                csv_path = file_pair[0]
                cache_idx = 1  # index for front view

                file_entry, _ = data_file_label[file_idx]
                base_dir = self._get_base_mmap()
                memmap_path = resolve_memmap_path(csv_path, base_dir)
                record = file_entry[cache_idx]
                if record != memmap_path:
                    file_entry[cache_idx] = memmap_path
                if not os.path.exists(memmap_path):
                    front_length = self._add_memmap_entry(memmap_path, csv_path, front=True, activity=activity)
                else:
                    mapped = np.load(memmap_path, mmap_mode='r')
                    front_length = mapped.shape[0]

                normalized_lengths[file_idx] = front_length

                # Handle back view if needed
                if self.args.two_viewpoints:
                    back_path = csv_path.replace('front', 'side')
                    cache_idx = 2  # index for front view

                    file_entry, _ = data_file_label[file_idx]
                    base_dir = self._get_base_mmap()
                    memmap_path = resolve_memmap_path(back_path, base_dir)
                    record = file_entry[cache_idx]
                    if record != memmap_path:
                        file_entry[cache_idx] = memmap_path

                    if not os.path.exists(memmap_path):
                        back_length = self._add_memmap_entry(memmap_path, back_path, front=False, activity=activity)
                    else:
                        mapped = np.load(memmap_path, mmap_mode='r')
                        back_length = mapped.shape[0]

                    if front_length > back_length:
                        print(f"Warning: front ({front_length}) > back ({back_length}) for {csv_path}")

            for file_idx, (file_pair, activity) in enumerate(data_file_label):
                actual_length = normalized_lengths[file_idx]

                # Calculate number of sequences based on actual length
                num_sequences = int((actual_length - self.overlap_sequence) //
                                    (self.size_seq - self.overlap_sequence))
                start_seq = 0
                if self.args.seq_per_person != -1:
                    # make the start sequence exactly in the middle where start_seq + seq_per_person < num_sequences
                    start_seq = max(0, (num_sequences - self.args.seq_per_person) // 2)
                    num_sequences = min(num_sequences, self.args.seq_per_person)

                if actual_length < self.size_seq:
                    print(f"Skipping {file_pair[0]}: normalized length {actual_length} < {self.size_seq}")
                    continue

                # Create entries
                if (self.test_set or self.eval_set) and not self.args.augment_eval:
                    for seq_idx in range(start_seq, start_seq + num_sequences):
                        index_mapping.append((file_idx, seq_idx, "None"))
                else:
                    for seq_idx in range(start_seq, start_seq + num_sequences):
                        for augment_name in self.augment_list:
                            index_mapping.append((file_idx, seq_idx, augment_name))

            print(f"✓ Created {len(index_mapping)} sequence indices from normalized gait cycles")

            # Save to cache

            save_index_cache(cache_path, data_file_label, index_mapping, normalized_lengths)

            return data_file_label, index_mapping

        else:
            class_lens = []
            running_total = 0
            for class_i in range(len(self.classes)):
                running_total += len(self.data_dict[self.classes[class_i]])
                class_lens.append(running_total - 1)

            prev = 0

            for file_idx in range(len(data_file_label)):
                num_frames = self.args.videos_lens[data_file_label[file_idx][1]]
                # round 5.1 to 6

                num_sequences = int((num_frames - self.overlap_sequence) // (self.size_seq - self.overlap_sequence))

                if self.args.seq_per_person != -1:
                    num_sequences = min(num_sequences, self.args.seq_per_person)

                # base_entries = [(file_idx, seq_idx, "None") for seq_idx in range(num_sequences)]
                # index_mapping.extend(base_entries)
                if (self.test_set or self.eval_set) and not self.args.augment_eval:
                    for seq_idx in range(num_sequences):
                        index_mapping.append((file_idx, seq_idx, "None"))
                    if file_idx in class_lens and self.eval_unique_ids:
                        index_mapping[prev:] = sorted(index_mapping[prev:], key=lambda x: x[1])
                        prev = len(index_mapping)
                    continue

                augmented_entries = [(file_idx, seq_idx, augment_name)
                                     for seq_idx in range(num_sequences)
                                     for augment_name in self.augment_list]
                index_mapping.extend(augmented_entries)

            return data_file_label, index_mapping


    def _get_min_max_values(self, first_k=None):
        if first_k is None:
            np_labels = np.array([a for a in self.labels.values()])
        else:
            np_labels = np.array([a for a in self.labels.values()])[:, :first_k]
        min_values = list(np.min(np_labels, axis=0))
        max_values = list(np.max(np_labels, axis=0))
        return min_values, max_values

    def _get_bins(self, size_bins, min_values, max_values, first3=False):
        range_values = [max_values[i] - min_values[i] for i in range(len(min_values))]
        if size_bins is None:
            size_bins = [10] * len(min_values)
        # add 1 to the bin size to include endpoints
        bin_sizes = [int(np.ceil(range_values[i] / size_bins[i])) + 1 for i in range(len(min_values))]
        min_values = [int(np.floor(a)) for a in min_values]
        return bin_sizes, size_bins, min_values

    def _get_num_frames(self, file_path):
        # Efficiently get the number of rows (frames) in a CSV file
        with open(file_path, 'r') as f:
            return sum(1 for _ in f) - 1  # Subtract 1 for header

    def _preprocess_labels(self, label):
        """
        Preprocess labels to ensure consistent scaling.
        VAT-related labels are scaled from cm³ to liters (divided by 1000).
        """
        if not isinstance(label, list):
            label = [label]
        
        processed_label = []
        for i, label_name in enumerate(self.label_names):
            if i < len(label):
                label_value = label[i]
                # Scale VAT labels from cm³ to liters
                if 'vat' in label_name.lower():
                    label_value = label_value / 100.0
                processed_label.append(label_value)
            else:
                processed_label.append(0)
        
        return processed_label

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        return self.data_file_label

    def __getitem__(self, idx):
        if self.five_seq_together:
            return self.get_item_five_seq(idx)

        file_idx, sequence_idx, augment_name = self.mapping[idx]

        try:
            tensor_data, np_data, original_tensor, full_mask = self._load_and_preprocess_sequence(file_idx,
                                                                                                  sequence_idx,
                                                                                                  augment_name)
        except ValueError as e:
            # log it once
            print(f"Skipping bad sequence at idx={idx}: {e}")
            # pick a different sample
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        except Exception as e:
            print(f"Error loading sequence at idx={idx}: {e}")
            print(f'Error in file: {self.data_file_label[file_idx][0], sequence_idx, augment_name}')
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        # if tensor frame len is different than size_seq (will ruin batching)
        if tensor_data.size(1) != self.size_seq:
            print(f'File is too short: {self.data_file_label[file_idx][0], sequence_idx, augment_name}')
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
        
        if tensor_data.dtype != torch.float32:
            print(f'File has wrong dtype: {self.data_file_label[file_idx][0], sequence_idx, augment_name}')
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        file_pair, activity_name = self.data_file_label[file_idx]
        id_10 = "_".join(file_pair[0].split('/')[-1].split('_')[:2])
        if not (file_pair[0].endswith('front.csv') or file_pair[0].endswith('back.csv')):
            date = file_pair[0].split('/')[-1].split('__')[-1].strip('.csv')
            research_stage = id_date2research_stage(id_10, date, self.research_stage_date_df)
        else:
            research_stage = None
            date = None
        view = file_pair[0].split('/')[-1].split('_')[-1].strip('.csv')
        if self.labels is not None:
            if (id_10, research_stage) in self.labels:
                label = self.labels[(id_10, research_stage)]
                # Preprocess VAT labels: scale from cm³ to liters (divide by 1000)
                label = self._preprocess_labels(label)
            else:
                print(f"Warning: {id_10} {research_stage} not in labels")
                label = [0]
        else:
            label = [0]

        if self.args.distances:
            distances = calculate_distances(np_data, self.edge_index)
            if augment_name == 'temporal_downsample_slicing':
                distances = [a[::self.args.augment_args['skip'], :] if a is not None else a for a in distances]
        else:
            distances = None

        if self.graph_data:
            from torch_geometric.data import Data
            # starts at 4, F, 32, 1 and ends at 32, 4, F, 1
            tensor_data = tensor_data.permute(2, 0, 1, 3)
            # starts at 32, 4, F, 1 and ends at 32*F, 4, 1 with adjusted edge index
            if tensor_data.size(2) > 1:
                tensor_data, edge_index = self.frames_to_graph(tensor_data)
            else:
                tensor_data, edge_index = tensor_data.squeeze(2), self.edge_index
            return Data(x=tensor_data, edge_index=edge_index, y=label[0], activity=activity_name,
                        distance1=distances[0], distance2=distances[1], id=id_10, camera_view=view, label_list=label)
        else:
            if self.args.two_viewpoints:
                # starts at 4, F, 32, 2 and ends at 2, F, 32, 4
                tensor_data = tensor_data.permute(3, 1, 2, 0)
                original_tensor = original_tensor.permute(3, 1, 2, 0)
            else:
                # starts at 3, F, 32, 1 and ends at F, 32, 3 because we squeeze the last dimension
                tensor_data = tensor_data.squeeze(3).permute(1, 2, 0)
                original_tensor = original_tensor.squeeze(3).permute(1, 2, 0)
                full_mask = full_mask.permute(1, 2, 0) # mask has only 3 dimensions

            return  dict({
                "_debug_idx": idx,
                "data": tensor_data,
                "original": original_tensor,
                "label": label[0],
                "activity": activity_name,
                "id": id_10,
                "label_list": label,
                "mask": full_mask,
                "research_stage": research_stage,
                "date": date,
                "sequence_idx": sequence_idx,
            })

    def get_item_five_seq(self, idx):
        group_index = self.mapping[idx]
        full_data = []
        full_original = []
        activities = []
        full_mask_data = []
        for file_idx, sequence_idx, augment_name in group_index:
            try:
                tensor_data, np_data, original_tensor, full_mask = self._load_and_preprocess_sequence(file_idx,
                                                                                                      sequence_idx,
                                                                                                      augment_name)
            except ValueError as e:
                # log it once
                print(f"Skipping bad sequence at idx={idx}: {e}")
                # pick a different sample
                new_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(new_idx)

            except Exception as e:
                print(f"Error loading sequence at idx={idx}: {e}")
                print(f'Error in file: {self.data_file_label[file_idx][0], sequence_idx, augment_name}')
                new_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(new_idx)

            # if tensor frame len is different than size_seq (will ruin batching)
            if tensor_data.size(1) != self.size_seq:
                print(f'File is too short: {self.data_file_label[file_idx][0], sequence_idx, augment_name}')
                new_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(new_idx)

            if tensor_data.dtype != torch.float32:
                print(f'File has wrong dtype: {self.data_file_label[file_idx][0], sequence_idx, augment_name}')
                new_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(new_idx)

            file_pair, activity_name = self.data_file_label[file_idx]
            activities.append(activity_name)

            view = file_pair[0].split('/')[-1].split('_')[-1].strip('.csv')

            if self.args.distances:
                distances = self.calculate_distances(np_data)
                if augment_name == 'temporal_downsample_slicing':
                    distances = [a[::self.args.augment_args['skip'], :] if a is not None else a for a in distances]
            else:
                distances = None

            if self.graph_data:
                from torch_geometric.data import Data
                # starts at 4, F, 32, 1 and ends at 32, 4, F, 1
                tensor_data = tensor_data.permute(2, 0, 1, 3)
                # starts at 32, 4, F, 1 and ends at 32*F, 4, 1 with adjusted edge index
                if tensor_data.size(2) > 1:
                    tensor_data, edge_index = self.frames_to_graph(tensor_data)
                else:
                    tensor_data, edge_index = tensor_data.squeeze(2), self.edge_index
                return Data(x=tensor_data, edge_index=edge_index, y=distances, )
            else:
                if self.args.two_viewpoints:
                    # starts at 4, F, 32, 2 and ends at F, 32, 4, 2
                    tensor_data = tensor_data.permute(1, 2, 0, 3)
                    original_tensor = original_tensor.permute(1, 2, 0, 3)
                else:
                    # starts at 3, F, 32, 1 and ends at F, 32, 3, 1
                    tensor_data = tensor_data.squeeze(3).permute(1, 2, 0)
                    original_tensor = original_tensor.squeeze(3).permute(1, 2, 0)
            full_data.append(tensor_data)
            full_original.append(original_tensor)
            full_mask_data.append(full_mask)

        # mix the order across the activities
        if self.args.shuffle_activities:
            # Use a fixed seed for reproducibility while ensuring shuffling is consistent
            shuffle_seed = random.randint(0, 100000)
            random_state = np.random.RandomState(shuffle_seed)

            # Generate shuffle indices and apply same shuffle to both lists
            shuffle_indices = list(range(len(full_data)))
            random_state.shuffle(shuffle_indices)

            # Apply shuffle to both lists
            full_data = [full_data[i] for i in shuffle_indices]
            full_original = [full_original[i] for i in shuffle_indices]
            full_mask_data = [full_mask_data[i] for i in shuffle_indices]

            # Also shuffle activities list to maintain correspondence
            activities = [activities[i] for i in shuffle_indices]

        # Concatenate all tensors along the first dimension
        tensor_data = torch.cat(full_data, dim=0)
        original_tensor = torch.cat(full_original, dim=0)
        full_mask_data = torch.cat(full_mask_data, dim=0)
        id_10 = "_".join(file_pair[0].split('/')[-1].split('_')[:2])
        if self.labels is not None:
            if id_10 in self.labels:
                label = self.labels[id_10]
                # Preprocess VAT labels: scale from cm³ to liters (divide by 1000)
                label = self._preprocess_labels(label)
            else:
                label = 0
        else:
            label = 0

        return dict({
            "_debug_idx": idx,
            "data": tensor_data,
            "original": original_tensor,
            "mask": full_mask_data,
            "label": label[0],
            "activity_list": activities,
            "id": id_10,
            "label_list": label
        })

    def _load_and_preprocess_sequence(self, file_idx, sequence_idx, augment_name):
        start_row = sequence_idx * (self.size_seq - self.overlap_sequence)
        file_pair, activity = self.data_file_label[file_idx]

        if self.args.load_to_ram:
            # front
            np_coords_front = self._get_or_load(file_idx, file_pair[0], 1, front=True, activity=activity)
            np_coords_front = np_coords_front[start_row:start_row + self.size_seq]

            # back
            if self.args.two_viewpoints:
                np_coords_back = self._get_or_load(file_idx, file_pair[0].replace('front', 'side'), 2, front=False, activity=activity)
                np_coords_back = np_coords_back[start_row:start_row + self.size_seq]
            else:
                np_coords_back = None
        elif self.args.use_memmap:
            np_coords_front = self._get_memmap_slice(
                file_idx=file_idx,
                csv_path=file_pair[0],
                cache_idx=1,
                front=True,
                start_row=start_row,
                length=self.size_seq,
                activity=activity
            )

            if self.args.two_viewpoints:
                np_coords_back = self._get_memmap_slice(
                    file_idx=file_idx,
                    csv_path=file_pair[0].replace('front', 'side'),
                    cache_idx=2,
                    front=False,
                    start_row=start_row,
                    length=self.size_seq,
                    activity=activity
                )
            else:
                np_coords_back = None
        else:
            # direct read
            skel_df_front = pd.read_csv(file_pair[0], dtype='float32')
            skel_raw = self._get_np_from_df(skel_df_front)
            skel_norm = self.normalize_seq_fast(skel_raw, front=True)
            np_coords_front = normalize_cycle_and_space(
                skel_norm,
                normalize_time=self.args.normalize_time,
                normalize_space=self.args.normalize_space,
                num_frames=self.args.cycle_len,
                activity=activity
            )
            # slice to the correct sequence
            np_coords_front = np_coords_front[start_row:start_row + self.size_seq]

            if self.args.two_viewpoints:
                skel_df_back = pd.read_csv(file_pair[0].replace('front', 'side'), dtype='float32')
                skel_raw_back = self._get_np_from_df(skel_df_back)
                skel_norm_back = self.normalize_seq_fast(skel_raw_back, front=False)
                np_coords_back = normalize_cycle_and_space(
                    skel_norm_back,
                    normalize_time=self.args.normalize_time,
                    normalize_space=self.args.normalize_space,
                    num_frames=self.args.cycle_len,
                    activity=activity
                )
                np_coords_back = np_coords_back[start_row:start_row + self.size_seq]
            else:
                np_coords_back = None

        return self._process_normalize_and_convert(
            skel1=np_coords_front,
            skel2=np_coords_back,
            augment_name=augment_name
        )

    def _get_or_load(self, file_idx, path, cache_idx, front=True, activity=None):
        file_entry, _ = self.data_file_label[file_idx]
        skel_cache = file_entry[cache_idx]

        if skel_cache is None:
            df = pd.read_csv(path, dtype='float32')
            skel_raw = self._get_np_from_df(df)
            skel_proc = self.normalize_seq_fast(skel_raw, front=front)
            skel_proc = normalize_cycle_and_space(
                skel_proc,
                normalize_time=self.args.normalize_time,
                normalize_space=self.args.normalize_space,
                num_frames=self.args.cycle_len,
                activity=activity
            )

            if self.args.save_as_float16:
                skel_proc = skel_proc.astype(np.float16, copy=False)
            self.data_file_label[file_idx][0][cache_idx] = skel_proc
            skel_cache = skel_proc
        if self.args.save_as_float16:
            skel_cache = skel_cache.astype(np.float32, copy=False)
        return skel_cache

    def _get_base_mmap(self):
        base_dir = self.args.memmap_dir
        base_dir += "despiked_"
        if self.args.normalize_time:
            base_dir += f"norm_time{self.args.cycle_len}_"
        if self.args.normalize_space:
            base_dir += "norm_space_"
        base_dir += f"num_users{self.args.limit_users}_"
        base_dir = base_dir + f"per_joint_{self.per_joint_amount}/"
        if self.args.centroid_type == 'second':
            base_dir += "_centroid2"
        if base_dir is None:
            raise ValueError("memmap_dir must be specified when using memmap")
        return base_dir


    def _add_memmap_entry(self, memmap_path, csv_path, front=True, activity=None):

        lock_path = memmap_path + '.lock'
        os.makedirs(os.path.dirname(lock_path) or '.', exist_ok=True)

        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            # Double-check after acquiring lock
            if not os.path.exists(memmap_path):
                os.makedirs(os.path.dirname(memmap_path) or '.', exist_ok=True)
                df = pd.read_csv(csv_path, dtype='float32')
                skel_raw = self._get_np_from_df(df)
                skel_proc = self.normalize_seq_fast(skel_raw, front=front)
                skel_proc = normalize_cycle_and_space(
                    skel_proc,
                    normalize_time=self.args.normalize_time,
                    normalize_space=self.args.normalize_space,
                    num_frames=self.args.cycle_len,
                    activity=activity,
                    sit_to_stand_frames=self.args.sit_to_stand_frames,
                )
                skel_proc = np.asarray(skel_proc, dtype=np.float32)
                actual_length = skel_proc.shape[0]
                memmap_arr = np.lib.format.open_memmap(
                    filename=memmap_path,
                    mode='w+',
                    dtype=np.float32,
                    shape=skel_proc.shape,
                )
                memmap_arr[:] = skel_proc
                memmap_arr.flush()
                del memmap_arr
                # Clean up lock file after memmap is created
            else:
                mapped = np.load(memmap_path, mmap_mode='r')
                return mapped.shape[0]  
        try:
            os.unlink(lock_path)
        except OSError:
            pass  # Another worker may have already deleted it
        return actual_length


    def _get_memmap_slice(self, file_idx, csv_path, cache_idx, front, start_row, length, activity):
        """Return a normalized slice from a disk-backed memmap without loading the full run."""
        file_entry, _ = self.data_file_label[file_idx]
        base_dir = self._get_base_mmap()
        memmap_path = resolve_memmap_path(csv_path, base_dir)
        record = file_entry[cache_idx]
        if record != memmap_path:
            file_entry[cache_idx] = memmap_path
        if not os.path.exists(memmap_path):
            self._add_memmap_entry(memmap_path, csv_path, front=front, activity=activity)

        mapped = np.load(memmap_path, mmap_mode='r')
        slice_end = start_row + length
        if slice_end > mapped.shape[0]:
            raise ValueError(f"Requested slice [{start_row}:{slice_end}] exceeds memmap length {mapped.shape[0]}")
        return mapped[start_row:slice_end].copy()

    

    def _get_np_from_df(self, skel_df):
        if self.per_joint_amount == 3:
            skel_df = skel_df.drop(columns=[col for col in skel_df.columns if '_c' in col or 'angle' in col])
        elif self.per_joint_amount == 4:
            skel_df = skel_df.drop(columns=[col for col in skel_df.columns if 'angle' in col])
        np_coords_front = skel_df.iloc[:, 2:].values.reshape(len(skel_df), -1, self.per_joint_amount)  # 4 for confidence
        return np_coords_front


    def _process_normalize_and_convert(self, skel1, augment_name='None', skel2=None):

        normalized_frames = skel1
        if normalized_frames.size == 0:
            raise ValueError(f"Empty sequence")

        if self.per_joint_amount == 8:
            normalized_frames = self.fix_quaternions(normalized_frames)


        # Reshape to (3, F, J)
        skel1_np_normalized = normalized_frames.transpose(2, 0, 1)
        tensor1 = torch.tensor(skel1_np_normalized, dtype=torch.float32)
        edges = self.edge_index

        # remove joints specified in joint_cutout updates edge index
        if self.args.cutout:
            tensor1, edges = apply_joint_subsampling(tensor1, self.edge_index)
        # Apply augmentations

        if augment_name != 'None':
            augment = Augmentation(tensor1, skel1_np_normalized, augmentations={augment_name: True},
                                   use_angular=self.args.use_angular, **self.args.augment_args)
            # augmented_data is a list where index 0 is the original
            tensor1 = augment.augmented_data[1]
            tensor0 = augment.augmented_data[0]
        else:
            tensor0 = tensor1.clone()

        # Apply masking for training OR when override_mask_set is True (for masked eval)
        if (not self.eval_set and not self.test_set) or self.override_mask_set:
            C, T, J = tensor1.shape
            # Build base masks at (T, J) / (T, 1), then broadcast once to (C, T, J)
            joint_mask_TJ = torch.zeros((T, J), dtype=torch.bool)
            frame_mask_T1 = torch.zeros((T, 1), dtype=torch.bool)

            # Random joint-wise masking
            if self.args.random_mask > 0:
                joint_mask_TJ |= (torch.rand((T, J)) < self.args.random_mask)

            # Random frame-wise masking (independent of random_mask)
            if getattr(self.args, "random_mask_frames", 0) > 0:
                frame_mask_T1 |= (torch.rand((T, 1)) < self.args.random_mask_frames)

            # Group masking over predefined joint groups (per frame)
            if self.args.group_masking > 0:
                groups = list(joints_file.noise_groups.values())  # ensure list for sampling

                # span masking expects (T, J)
                joint_mask_TJ = group_span_masking(groups, joint_mask_TJ, self.args.span_masking, self.args.group_masking)

            # Combine and broadcast to channels
            full_mask_TJ = joint_mask_TJ | frame_mask_T1.expand(T, J)
            full_mask = full_mask_TJ.unsqueeze(0).expand(C, T, J)

            # Zero-out where masked
            if self.args.zero_out_masked_joints:
                tensor1.masked_fill_(full_mask, 0)
        else:
            full_mask = torch.zeros_like(tensor1, dtype=torch.bool, device=tensor1.device)

        if self.args.specific_joint_keep_mask is not None and self.args.random_mask == 0:
            if len(self.args.specific_joint_keep_mask) > tensor1.size(2):
                raise ValueError(f"Specific joint mask length {len(self.args.specific_joint_keep_mask)} exceeds number of joints {tensor1.size(2)}")
            specific_mask = torch.ones(tensor1.size(1), tensor1.size(2), dtype=torch.bool)
            for joint in self.args.specific_joint_keep_mask:
                if joint < tensor1.size(2):
                    specific_mask[:, joint] = False
            specific_mask = specific_mask.unsqueeze(0).expand_as(tensor1)
            full_mask = torch.logical_or(full_mask, specific_mask)


        # These lines is for implementation of two viewpoint channel - returns a tensor of shape (3, F, 32, 2)
        if skel2 is not None:
            normalized_frames = skel2
            if self.per_joint_amount == 8:
                normalized_frames = self.fix_quaternions(normalized_frames)
            skel2_np_normalized = normalized_frames.transpose(2, 0, 1)
            tensor2 = torch.tensor(skel2_np_normalized, dtype=torch.float32)

            # remove joints specified in joint_cutout updates edge index
            if self.args.cutout:
                tensor2, edges = apply_joint_subsampling(tensor2, self.edge_index)
                self.edge_tensor = edges
            if augment_name != 'None':
                augment = Augmentation(tensor2, skel2_np_normalized, augmentations={augment_name: True},
                                       **self.args.augment_args)
                # augmented_data is a list where index 0 is the original
                tensor2 = augment.augmented_data[1]
            if self.args.zero_out_masked_joints:
                tensor2[full_mask] = 0
            the_tensor_combined = torch.stack((tensor1, tensor2), dim=-1)
            return the_tensor_combined, (skel1_np_normalized, skel2_np_normalized) , full_mask

        else:
            self.edge_index = edges
            the_tensor_combined = tensor1[:, :, :, torch.newaxis]
            tensor0 = tensor0[:, :, :, torch.newaxis]
            return the_tensor_combined, (skel1_np_normalized, None), tensor0, full_mask

    def normalize_seq_fast(self, coords: np.array, front=True) -> np.array:
        """
        Normalize a 3D array of joint coordinates.
        If self.use_angular is True and coords has quaternion at channels 4:8 (w,x,y,z),
        enforce |q|=1 and flip sign so w>=0.
        """
        if coords.size == 0:
            return np.array([])

        # --- Optional reformatting (your existing logic) ---
        if self.args.motionbert_format:
            coords = coords[:, joints_file.motionbert_keep_idxs]
            reordered_coords = np.zeros_like(coords)
            for i, idx in enumerate(joints_file.motionbert_order):
                pos = joints_file.motionbert_keep_idxs.index(idx)
                reordered_coords[:, i] = coords[:, pos]
            coords = reordered_coords

        if self.args.remove_noise_joints and not self.args.motionbert_format:
            keep_idx = [j for j in range(joints_file.k4abt_joints.len) if j not in joints_file.noise_joints]
            coords = coords[:, keep_idx]

    
        # Expecting layout: [x, y, z, confidence, (optional) q_w, q_x, q_y, q_z, ...]
        if coords.shape[2] < 4:
            raise ValueError ("Input coords must have at least 4 channels (x,y,z,confidence)")
        
        # apply median filter to remove shot noise
        coords = median_filter(coords, size=(3, 1, 1), mode='nearest')

        xyz = coords[:, :, :3]
        confidence = coords[:, :, 3]

        quat = None
        if self.per_joint_amount == 8:
            # channels 4:8 hold quaternion (w,x,y,z)
            quat = coords[:, :, 4:8].astype(np.float64)  # use float64 for stable norming

        # --- Centroid computation (uses only xyz) ---
        centroid_indices = [j for j in range(joints_file.k4abt_joints.len) if j in joints_file.joints_centroid]

        if self.args.centroid_type == 'second':
            centroid_indices = [j for j in range(joints_file.k4abt_joints.len) if j in joints_file.joints_centroid2]

        if not self.args.global_centroid:
            centroid = np.mean(xyz[:, centroid_indices], axis=1, keepdims=True)
        else:
            centroid = np.mean(xyz[0, centroid_indices], axis=0, keepdims=True)

        if front:
            self.centroid = centroid
        else:
            centroid = self.centroid

        # --- Normalize XYZ by furthest distance per frame ---
        xyz_centered = xyz - centroid
        furthest_distance = np.max(np.sqrt(np.sum(xyz_centered ** 2, axis=2)), axis=1, keepdims=True)
        # Avoid divide-by-zero
        furthest_distance = np.where(furthest_distance == 0, 1.0, furthest_distance)
        xyz_norm = xyz_centered / furthest_distance[:, np.newaxis]

        # --- Normalize quaternion (if present) ---
        if quat is not None:
            # Enforce unit norm
            q_norm = np.linalg.norm(quat, axis=2, keepdims=True)
            # Guard: if any norm is ~0, set to identity to avoid NaNs
            tiny = 1e-12
            bad = (q_norm < tiny)
            # normalize
            quat = quat / np.maximum(q_norm, tiny)
            # Replace ill-defined rows with identity quaternion [1,0,0,0]
            if np.any(bad):
                quat[bad.repeat(4, axis=2)] = 0.0
                quat[bad.squeeze(-1), 0] = 1.0  # set w=1 where bad

            # Fix sign ambiguity: ensure w >= 0
            sign = np.where(quat[:, :, 0:1] < 0.0, -1.0, 1.0)
            quat = quat * sign

        # --- Clean confidence to [0,1] per frame ---
        conf_max = np.max(confidence, axis=1, keepdims=True)
        conf_max = np.where(conf_max == 0, 1.0, conf_max)
        confidence = confidence / conf_max

        # --- Reassemble output ---
        if quat is not None:
            out = np.concatenate((xyz_norm, confidence[:, :, None], quat), axis=2)
        else:
            out = np.concatenate((xyz_norm, confidence[:, :, None]), axis=2)

        return np.nan_to_num(out)

    def fix_quaternions(self, skel_np):
        # ensure quaternions are unit norm and w>=0
        quats = skel_np[:, :, 4:8].astype(np.float64)
        norms = np.linalg.norm(quats, axis=2, keepdims=True)
        norms[norms == 0] = 1.0  # prevent division by zero
        quats /= norms
        quats[:, :, 0] = np.abs(quats[:, :, 0])  # enforce w >= 0
        skel_np[:, :, 4:8] = quats.astype(skel_np.dtype)
        return skel_np

    def frames_to_graph(self, tensor_data):
        """
        recieves tensor of shape (32, 3, F, 1) turns it into (32*F, 3, 1) and returns the edge index
        and the reshaped tensor
        :param tensor_data:
        :return:
        """
        # Reshape to (32*F, 3, 1)
        N, C, Fr, _ = tensor_data.size()
        tensor_data = tensor_data.permute(2, 0, 1, 3).reshape(-1, C, 1)
        adjusted_edge_indices = []
        edge_index = self.edge_index
        for i in range(Fr):
            node_offset = i * N
            adjusted_edge_index = edge_index + node_offset
            adjusted_edge_indices.append(adjusted_edge_index)
        edge_index = torch.cat(adjusted_edge_indices, dim=1)
        return tensor_data, edge_index

    @staticmethod
    def _get_edge_tensor(directional=True, args_cfg=None):
        args_cfg = args_cfg or args.active_args()

        if not args_cfg.motionbert_format and not args_cfg.remove_noise_joints:
            if not directional:
                joints = joints_file.k4abt_bones.copy()
                joints.extend([(j[1], j[0]) for j in joints])
                return torch.tensor(joints, dtype=torch.long).t()
            return torch.tensor(joints_file.k4abt_bones, dtype=torch.long).t()
        elif args_cfg.remove_noise_joints and not args_cfg.motionbert_format:
            # remove noise joints
            joints = joints_file.noise_bones.copy()
            if not directional:
                joints.extend([(j[1], j[0]) for j in joints])
                return torch.tensor(joints, dtype=torch.long).t()
            
            return torch.tensor(joints, dtype=torch.long).t()
        else:
            # motionbert format
            if not directional:
                joints = joints_file.motionbert_bones.copy()
                joints.extend([(j[1], j[0]) for j in joints])
                return torch.tensor(joints, dtype=torch.long).t()
            return torch.tensor(joints_file.motionbert_bones, dtype=torch.long).t()

    def _test(self, idx):
        file_idx, sequence_idx, augment_name = self.mapping[idx]
        file_path, label_name = self.data_file_label[file_idx]
        file_path2, l = self.data_file_label[file_idx + 1]
        print(file_path, file_path2)

        skel_df = pd.read_csv(file_path, dtype='float32', skiprows=0, nrows=5000)

        processed_df, np_process = self._process_normalize_and_convert(skel_df, 'None')
        tensors_list = torch.split(processed_df, 1, dim=-1)
        # Remove the singleton dimension that split introduces at index -1
        tensors_list = [t.squeeze(-1) for t in tensors_list]
        F = tensors_list[0].size(1)

        # Correct reshaping and permutation
        reshaped_tensor = tensors_list[0].permute(1, 2, 0).reshape(F, -1)
        processed_df = pd.DataFrame(reshaped_tensor.numpy())

        # Assign column names
        processed_df.columns = skel_df.columns[3:]

        # Concatenate with the first two columns of the original DataFrame
        processed_df = pd.concat([skel_df.iloc[:, 1:3].reset_index(drop=True), processed_df], axis=1)

        normalize_dir = '/home/adam/Documents/coding/gait/normalized_orig.csv'
        processed_df.to_csv(normalize_dir, index=False)

        # Process, normalize, and convert to tensor
        skel_df = pd.read_csv(file_path2, dtype='float32', skiprows=0, nrows=5000)
        tensor_data, np_data = self._process_normalize_and_convert(skel_df, 'kalman')
        tensors_list = torch.split(tensor_data, 1, dim=-1)
        # Remove the singleton dimension that split introduces at index -1
        tensors_list = [t.squeeze(-1) for t in tensors_list]

        # test augmentation
        augment = Augmentation(tensors_list[0], np_data, augmentations={})
        #repeat = augment.calculate_average_cycle_length()
        #augment.kwargs['skip'] = repeat
        #print(repeat)
        aug_data = [0]

        aug_data[0] = tensors_list[0]
        # aug_data[0] = torch.tensor(the_data, dtype=torch.float32)
        # turn back to pandas dataframe
        F = aug_data[0].size(1)
        rotated_df = pd.DataFrame(aug_data[0].permute(1, 2, 0).reshape(F, -1).numpy())
        new_columns = [a for a in
                       list(skel_df.columns[3:])]  # if int(a[:2].strip('_')) not in joints_file.joints_cutout]
        rotated_df.columns = new_columns
        # add first two columns of skel_df to rotated_df
        rotated_df = pd.concat([skel_df.iloc[:, 1:3], rotated_df], axis=1)

        save_dir = '/home/adam/Documents/coding/gait/augmented.csv'
        rotated_df.to_csv(save_dir)


def get_datasets(root_dir=None, data_dict=None, size_seq=500, graph_data=False, labels=None, overlap_sequence=0,
                 single_activity=None, eval_unique_ids=False, five_seq_together=False, args_cfg=None):
    """
    split into train, val, test, the train can contain augmentation but the val and test should not
    :return: train_dataset, val_dataset, test_dataset
    """
    #train_i, test_i, eval_i = DualCameraDataset.train_test_eval_split(dir=root_dir, test_size=test_size, eval_size=eval_size, r_seed=r_seed)
    if labels is None:
        labels = ['age']

    args_cfg = args_cfg or args.active_args()

    if root_dir is not None:
        train_dir = os.path.join(root_dir, 'train/')
        test_dir  = os.path.join(root_dir, 'test/')
        eval_dir  = os.path.join(root_dir, 'eval/')
    elif args_cfg.use_kalman:
        base = os.getenv(
            'SKELETON_DATA_KALMAN_DIR',
            '/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/',
        )
        train_dir = os.path.join(base, 'pointcloud_kalman_train/')
        test_dir  = os.path.join(base, 'pointcloud_kalman_test/')
        eval_dir  = os.path.join(base, 'pointcloud_kalman_eval/')
    else:
        base = os.getenv(
            'SKELETON_DATA_DIR',
            '/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/skeleton_full_data_sept/',
        )
        train_dir = os.path.join(base, 'train/')
        test_dir  = os.path.join(base, 'test/')
        eval_dir  = os.path.join(base, 'eval/')
    train_dataset = DualCameraDataset(train_dir, data_dict=data_dict, graph_data=graph_data, size_seq=size_seq,
                                      labels=labels,
                                      overlap_sequence=overlap_sequence, single_activity=single_activity,
                                      five_seq_together=five_seq_together,
                                      args_cfg=args_cfg)

    test_dataset = DualCameraDataset(test_dir, data_dict=data_dict, graph_data=graph_data, size_seq=size_seq,
                                     labels=labels, test_set=True,
                                     overlap_sequence=0, single_activity=single_activity,
                                     eval_unique_ids=eval_unique_ids,
                                     five_seq_together=five_seq_together,
                                     args_cfg=args_cfg)

    eval_dataset = DualCameraDataset(eval_dir, data_dict=data_dict, graph_data=graph_data, size_seq=size_seq,
                                     labels=labels, eval_set=True,
                                     overlap_sequence=0, single_activity=single_activity,
                                     eval_unique_ids=eval_unique_ids,
                                     five_seq_together=five_seq_together,
                                     args_cfg=args_cfg)

    return train_dataset, test_dataset, eval_dataset


if __name__ == '__main__':

    train_dataset, eval_dataset, test_dataset = get_datasets(size_seq=128,
                                                             labels=['age', 'hr_bpm', 'bmi', "Anxiety", "Depression"],
                                                             graph_data=False,
                                                             overlap_sequence=0,
                                                             single_activity=None)

    a = train_dataset[100]
    print(a['label_list'])
    #print(train_dataset[0])
    #print(train_dataset[0]['mask'].shape)
    exit()

    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data._utils.collate import default_collate
    # create a data
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=6, pin_memory=True,
                              persistent_workers=True)
    # check all the data loads correctly and print the files that are loaded to understand where it stopped
    for i, data in enumerate(train_loader):
        print(i, data['_debug_idx'], data['activity'])

    exit()
    a = 5
    from utils.visualyze_skeleton import visualize_skeleton_sequence
    # Visualize the skeleton sequence
    visualize_skeleton_sequence(
        a,
        bones=joints_file.motionbert_bones,
        title=f"Activity: {a.activity}"
    )

    exit()
    a = dataset.__getitem__(0)
    b = dataset.__getitem__(1)
    print(a[0].shape)
    edge_index = dataset.edge_index
    print(edge_index)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    exit()
    # Split into train, validation, and test sets
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    exit()
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
