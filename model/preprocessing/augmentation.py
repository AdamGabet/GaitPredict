import torch
from scipy.ndimage import gaussian_filter1d
import numpy as np


class Augmentation():
    def __init__(self, tensor_data, np_data, augmentations=None, use_angular=False, **kwargs):

        self.kwargs = kwargs
        self.tensor_data = tensor_data  # Shape: (3, F, 32)
        self.np_data = np_data  # Shape: (3, F, 32)
        self.augmentations = augmentations or {}
        self.augmented_data = [self.tensor_data]
        self.use_angular = use_angular
        self.apply_augmentations()


    def apply_augmentations(self):
        if self.augmentations.get('rotation', False):
            aug_data =  self.apply_rotation()
            self.augmented_data.append(aug_data)
        if self.augmentations.get('scaling', False):
            aug_data = self.apply_scaling()
            self.augmented_data.append(aug_data)

        if self.augmentations.get('translation', False):
            aug_data = self.apply_translation()
            self.augmented_data.append(aug_data)

        if self.augmentations.get('jittering', False):
            aug_data = self.apply_jittering()
            self.augmented_data.append(aug_data)

        if self.augmentations.get('mirroring', False):
            aug_data = self.apply_mirroring()
            self.augmented_data.append(aug_data)

        slicing = [key for key in self.augmentations.keys() if 'temporal_downsample_slicing' in key]
        if len(slicing) > 0:
            for key in slicing:
                start = int(key[-1])
                aug_data = self.apply_temporal_downsample_slicing(start=start)
                self.augmented_data.append(aug_data)

        if self.augmentations.get('temporal_downsample_interpole', False):
            aug_data = self.apply_temporal_downsample_interpole()
            self.augmented_data.append(aug_data)

        if self.augmentations.get('temporal_upsample', False):
            aug_data = self.apply_temporal_upsample()
            self.augmented_data.append(aug_data)

        if self.augmentations.get('elastic_deformation', False):
            aug_data = self.apply_elastic_deformation()
            self.augmented_data.append(aug_data)


    def apply_rotation(self):
        # self.original_data is of shape (3, F, 32, 1) rotated data will be of shape (3, F, 32)
        rotated_data = self.tensor_data.clone()
        max_angle = self.kwargs.get('rotation_max_angle', 2*np.pi)
        angle = torch.rand(1) * max_angle
        rotation_matrix = torch.tensor([
            [torch.cos(angle), 0,torch.sin(angle)],
            [0, 1,0],
            [-torch.sin(angle), 0, torch.cos(angle)]
        ])

        # We need to add extra dimensions to rotation_matrix for broadcasting to work across F and num_joints
        # Assuming rotated_data has shape (3, F, num_joints)
        F, num_joints = rotated_data.shape[1], rotated_data.shape[2]

        rotation_matrix = rotation_matrix.view(1, 1, 3, 3).expand(F, num_joints, 3, 3)

        # Transpose tensor to shape (F, num_joints, 3) to align for matmul
        tensor_t = rotated_data.permute(1, 2, 0)

        # Apply rotation
        rotated_tensor = torch.matmul(rotation_matrix, tensor_t.unsqueeze(-1)).squeeze(-1)

        # Transpose back to original shape
        rotated_data = rotated_tensor.permute(2, 0, 1)  # Now shape (3, F, num_joints)

        return rotated_data

    def apply_scaling(self):
        max_scale = self.kwargs.get('scaling_max', 1.3)
        min_scale = self.kwargs.get('scaling_min', 0.7)
        scale_factor = min_scale + torch.rand(1) * (max_scale - min_scale)
        scaled_data = self.tensor_data * scale_factor
        return scaled_data

    def apply_translation(self):
        translation_max = self.kwargs.get('translation_max', 0.1)
        translation = torch.rand(3, 1, 1) * 2 * translation_max - translation_max
        translated_data = self.tensor_data + translation
        return translated_data

    def apply_jittering(self):
        """
        Apply Gaussian noise to the skeleton data.
        - Always jitter the first 3 dims (0:3).
        - Leave dim 3 unchanged.
        - If self.use_angular is True, also jitter dims 4:8 (quaternion) with the same sigma
          and renormalize to keep unit quaternions.
        """
        jitter_sigma = self.kwargs.get('jitter_sigma', 0.005)

        gaussian_noise = torch.normal(
            mean=0.0,
            std=jitter_sigma,
            size=self.tensor_data.shape,
            device=self.tensor_data.device
        )

        # Jitter positions (dims 0:3)
        pos = self.tensor_data[:3, :, :] + gaussian_noise[:3, :, :]

        # Keep dim 3 as-is
        d3 = self.tensor_data[3:4, :, :]

        if getattr(self, "use_angular", False):
            # Jitter quaternion (dims 4:8) with same sigma, then renormalize
            quat = self.tensor_data[4:8, :, :] + gaussian_noise[4:8, :, :]
            quat = quat / (quat.norm(dim=0, keepdim=True) + 1e-8)
            rest = self.tensor_data[8:, :, :]
            jittered_data = torch.cat([pos, d3, quat, rest], dim=0)
        else:
            # No angular jitter: leave dims 4: untouched
            jittered_data = torch.cat([pos, d3, self.tensor_data[4:, :, :]], dim=0)

        return jittered_data

    def apply_mirroring(self):
        mirrored_data = self.tensor_data.clone()
        mirrored_data[0] = -mirrored_data[0]  # Flip x-axis
        return mirrored_data

    def apply_temporal_downsample_interpole(self):
        """
        uses interpolation to downsample the data
        :return:
        """
        framesize = self.kwargs.get('down_frame_size', 100)
        # Downsampling
        downsampled_data = torch.nn.functional.interpolate(self.tensor_data.permute(0, 2, 1),
                                                           size=framesize, mode='linear')
        return downsampled_data.permute(0, 2, 1)

    def apply_temporal_downsample_slicing(self, start=0):
        """
        uses slicing to remove frames
        :return:
        """
        skip = self.kwargs.get('skip', 2)
        downsampled_data = self.tensor_data[:, start::skip, :]
        return downsampled_data




    def apply_temporal_upsample(self):
        """
        Interpolation (upsampling)
        uses a tensor of 3,F,32
        """
        framesize = self.kwargs.get('up_frame_size', 100)
        # Downsampling
        upsampled_data = torch.nn.functional.interpolate(self.tensor_data.permute(0, 2, 1),
                                                           size=framesize, mode='linear')
        return upsampled_data.permute(0, 2, 1)

    def apply_temporal_split(self):
        """
        :arg add to kwargs frames_split: int
        split tensor into sequences of length frames_split
        :return: list of split data and original edge index
        """
        frames_split = self.kwargs.get('frames_split', 100)
        split_data = []
        for a in range(0, self.tensor_data.shape[1], frames_split):
            split_data.append(self.tensor_data[:, a:a+frames_split, :])
        return split_data

    def find_closest_frame(self, sequence, min_gap=10, max_gap=50, sts=False):
        """
        Find the frame that is closest in Euclidean distance to a previous frame.

        :param sequence: numpy array of shape (3, F, 32) where F is the number of frames
        :param min_gap: minimum number of frames between the current and previous frame to consider
        :return: tuple (frame_index, previous_frame_index, distance)
        """
        F = sequence.shape[1]
        min_distance = float('inf')
        closest_frame = None
        closest_previous_frame = None
        joint_indexs = joints_file.joints_cycle
        if sts:
            joint_indexs = joints_file.joints_cycle_sts
        counter = 0
        current_frame = sequence[:, max_gap, joint_indexs]
        for j in range(0, max_gap - min_gap):
            previous_frame = sequence[:, j, joint_indexs]
            distance = np.linalg.norm(current_frame - previous_frame)

            if distance < min_distance:
                min_distance = distance
                closest_previous_frame = j
                current = max_gap

        return current, closest_previous_frame, min_distance

    def calculate_average_cycle_length(self, min_gap=10, jump_gap=50,  num_cycles=30, sts=False):
        """
        Calculate the average cycle length based on multiple detected cycles.

        :param sequence: numpy array of shape (3, F, 32) where F is the number of frames
        :param min_gap: minimum number of frames between the current and previous frame to consider
        :param num_cycles: number of cycles to detect for averaging
        :return: average cycle length
        """
        sequence = self.np_data[:, :4000, :]
        cycle_lengths = []
        start_frame = 0
        if sts:
            start_cycle_joint = joints_file.k4abt_joints.PELVIS
        else:
            start_cycle_joint = joints_file.k4abt_joints.ANKLE_LEFT
        for _ in range(num_cycles):
            closest_to_zero = float('inf')
            while sequence[1, start_frame, start_cycle_joint] < closest_to_zero:
                start_frame += 1
                if start_frame == sequence.shape[1]:
                    break
                closest_to_zero = sequence[1, start_frame, start_cycle_joint]
            if start_frame + jump_gap > sequence.shape[1]:
                break
            start, previous_frame, _ = self.find_closest_frame(sequence[:, start_frame:, :], min_gap, jump_gap, sts)

            cycle_length = start - previous_frame
            cycle_lengths.append(cycle_length)
            start_frame += cycle_length

        if not cycle_lengths:
            return None

        return int(np.mean(cycle_lengths))

    def temporal_elastic_deformation(self): #TODO
        """
        Apply temporal elastic deformation to skeleton data.

        Args:
        - data: numpy array of shape (3, F, 32) where 3 is x,y,z, F is frames, 32 is joints
        - sigma: standard deviation for Gaussian kernel (controls smoothness)
        - alpha: magnitude of deformation

        Returns:
        - Deformed data of the same shape as input
        """
        data = self.np_data  # (3, F, 32)
        sigma = self.kwargs.get('elastic_sigma', 0.02)
        alpha = self.kwargs.get('elastic_alpha', 0.02)

        F = 35
        # Generate random displacement fields
        dx = np.random.randn(F, data.shape[2])  # (F, 32)
        dy = np.random.randn(F, data.shape[2])  # (F, 32)
        dz = np.random.randn(F, data.shape[2])  # (F, 32)

        # Apply temporal smoothing to displacement fields
        dx_smooth = gaussian_filter1d(dx, sigma=sigma, axis=0, mode='reflect')
        dy_smooth = gaussian_filter1d(dy, sigma=sigma, axis=0, mode='reflect')
        dz_smooth = gaussian_filter1d(dz, sigma=sigma, axis=0, mode='reflect')

        # Combine smoothed displacement fields
        displacement = np.stack([dx_smooth, dy_smooth, dz_smooth], axis=0)  # (3, F, 32)

        # Apply deformation
        for i in range(data.shape[1] // F):
            if data.shape[1] < (i+1)*F:
                data = data[:, :i*F, :]
                break
            data[:,i*F:(i+1)*F,:] = data[:,i*F:(i+1)*F,:] + alpha * displacement

        return [data]


    # Example usage:
    if __name__ == "__main__":
        # Create dummy data
        num_frames = 100
        num_joints = 32
        data = torch.randn(3, num_frames, num_joints)
        edge_index = torch.randint(0, num_joints, (2, 50))  # 50 random edges

        # Define augmentations to apply
        augmentations = {
            'rotation': True,
            'scaling': True,
            'translation': True,
            'jittering': True,
            'mirroring': True,
            'subsampling': True,
            'temporal_adjustment': True,
            'noise_injection': True,
            'cutout': True,
            'elastic_deformation': True
        }