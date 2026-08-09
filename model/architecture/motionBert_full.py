from ast import Not
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import warnings
import random
import numpy as np
from collections import OrderedDict
from functools import partial
from itertools import repeat
from model.architecture.droppath import DropPath
from model.preprocessing.joints_file import noise_groups


class ReconstructNet(nn.Module):
    """Wraps a DSTformer backbone with joint projection and reconstruction heads."""

    def __init__(self, model_backbone, dim_in=4, dim_out=3, modality_dropout_prob=0.0, zero_out_masked_joints=True,
                 return_block_embeddings=False, koleo_dim=None):
        super(ReconstructNet, self).__init__()
        self.model_backbone = model_backbone
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.modality_dropout_prob = modality_dropout_prob
        self.zero_out_masked_joints = zero_out_masked_joints
        self.return_block_embeddings = return_block_embeddings
        # Project raw joint coordinates into the backbone feature space
        self.joint_embd = nn.Linear(dim_in, model_backbone.dim_feat)

        # Configure modality-specific decoding heads and remember their sizes.
        self.pos_components = 3
        
        self.reconstruct_head = nn.Linear(model_backbone.embedding_dim, self.pos_components)
        self.has_quaternion = self.dim_out == 7
        if self.has_quaternion:
            self.quat_head = nn.Linear(model_backbone.embedding_dim, 4)

        # Koleo head: projects pooled joint-group embeddings to koleo space
        num_groups = len(noise_groups)
        if koleo_dim is not None:
            self.koleo_head = nn.Linear(num_groups * model_backbone.dim_feat, koleo_dim)
        else:
            self.koleo_head = None

        # Cached size of the pooled embedding (mean + max)
        self.pooled_dim = 2 * model_backbone.embedding_dim
        # Learned placeholder injected for masked joints after projection
        self.mask_embedding = nn.Parameter(torch.zeros(model_backbone.dim_feat))

    def forward(self, x, mask=None, representation=False, capture_attention=False):
        """
        Args:
            x: Tensor of shape [B, T, J, C_in]
            mask: Optional boolean tensor broadcastable to [B, T, J]
            representation: if True, return only embeddings
            return_pooled: when True also return mean+max pooled embeddings

        Returns:
            reconstructed: Tensor of shape [B, T, J, C_out]
            bundle: Dictionary containing the backbone bundle 
            bundle = {
                'pre_logits_embeddings': Tensor of shape [B, T, J, C_feat]
                'attention_maps': Dictionary containing the attention maps or None
                'block_embeddings': Tensor of shape [B, T, J, C_feat]
            }
        """
        batch_size, seq_len, num_joints, channels = x.size()
        # During training optionally remove either XYZ or quaternion channels per sample
        x = self._maybe_modality_dropout(x)
        x = x.reshape(batch_size * seq_len, num_joints, channels)
        projected = self.joint_embd(x)
        projected = projected.reshape(batch_size, seq_len, num_joints, -1)

        if mask is not None and not self.zero_out_masked_joints:
            mask = mask.to(projected.device)
            if mask.dim() == 4:
                mask = mask.any(dim=-1)
            mask = mask.unsqueeze(-1).to(dtype=torch.bool)
            mask_token = self.mask_embedding.view(1, 1, 1, -1)
            projected = torch.where(mask, mask_token, projected)

        backbone_bundle = self.model_backbone(projected, capture_attention=capture_attention)

        # copy backbone bundle so we can return it
        bundle = backbone_bundle.copy()

        pooled_pre_logits_embeddings = self._pool_embeddings_v1(backbone_bundle['pre_logits_embeddings'])
        pooled_block_embeddings = self._pool_embeddings_v1(backbone_bundle['block_embeddings'])

        bundle['pooled_pre_logits_embeddings'] = pooled_pre_logits_embeddings
        bundle['pooled_block_embeddings'] = pooled_block_embeddings

        # Koleo embeddings: pool by joint groups then project through koleo head
        pooled_v3 = self._pool_embeddings_v3(backbone_bundle['block_embeddings'])
        if self.koleo_head is not None:
            bundle['koleo_embeddings'] = self.koleo_head(pooled_v3)

        if representation:
            return bundle

        embeddings = backbone_bundle['pre_logits_embeddings']

        position_out = self.reconstruct_head(embeddings)
        if self.has_quaternion:
            quat_out = self.quat_head(embeddings)
            quat_out = F.normalize(quat_out, p=2.0, dim=-1, eps=1e-8)
            reconstructed = torch.cat((position_out, quat_out), dim=-1)
        else:
            reconstructed = position_out

        return reconstructed, bundle


    @staticmethod
    def _pool_embeddings(embeddings):
        """Mean and max pool embeddings across time and joints."""
        batch_size, seq_len, num_joints, dim = embeddings.shape
        flattened = embeddings.reshape(batch_size, seq_len * num_joints, dim)
        mean_pool = flattened.mean(dim=1)
        max_pool = flattened.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=1)

    @staticmethod
    def _pool_embeddings_v3(block_emb: torch.Tensor, joint_groups: dict = None) -> torch.Tensor:
        """
        Variant 3 pooling: Mean per joint group, mean over time.
        
        Args:
            block_emb: [B, T, J, D] block embeddings from model
            joint_groups: dict mapping group name to list of joint indices
            
        Returns:
            pooled: [B, num_groups * D] pooled embeddings
        """
        if joint_groups is None:
            joint_groups = noise_groups
        
        parts = []
        for indices in joint_groups.values():
            # Mean over joints in group, then mean over time
            group_pooled = block_emb[:, :, indices, :].mean(dim=2).mean(dim=1)
            parts.append(group_pooled)
        return torch.cat(parts, dim=1)

    @staticmethod
    def _pool_embeddings_v1(embeddings, joint_indices=None):
        """
        Pools embeddings to create a 3 * Dim feature vector:
        1. Mean over both time and joints
        2. Max over both time and joints  
        3. Std over time, then mean over joints
        
        Args:
            embeddings: [Batch, Seq, Num_Joints, Dim]
            joint_indices: (Optional) List of integers (e.g., arm indices). 
                        If None, uses all joints.
        
        Returns:
            pooled: [Batch, Dim * 3]
        """
        batch_size, seq_len, num_joints, dim = embeddings.shape

        # ==========================================
        # JOINT SELECTION (Filtering) - if specified
        # ==========================================
        if joint_indices is not None:
            # Select only the specific joints (e.g., Arms)
            embeddings = embeddings[:, :, joint_indices, :]
        
        # ==========================================
        # POOLING OPERATIONS
        # ==========================================
        
        # 1. Mean over both time and joints
        # Shape: [Batch, Dim]
        mean_pool = embeddings.mean(dim=(1, 2))
        
        # 2. Max over both time and joints
        # Shape: [Batch, Dim]
        max_pool = embeddings.max(dim=1).values.max(dim=1).values
        
        # 3. Std over time, then mean over joints
        # First: std over time -> [Batch, Num_Joints, Dim]
        # Then: mean over joints -> [Batch, Dim]
        std_time = embeddings.std(dim=1, unbiased=False)
        std_pool = std_time.mean(dim=1)
        
        # Final Embedding: [Batch, Dim * 3]
        return torch.cat([mean_pool, max_pool, std_pool], dim=1)

    def get_representation(self, x):
        return self.forward(x, representation=True)

    def _maybe_modality_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly drop XYZ or quaternion channels for selected batch items while training."""
        if not self.training or self.modality_dropout_prob <= 0.0 or not self.has_quaternion:
            return x

        batch_size = x.size(0)
        device = x.device
        # Bernoulli draw deciding which samples lose a modality.
        sample_mask = torch.rand(batch_size, device=device) < self.modality_dropout_prob
        if not sample_mask.any():
            return x

        pos_slice = slice(0, min(3, x.size(-1)))
        quat_start = 4
        quat_end = min(x.size(-1), 8)
        if quat_start >= quat_end:
            return x

        affected_indices = sample_mask.nonzero(as_tuple=False).view(-1)
        modality_choice = torch.rand(affected_indices.size(0), device=device) < 0.5
        x = x.clone()

        # Drop XYZ (keep confidence and quaternions) for half the selected samples.
        pos_indices = affected_indices[~modality_choice]
        if pos_indices.numel() > 0 and pos_slice.stop > pos_slice.start:
            x[pos_indices, :, :, pos_slice] = 0.0

        # Drop quaternion channels for the remaining selected samples.
        quat_indices = affected_indices[modality_choice]
        if quat_indices.numel() > 0:
            x[quat_indices, :, :, quat_start:quat_end] = 0.0

        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (torch.Tensor, float, float, float, float) -> torch.Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # cos, sin have shape [seq_len, dim]
    # q, k have shape [batch_like, seq_len, dim]
    # Unsqueeze cos and sin to be [1, seq_len, dim] for broadcasting
    cos = cos.unsqueeze(0) 
    sin = sin.unsqueeze(0)  
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla',
                 use_rope=False, use_flash_attn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = st_mode
        if self.mode == 'parallel':
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # Add RoPE support
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(head_dim)

        # Add Flash Attention support
        self.use_flash_attn = use_flash_attn

        self.attn_count_s = None
        self.attn_count_t = None

        # Runtime cache for logging attention weights on demand
        self._record_attention = False
        self._cached_attention = {'spatial': None, 'temporal': None}

    def enable_attention_recording(self):
        """Enable capturing attention weights for subsequent forward passes."""
        self._record_attention = True
        self._cached_attention = {'spatial': None, 'temporal': None}

    def disable_attention_recording(self):
        """Stop capturing attention weights."""
        self._record_attention = False

    def pop_attention_cache(self):
        """Return and reset the most recently cached attention weights."""
        cache = self._cached_attention
        self._cached_attention = {'spatial': None, 'temporal': None}
        return cache

    def forward(self, x, seqlen=1, num_sink_tokens=0):
        B, N, C = x.shape

        if self.mode == 'series':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v, seqlen=seqlen)
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal(q, k, v, seqlen=seqlen, num_sink_tokens=num_sink_tokens)
        elif self.mode == 'parallel':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x_t = self.forward_temporal(q, k, v, seqlen=seqlen, num_sink_tokens=num_sink_tokens)
            x_s = self.forward_spatial(q, k, v, seqlen=seqlen)

            alpha = torch.cat([x_s, x_t], dim=-1)
            alpha = alpha.mean(dim=1, keepdim=True)
            alpha = self.ts_attn(alpha).reshape(B, 1, C, 2)
            alpha = alpha.softmax(dim=-1)
            x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
        elif self.mode == 'coupling':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_coupling(q, k, v, seqlen=seqlen)
        elif self.mode == 'vanilla':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v, seqlen=seqlen)
        elif self.mode == 'temporal':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_temporal(q, k, v, seqlen=seqlen, num_sink_tokens=num_sink_tokens)
        elif self.mode == 'spatial':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            x = self.forward_spatial(q, k, v, seqlen=seqlen)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def reshape_T(self, x, seqlen=1, inverse=False):
        if not inverse:
            N, C = x.shape[-2:]
            x = x.reshape(-1, seqlen, self.num_heads, N, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, seqlen * N, C)  # (B, H, TN, c)
        else:
            TN, C = x.shape[-2:]
            x = x.reshape(-1, self.num_heads, seqlen, TN // seqlen, C).transpose(1, 2)
            x = x.reshape(-1, self.num_heads, TN // seqlen, C)  # (BT, H, N, C)
        return x

    def forward_coupling(self, q, k, v, seqlen=8):
        BT, _, N, C = q.shape
        q = self.reshape_T(q, seqlen)
        k = self.reshape_T(k, seqlen)
        v = self.reshape_T(v, seqlen)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = self.reshape_T(x, seqlen, inverse=True)
        x = x.transpose(1, 2).reshape(BT, N, C * self.num_heads)
        return x

    def forward_spatial(self, q, k, v, seqlen=None):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self._record_attention:
            batch = None
            if seqlen is not None and seqlen > 0:
                batch = B // seqlen if B % max(seqlen, 1) == 0 else B
            self._cached_attention['spatial'] = {
                'weights': attn.detach().cpu(),
                'batch': batch,
                'seqlen': seqlen,
                'num_joints': N,
                'num_heads': self.num_heads,
            }
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        return x

    def forward_temporal(self, q, k, v, seqlen=8, num_sink_tokens=0):
        B_q, H_q, N_q, C_q = q.shape # B_q is B_orig * seqlen (actual sequence length of x to qkv)
                                     # H_q is self.num_heads
                                     # N_q is num_joints
                                     # C_q is head_dim
        
        # Calculate B_orig (original batch size before temporal dimension was part of B_q)
        B_orig = B_q // seqlen

        # Reshape q, k, v to expose the temporal dimension (seqlen)
        # Original q, k, v shape: (B_orig * seqlen, H_q, N_q, C_q)
        # Target shape for qt, kt, vt before RoPE processing: (B_orig, H_q, N_q, seqlen, C_q)
        
        qt = q.reshape(B_orig, seqlen, H_q, N_q, C_q).permute(0, 2, 3, 1, 4) # (B_orig, H_q, N_q, seqlen, C_q)
        kt = k.reshape(B_orig, seqlen, H_q, N_q, C_q).permute(0, 2, 3, 1, 4) # (B_orig, H_q, N_q, seqlen, C_q)
        vt = v.reshape(B_orig, seqlen, H_q, N_q, C_q).permute(0, 2, 3, 1, 4) # (B_orig, H_q, N_q, seqlen, C_q)


        # Apply RoPE if enabled (excluding sink tokens from rotation)
        if self.use_rope:
            if num_sink_tokens > 0:
                # Split sink and real tokens along temporal dimension
                # qt shape: (B_orig, H_q, N_q, seqlen, C_q) where seqlen includes sinks
                qt_sinks = qt[:, :, :, :num_sink_tokens, :]
                kt_sinks = kt[:, :, :, :num_sink_tokens, :]
                qt_real = qt[:, :, :, num_sink_tokens:, :]
                kt_real = kt[:, :, :, num_sink_tokens:, :]
                real_seqlen = seqlen - num_sink_tokens

                # Reshape real tokens to apply rotary embeddings: (B_orig*N_q*H_q, real_seqlen, C_q)
                qt_flat = qt_real.permute(0, 2, 1, 3, 4).reshape(-1, real_seqlen, C_q)
                kt_flat = kt_real.permute(0, 2, 1, 3, 4).reshape(-1, real_seqlen, C_q)

                # Apply rotary embeddings only to real tokens
                cos, sin = self.rope(real_seqlen, q.device)
                qt_flat, kt_flat = apply_rotary_pos_emb(qt_flat, kt_flat, cos, sin)

                # Reshape back to (B_orig, H_q, N_q, real_seqlen, C_q)
                qt_real = qt_flat.reshape(B_orig, N_q, H_q, real_seqlen, C_q).permute(0, 2, 1, 3, 4)
                kt_real = kt_flat.reshape(B_orig, N_q, H_q, real_seqlen, C_q).permute(0, 2, 1, 3, 4)

                # Recombine sink and real tokens
                qt = torch.cat([qt_sinks, qt_real], dim=3)
                kt = torch.cat([kt_sinks, kt_real], dim=3)
            else:
                # No sink tokens - apply RoPE to all positions
                qt_flat = qt.permute(0, 2, 1, 3, 4).reshape(-1, seqlen, C_q)
                kt_flat = kt.permute(0, 2, 1, 3, 4).reshape(-1, seqlen, C_q)

                cos, sin = self.rope(seqlen, q.device)
                qt_flat, kt_flat = apply_rotary_pos_emb(qt_flat, kt_flat, cos, sin)

                qt = qt_flat.reshape(B_orig, N_q, H_q, seqlen, C_q).permute(0, 2, 1, 3, 4)
                kt = kt_flat.reshape(B_orig, N_q, H_q, seqlen, C_q).permute(0, 2, 1, 3, 4)

        # Use Flash Attention if enabled and not recording attention
        if self.use_flash_attn and not self._record_attention:
            # Flash Attention path - memory efficient, no materialized attention matrix
            # Reshape to (B_orig*N_q, H_q, seqlen, C_q) for SDPA
            qt_sdpa = qt.permute(0, 2, 1, 3, 4).reshape(B_orig * N_q, H_q, seqlen, C_q)
            kt_sdpa = kt.permute(0, 2, 1, 3, 4).reshape(B_orig * N_q, H_q, seqlen, C_q)
            vt_sdpa = vt.permute(0, 2, 1, 3, 4).reshape(B_orig * N_q, H_q, seqlen, C_q)
            
            # Apply sink token mask if present
            sdpa_mask = None

            x = F.scaled_dot_product_attention(
                qt_sdpa, kt_sdpa, vt_sdpa,
                attn_mask=sdpa_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )
            # x shape: (B_orig*N_q, H_q, seqlen, C_q)
            # Reshape back to (B_orig, N_q, H_q, seqlen, C_q) then to output format
            x = x.reshape(B_orig, N_q, H_q, seqlen, C_q).permute(0, 3, 1, 2, 4)
        else:
            # Standard attention path - used when recording attention or flash_attn disabled
            attn = (qt @ kt.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            if self._record_attention:
                max_frames = 256
                if seqlen > max_frames:
                    # Force a contiguous copy of the slice to free the full tensor reference
                    sliced_attn = attn[:, :, :, :max_frames, :max_frames].contiguous().detach().cpu()
                    recorded_seqlen = max_frames
                else:
                    sliced_attn = attn.detach().cpu()
                    recorded_seqlen = seqlen
                
                self._cached_attention['temporal'] = {
                    'weights': sliced_attn,
                    'seqlen': recorded_seqlen,
                    'num_joints': N_q,
                    'num_heads': H_q,
                }
            attn = self.attn_drop(attn)
            x = attn @ vt  # (B_orig, H_q, N_q, seqlen, C_q)
            # Permute to (B_orig, seqlen, N_q, H_q, C_q)
            x = x.permute(0, 3, 2, 1, 4)

        # Final reshape to output format
        B = B_orig * seqlen
        N = N_q
        C = H_q * C_q
        x = x.reshape(B, N, C)
        return x

    def count_attn(self, attn):
        attn = attn.detach().cpu().numpy()
        attn = attn.mean(axis=1)
        attn_t = attn[:, :, 1].mean(axis=1)
        attn_s = attn[:, :, 0].mean(axis=1)
        if self.attn_count_s is None:
            self.attn_count_s = attn_s
            self.attn_count_t = attn_t
        else:
            self.attn_count_s = np.concatenate([self.attn_count_s, attn_s], axis=0)
            self.attn_count_t = np.concatenate([self.attn_count_t, attn_t], axis=0)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 st_mode='stage_st', att_fuse=False, use_rope=False, use_flash_attn=False):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            st_mode="spatial", use_rope=False, use_flash_attn=False)  # Spatial doesn't use RoPE or Flash Attention
        self.attn_t = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            st_mode="temporal", use_rope=use_rope, use_flash_attn=use_flash_attn)  # Temporal can use RoPE and Flash Attention


        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer,
                         drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer,
                         drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)

    def forward(self, x, seqlen=1, num_sink_tokens=0):
        if self.st_mode == 'stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen, num_sink_tokens=num_sink_tokens))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'stage_ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen, num_sink_tokens=num_sink_tokens))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        elif self.st_mode == 'stage_para':
            x_t = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen, num_sink_tokens=num_sink_tokens))
            x_t = x_t + self.drop_path(self.mlp_t(self.norm2_t(x_t)))
            x_s = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x_s = x_s + self.drop_path(self.mlp_s(self.norm2_s(x_s)))
            if self.att_fuse:
                #             x_s, x_t: [BF, J, dim]
                alpha = torch.cat([x_s, x_t], dim=-1)
                BF, J = alpha.shape[:2]
                # alpha = alpha.mean(dim=1, keepdim=True)
                alpha = self.ts_attn(alpha).reshape(BF, J, -1, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_t * alpha[:, :, :, 1] + x_s * alpha[:, :, :, 0]
            else:
                x = (x_s + x_t) * 0.5
        else:
            raise NotImplementedError(self.st_mode)
        return x

    def set_attention_recording(self, enabled: bool):
        """Toggle attention caching on child attention blocks."""
        if enabled:
            self.attn_s.enable_attention_recording()
            self.attn_t.enable_attention_recording()
        else:
            self.attn_s.disable_attention_recording()
            self.attn_t.disable_attention_recording()

    def pop_attention_cache(self):
        """Retrieve and reset cached spatial and temporal attention."""
        spatial = self.attn_s.pop_attention_cache().get('spatial')
        temporal = self.attn_t.pop_attention_cache().get('temporal')
        return {'spatial': spatial, 'temporal': temporal}


class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_feat=256, dim_rep=512,
                 depth=5, num_heads=8, mlp_ratio=2,
                 num_joints=17, maxlen=243,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, att_fuse=True, use_rope=False, use_flash_attn=False,
                 num_sink_tokens=0, use_grad_checkpointing=False):
        super().__init__()
        self.dim_feat = dim_feat
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn
        self.num_sink_tokens = num_sink_tokens
        # Recompute each (blk_st, blk_ts) pair's activations during backward instead
        # of storing them -- mathematically identical gradients (verified: bit-exact
        # in testing), trades ~30% extra compute for a large activation-memory cut.
        # Off by default; only meaningful during training (self.training), since eval
        # has no backward pass to save memory for.
        self.use_grad_checkpointing = use_grad_checkpointing

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks_st = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_st", use_rope=use_rope, use_flash_attn=use_flash_attn)
            for i in range(depth)])
        self.blocks_ts = nn.ModuleList([
            Block(
                dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode="stage_ts", use_rope=use_rope, use_flash_attn=use_flash_attn)
            for i in range(depth)])

        # Existing initialization code...
        self.norm = norm_layer(dim_feat)
        if dim_rep and dim_rep != 0:
            self.embedding_dim = dim_rep
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh())
            ]))
        else:
            self.embedding_dim = dim_feat
            self.pre_logits = nn.Identity()

        # Keep the spatial positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        trunc_normal_(self.pos_embed, std=.02)

        # Only create temporal embedding if not using RoPE
        if not use_rope:
            self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
            trunc_normal_(self.temp_embed, std=.02)
        else:
            self.register_buffer('temp_embed', None)

        # Attention sink tokens: learned embeddings that absorb excess attention
        # Shape [1, num_sink, 1, dim_feat] broadcasts across batch and joints
        if num_sink_tokens > 0:
            self.sink_tokens = nn.Parameter(torch.zeros(1, num_sink_tokens, 1, dim_feat))
            trunc_normal_(self.sink_tokens, std=.02)
        else:
            self.register_buffer('sink_tokens', None)

        self.apply(self._init_weights)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)

    def _set_attention_recording(self, enabled: bool):
        """Toggle attention caching across all dual-stream stages."""
        for blk in self.blocks_st:
            blk.set_attention_recording(enabled)
        for blk in self.blocks_ts:
            blk.set_attention_recording(enabled)

    def _collect_attention_records(self):
        """Gather cached attention weights from every spatial/temporal block."""
        records = {'stage_st': [], 'stage_ts': []}
        for idx, blk in enumerate(self.blocks_st):
            cached = blk.pop_attention_cache()
            records['stage_st'].append({'block_index': idx, **cached})
        for idx, blk in enumerate(self.blocks_ts):
            cached = blk.pop_attention_cache()
            records['stage_ts'].append({'block_index': idx, **cached})
        return records

    def get_sink_attention_stats(self):
        """Return attention mass flowing to sink tokens across temporal blocks.
        
        Call after forward pass with capture_attention=True. Returns dict with:
        - sink_mass_mean: average attention mass to sinks
        - sink_mass_max: max attention mass to sinks
        - sink_mass_per_block: per-block breakdown
        """
        if self.sink_tokens is None or self.num_sink_tokens == 0:
            return None
        
        stats = {'sink_mass_per_block': {}}
        all_masses = []
        
        for stream_name, blocks in [('st', self.blocks_st), ('ts', self.blocks_ts)]:
            for idx, blk in enumerate(blocks):
                cache = blk.attn_t._cached_attention.get('temporal')
                if cache and cache.get('weights') is not None:
                    attn = cache['weights']  # [B, H, N, T, T] on CPU
                    # Sum attention to sink positions (first num_sink_tokens columns)
                    sink_mass = attn[:, :, :, :, :self.num_sink_tokens].sum(dim=-1).mean().item()
                    stats['sink_mass_per_block'][f'block_{stream_name}_{idx}'] = sink_mass
                    all_masses.append(sink_mass)
        
        if all_masses:
            sorted_masses = sorted(all_masses)
            n = len(sorted_masses)
            stats['sink_mass_mean'] = sum(all_masses) / n
            stats['sink_mass_max'] = max(all_masses)
            stats['sink_mass_median'] = sorted_masses[n // 2]
            stats['sink_mass_p80'] = sorted_masses[int(n * 0.8)]
            stats['sink_mass_std'] = (sum((m - stats['sink_mass_mean'])**2 for m in all_masses) / n) ** 0.5
        else:
            stats['sink_mass_mean'] = 0.0
            stats['sink_mass_max'] = 0.0
            stats['sink_mass_median'] = 0.0
            stats['sink_mass_p80'] = 0.0
            stats['sink_mass_std'] = 0.0
        
        return stats

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return None

    def reset_classifier(self, dim_out, global_pool=''):
        # No classifier head in embedding-only backbone
        return

    def forward(self, x, return_rep=False, capture_attention=False):
        B, F, J, C = x.shape
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        x = x + self.pos_embed

        # Apply temporal embeddings only if not using RoPE
        if not self.use_rope:
            _, J, C = x.shape
            x = x.reshape(-1, F, J, C) + self.temp_embed[:, :F, :, :]
            x = x.reshape(BF, J, C)

        x = self.pos_drop(x)

        # Prepend sink tokens to temporal dimension (before transformer blocks)
        # Sink tokens absorb excess attention and are excluded from RoPE
        F_effective = F
        if self.sink_tokens is not None:
            # sink_tokens: [1, S, 1, dim_feat] -> expand to [B, S, J, dim_feat]
            sinks = self.sink_tokens.expand(B, -1, J, -1)
            # Reshape x back to [B, F, J, C] to prepend sinks
            x = x.reshape(B, F, J, -1)
            x = torch.cat([sinks, x], dim=1)  # [B, F+S, J, C]
            F_effective = F + self.num_sink_tokens
            x = x.reshape(B * F_effective, J, -1)

        # Rest of the existing forward method...
        attention_bundle = None
        if capture_attention:
            self._set_attention_recording(True)
        alphas = []
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            def _run_block_pair(x_in, blk_st=blk_st, blk_ts=blk_ts, idx=idx):
                x_st = blk_st(x_in, F_effective, num_sink_tokens=self.num_sink_tokens)
                x_ts = blk_ts(x_in, F_effective, num_sink_tokens=self.num_sink_tokens)
                if self.att_fuse:
                    att = self.ts_attn[idx]
                    alpha = torch.cat([x_st, x_ts], dim=-1)
                    alpha = att(alpha)
                    alpha = alpha.softmax(dim=-1)
                    return x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
                else:
                    return (x_st + x_ts) * 0.5

            if self.use_grad_checkpointing and self.training:
                x = checkpoint.checkpoint(_run_block_pair, x, use_reentrant=False)
            else:
                x = _run_block_pair(x)

        # Strip sink tokens before output (they should not be in embeddings)
        sink_embeddings = None
        if self.sink_tokens is not None:
            x = x.reshape(B, F_effective, J, -1)
            sink_embeddings = x[:, :self.num_sink_tokens, :, :]  # [B, S, J, C]
            x = x[:, self.num_sink_tokens:, :, :]  # Remove sink tokens
            x = x.reshape(B * F, J, -1)

        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        embeddings = self.pre_logits(x)  # [B, F, J, embedding_dim]
        bundle = {
            'pre_logits_embeddings': embeddings,
            'block_embeddings': x,
            'sink_embeddings': sink_embeddings,  # [B, S, J, C] or None if no sink tokens
        }
        if capture_attention:
            attention_bundle = self._collect_attention_records()
            self._set_attention_recording(False)
            bundle['attention_maps'] = attention_bundle
        return bundle

    def get_representation(self, x):
        return self.forward(x)
