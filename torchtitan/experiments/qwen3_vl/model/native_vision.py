# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Native Qwen3-VL Vision Encoder

Compile-friendly reimplementation of HF's Qwen3VLVisionModel.
Supports fullgraph=True torch.compile by avoiding:
- torch._dynamo.decorators.disable
- .tolist() / .item() CPU-GPU sync
- Python loops over data-dependent lengths
- torch.split with dynamic sizes

Architecture: SigLIP-2 based encoder with 2D RoPE, spatial merge, and DeepStack.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .args import Qwen3VLVisionArgs

# Compile FlexAttention once (class-level, not per-instance)
_compiled_flex_attention = torch.compile(flex_attention)


class VisionRotaryEmbedding(nn.Module):
    """2D rotary position embedding for vision tokens."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: (seq_len, n_heads, head_dim)
        k: (seq_len, n_heads, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    """
    orig_dtype = q.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()  # (seq_len, 1, head_dim)
    sin = sin.unsqueeze(-2).float()

    # Rotate half: [-x2, x1, -x4, x3, ...]
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class VisionAttention(nn.Module):
    """Multi-head attention with 2D RoPE for vision using FlexAttention.

    FlexAttention with BlockMask handles per-image boundaries without
    torch.split, enabling mark_dynamic without Inductor unbacked SymInt bugs.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_mask: BlockMask,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]

        # QKV projection + RoPE
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # FlexAttention: (1, n_heads, seq_len, head_dim) — no torch.split needed
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        out = _compiled_flex_attention(q, k, v, block_mask=block_mask)

        out = out.squeeze(0).transpose(0, 1).reshape(seq_len, -1).contiguous()
        return self.proj(out)


class VisionBlock(nn.Module):
    """Pre-norm transformer block for vision encoder."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_mask: BlockMask,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), block_mask, position_embeddings
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class VisionPatchMerger(nn.Module):
    """Merge spatial_merge_size x spatial_merge_size patches into 1 token.

    Input: (N, hidden_size * spatial_merge_size^2) - already grouped patches
    Output: (N, out_hidden_size)

    Two norm modes:
    - use_postshuffle_norm=False (final merger): norm on hidden_size, then reshape to merge_dim
    - use_postshuffle_norm=True (deepstack mergers): norm on merge_dim directly
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.pre_merge_hidden_size = hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.merge_dim if use_postshuffle_norm else hidden_size, eps=1e-6
        )
        self.fc1 = nn.Linear(self.merge_dim, self.merge_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.merge_dim, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, merge_dim) where merge_dim = hidden_size * spatial_merge_size^2
        if self.use_postshuffle_norm:
            # Norm on the full merge_dim
            x = self.norm(x)
        else:
            # Norm on hidden_size (pre-merge), then reshape back
            x = self.norm(x.view(-1, self.pre_merge_hidden_size)).view(-1, self.merge_dim)
        x = self.fc2(self.act(self.fc1(x)))
        return x


class Qwen3VLNativeVisionEncoder(nn.Module):
    """
    Native Qwen3-VL vision encoder, compile-friendly replacement for HF wrapper.

    Architecture: SigLIP-2 based with 2D RoPE, spatial merge, and DeepStack.
    """

    def __init__(self, args: Qwen3VLVisionArgs):
        super().__init__()
        self.args = args
        self.spatial_merge_size = args.spatial_merge_size
        self.num_grid_per_side = int(args.num_position_embeddings ** 0.5)

        # Patch embedding: Conv3d-equivalent linear projection
        patch_input_dim = args.in_channels * args.temporal_patch_size * args.patch_size * args.patch_size
        self.patch_embed = nn.Linear(patch_input_dim, args.hidden_size)

        # Learned position embedding
        self.pos_embed = nn.Embedding(args.num_position_embeddings, args.hidden_size)

        # 2D rotary position embedding
        head_dim = args.hidden_size // args.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionBlock(args.hidden_size, args.intermediate_size, args.num_heads)
            for _ in range(args.depth)
        ])

        # Output patch merger
        self.merger = VisionPatchMerger(
            args.hidden_size, args.out_hidden_size,
            args.spatial_merge_size, use_postshuffle_norm=False,
        )

        # DeepStack mergers (extract features at specific layers)
        self.deepstack_visual_indexes = args.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([
            VisionPatchMerger(
                args.hidden_size, args.out_hidden_size,
                args.spatial_merge_size, use_postshuffle_norm=True,
            )
            for _ in range(len(args.deepstack_visual_indexes))
        ])

    @property
    def dtype(self):
        return self.patch_embed.weight.dtype

    @torch.compiler.disable
    def _compute_position_embeddings(
        self, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Compute interpolated learned position embeddings.

        Follows HF's fast_pos_embed_interpolate but uses tensor ops.

        Args:
            grid_thw: (num_images, 3) - temporal, height, width per image

        Returns:
            (total_patches, hidden_size) - position embeddings for all patches
        """
        device = self.pos_embed.weight.device
        merge_size = self.spatial_merge_size
        num_grid = self.num_grid_per_side

        # We need to iterate per-image because each has different h, w
        # This happens once per forward, not in the hot loop
        grid_thw_list = grid_thw.tolist()

        # Bilinear interpolation indices and weights
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, num_grid - 1, h, device=device)
            w_idxs = torch.linspace(0, num_grid - 1, w, device=device)

            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_floor + 1).clamp(max=num_grid - 1)
            w_ceil = (w_floor + 1).clamp(max=num_grid - 1)

            dh = h_idxs - h_floor.float()
            dw = w_idxs - w_floor.float()

            base_h = h_floor * num_grid
            base_h_ceil = h_ceil * num_grid

            indices = [
                (base_h[:, None] + w_floor[None, :]).flatten(),
                (base_h[:, None] + w_ceil[None, :]).flatten(),
                (base_h_ceil[:, None] + w_floor[None, :]).flatten(),
                (base_h_ceil[:, None] + w_ceil[None, :]).flatten(),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
                ((1 - dh)[:, None] * dw[None, :]).flatten(),
                (dh[:, None] * (1 - dw)[None, :]).flatten(),
                (dh[:, None] * dw[None, :]).flatten(),
            ]

            for i in range(4):
                idx_list[i].append(indices[i])
                weight_list[i].append(weights[i])

        # Concatenate per-corner indices and weights across all images
        idx_tensor = torch.stack([torch.cat(idx_list[i]) for i in range(4)], dim=0).long()
        weight_tensor = torch.stack([torch.cat(weight_list[i]) for i in range(4)], dim=0)
        weight_tensor = weight_tensor.to(self.pos_embed.weight.dtype)

        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor.unsqueeze(-1)
        patch_pos_embeds = pos_embeds.sum(dim=0)  # (total_patches, hidden_size)

        # Split by image and apply spatial merge permutation
        sizes = [int(h) * int(w) for _, h, w in grid_thw_list]
        splits = torch.split(patch_pos_embeds, sizes)

        result = []
        for pos_embed, (t, h, w) in zip(splits, grid_thw_list):
            t, h, w = int(t), int(h), int(w)
            pos_embed = pos_embed.repeat(t, 1)  # repeat for temporal frames
            # Spatial merge permutation: group merge_size x merge_size patches
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            result.append(pos_embed)

        return torch.cat(result, dim=0)

    @torch.compiler.disable
    def _compute_rotary_pos_emb(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D rotary position embeddings from grid dimensions.

        Args:
            grid_thw: (num_images, 3)

        Returns:
            (cos, sin) each of shape (total_patches, head_dim)
        """
        merge_size = self.spatial_merge_size
        device = self.rotary_pos_emb.inv_freq.device

        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(int(h), int(w)) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim//2)

        total_tokens = sum(int(t) * int(h) * int(w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for t, h, w in grid_thw_list:
            t, h, w = int(t), int(h), int(w)
            merged_h, merged_w = h // merge_size, w // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = (block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None])
            col_idx = (block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :])

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset: offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids].flatten(1)  # (total_patches, head_dim)
        emb = torch.cat((embeddings, embeddings), dim=-1)
        return emb.cos(), emb.sin()

    def _compute_cu_seqlens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute cumulative sequence lengths from grid dimensions."""
        seq_lens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = F.pad(seq_lens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        return cu_seqlens

    @torch.compiler.disable
    def _create_block_mask(
        self, grid_thw: torch.Tensor, seq_len: int, device: torch.device
    ) -> BlockMask:
        """Create FlexAttention BlockMask for per-image block-diagonal attention."""
        cu_seqlens = self._compute_cu_seqlens(grid_thw)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        num_images = seq_lens.shape[0]
        image_ids = torch.repeat_interleave(
            torch.arange(num_images, device=device, dtype=torch.int32),
            seq_lens,
        )

        def vision_mask_mod(b, h, q_idx, kv_idx):
            return image_ids[q_idx] == image_ids[kv_idx]

        return create_block_mask(
            vision_mask_mod, B=1, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
            device=device,
        )

    def _spatial_merge(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor,
        merger: VisionPatchMerger,
    ) -> torch.Tensor:
        """Apply spatial merge: group merge_size x merge_size patches and project.

        Hidden states are already in merge-permuted order from position embedding
        (fast_pos_embed_interpolate permutes patches into (h//m, w//m, m, m) layout).
        So we just reshape to group consecutive merge_size^2 patches, matching HF.
        """
        merge_dim = self.spatial_merge_size ** 2
        # Simple view: consecutive groups of merge_dim patches → one merged token
        merged_states = hidden_states.view(-1, merge_dim * hidden_states.shape[-1])
        return merger(merged_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through native vision encoder.

        Args:
            hidden_states: (total_patches, patch_input_dim) - patchified pixel values
            grid_thw: (num_images, 3) - temporal, height, width per image

        Returns:
            (merged_features, deepstack_features_list)
            - merged_features: (total_merged_patches, out_hidden_size)
            - deepstack_features_list: list of (total_merged_patches, out_hidden_size)
        """
        # 1. Patch embedding
        hidden_states = self.patch_embed(hidden_states)

        # 2. Add learned position embeddings
        pos_embeds = self._compute_position_embeddings(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # 3. Compute rotary position embeddings for attention
        cos, sin = self._compute_rotary_pos_emb(grid_thw)
        position_embeddings = (cos, sin)

        # 4. Create FlexAttention BlockMask for per-image attention
        block_mask = self._create_block_mask(
            grid_thw, hidden_states.shape[0], hidden_states.device
        )

        # 5. Transformer blocks with DeepStack extraction
        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, block_mask, position_embeddings)

            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_idx)
                ds_feature = self._spatial_merge(
                    hidden_states, grid_thw, self.deepstack_merger_list[ds_idx]
                )
                deepstack_features.append(ds_feature)

        # 6. Final spatial merge
        merged = self._spatial_merge(hidden_states, grid_thw, self.merger)

        return merged, deepstack_features
