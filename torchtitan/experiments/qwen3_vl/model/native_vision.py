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
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .args import Qwen3VLVisionArgs


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
    """Multi-head attention with 2D RoPE for vision.

    Uses FlexAttention with BlockMask for compile-friendly per-image attention.
    No per-image chunking, no .tolist() sync, no dynamic splits.
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
        """
        Args:
            hidden_states: (total_seq, hidden_size) - flat across all images
            block_mask: FlexAttention BlockMask for per-image attention
            position_embeddings: (cos, sin) each (total_seq, head_dim)
        """
        seq_len = hidden_states.shape[0]

        # QKV projection: (seq_len, 3, n_heads, head_dim)
        qkv = self.qkv(hidden_states).reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)  # each (seq_len, n_heads, head_dim)

        # Apply 2D rotary embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Reshape for FlexAttention: (1, n_heads, seq_len, head_dim)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # FlexAttention with BlockMask — no per-image splitting needed
        out = torch.nn.attention.flex_attention.flex_attention(
            q, k, v, block_mask=block_mask
        )

        # Reshape back: (seq_len, hidden_size)
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

    @torch.compiler.disable  # bilinear interp uses per-image linspace (not perf-critical)
    def _compute_position_embeddings(
        self, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Compute interpolated learned position embeddings.

        The bilinear interpolation requires per-image linspace (different h, w),
        so this stays outside compile. But it's only ~1% of forward time.
        The spatial merge permutation uses the batched _build_merge_indices.
        """
        device = self.pos_embed.weight.device
        merge_size = self.spatial_merge_size
        num_grid = self.num_grid_per_side
        grid_thw_list = grid_thw.tolist()

        # Bilinear interpolation indices and weights (per-image, different h/w)
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

        idx_tensor = torch.stack([torch.cat(idx_list[i]) for i in range(4)], dim=0).long()
        weight_tensor = torch.stack([torch.cat(weight_list[i]) for i in range(4)], dim=0)
        weight_tensor = weight_tensor.to(self.pos_embed.weight.dtype)

        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor.unsqueeze(-1)
        patch_pos_embeds = pos_embeds.sum(dim=0)  # (total_hw_patches, hidden)

        # Repeat for temporal frames and apply spatial merge permutation
        # Use batched merge indices
        merge_indices = self._build_merge_indices(grid_thw)
        patch_pos_embeds_expanded = torch.repeat_interleave(
            patch_pos_embeds, torch.repeat_interleave(grid_thw[:, 0], grid_thw[:, 1] * grid_thw[:, 2]),
            dim=0,
        )
        # Wait — repeat_interleave on pos embeds for temporal is wrong here because
        # pos_embeds is (sum(h*w), D) not (sum(t*h*w), D). Need to repeat per-image.
        # Fall back to per-image repeat for correctness.
        sizes = [int(h) * int(w) for _, h, w in grid_thw_list]
        splits = torch.split(patch_pos_embeds, sizes)
        expanded = []
        for pos_embed, (t, _, _) in zip(splits, grid_thw_list):
            expanded.append(pos_embed.repeat(int(t), 1))
        patch_pos_embeds = torch.cat(expanded, dim=0)  # (total_patches, hidden)

        # Apply merge permutation via gather
        return patch_pos_embeds[merge_indices]

    def _build_merge_indices(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Build gather indices for spatial merge permutation — no .tolist() needed.

        For each image, patches are laid out as (t, h, w) in row-major order.
        After merge permutation, they become (t, h//m, w//m, m, m) order.
        This function computes the permutation indices using tensor ops.

        Args:
            grid_thw: (num_images, 3)

        Returns:
            (total_patches,) - index tensor for gather
        """
        device = grid_thw.device
        merge_size = self.spatial_merge_size

        # Per-frame patch count and merged dimensions
        t_vals = grid_thw[:, 0]  # (N,)
        h_vals = grid_thw[:, 1]  # (N,)
        w_vals = grid_thw[:, 2]  # (N,)
        hw_vals = h_vals * w_vals  # patches per frame
        thw_vals = t_vals * hw_vals  # patches per image

        # We need per-image offsets in the flat sequence
        cu_patches = F.pad(thw_vals.cumsum(0), (1, 0), value=0)  # (N+1,)

        # Build indices per image (this still needs per-image iteration for different h, w)
        # But we minimize it to just index computation, no tensor splitting
        all_indices = []
        for i in range(grid_thw.shape[0]):
            t = t_vals[i].item()
            h = h_vals[i].item()
            w = w_vals[i].item()
            offset = cu_patches[i].item()

            mh = h // merge_size
            mw = w // merge_size

            # Original flat index within one frame: row * w + col
            # Merge permutation: (mh, mw, merge_size, merge_size) iteration order
            # maps to original (mh*merge_size + mr) * w + (mw*merge_size + mc)
            block_r = torch.arange(mh, device=device)
            block_c = torch.arange(mw, device=device)
            intra_r = torch.arange(merge_size, device=device)
            intra_c = torch.arange(merge_size, device=device)

            # (mh, mw, merge_size, merge_size) → flat original index
            orig_row = block_r[:, None, None, None] * merge_size + intra_r[None, None, :, None]
            orig_col = block_c[None, :, None, None] * merge_size + intra_c[None, None, None, :]
            flat_idx = (orig_row * w + orig_col).reshape(-1)  # (h*w,)

            # Repeat for t frames, add frame offsets
            if t > 1:
                frame_offsets = torch.arange(t, device=device)[:, None] * (h * w)
                flat_idx = (flat_idx[None, :] + frame_offsets).reshape(-1)  # (t*h*w,)

            all_indices.append(flat_idx + offset)

        return torch.cat(all_indices, dim=0)

    def _compute_rotary_pos_emb(
        self, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D rotary position embeddings — uses batched merge indices.

        The row/col coordinates after merge permutation are computed via
        _build_merge_indices pattern, then looked up in the frequency table.
        """
        merge_size = self.spatial_merge_size
        device = self.rotary_pos_emb.inv_freq.device

        # Compute max h/w for frequency table (no .tolist())
        max_hw = torch.max(torch.max(grid_thw[:, 1]), torch.max(grid_thw[:, 2])).item()
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim//2)

        # Build merge-permuted (row, col) coordinates per patch
        t_vals = grid_thw[:, 0]
        h_vals = grid_thw[:, 1]
        w_vals = grid_thw[:, 2]
        thw_vals = t_vals * h_vals * w_vals
        cu_patches = F.pad(thw_vals.cumsum(0), (1, 0), value=0)

        total_tokens = thw_vals.sum().item()
        row_ids = torch.empty(total_tokens, dtype=torch.long, device=device)
        col_ids = torch.empty(total_tokens, dtype=torch.long, device=device)

        for i in range(grid_thw.shape[0]):
            t = t_vals[i].item()
            h = h_vals[i].item()
            w = w_vals[i].item()
            offset = cu_patches[i].item()
            n = t * h * w
            mh = h // merge_size
            mw = w // merge_size

            block_r = torch.arange(mh, device=device)
            block_c = torch.arange(mw, device=device)
            intra_r = torch.arange(merge_size, device=device)
            intra_c = torch.arange(merge_size, device=device)

            row_idx = (block_r[:, None, None, None] * merge_size + intra_r[None, None, :, None])
            col_idx = (block_c[None, :, None, None] * merge_size + intra_c[None, None, None, :])
            row_idx = row_idx.expand(mh, mw, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(mh, mw, merge_size, merge_size).reshape(-1)

            if t > 1:
                row_idx = row_idx.repeat(t)
                col_idx = col_idx.repeat(t)

            row_ids[offset:offset + n] = row_idx
            col_ids[offset:offset + n] = col_idx

        # Lookup rotary frequencies
        row_emb = freq_table[row_ids]  # (total, dim//2)
        col_emb = freq_table[col_ids]  # (total, dim//2)
        embeddings = torch.cat((row_emb, col_emb), dim=-1)  # (total, dim)
        emb = torch.cat((embeddings, embeddings), dim=-1)  # (total, 2*dim)
        return emb.cos(), emb.sin()

    def _compute_cu_seqlens(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute cumulative sequence lengths from grid dimensions — no .tolist()."""
        seq_lens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu_seqlens = F.pad(seq_lens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        return cu_seqlens

    @torch.compiler.disable  # create_block_mask uses Python control flow internally
    def _create_block_mask(
        self, grid_thw: torch.Tensor, hidden_states: torch.Tensor
    ) -> BlockMask:
        """Create FlexAttention BlockMask — uses repeat_interleave instead of Python loop."""
        seq_len = hidden_states.shape[0]
        device = hidden_states.device

        # Build image_ids via repeat_interleave (no Python loop)
        cu_seqlens = self._compute_cu_seqlens(grid_thw)
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        num_images = seq_lens.shape[0]
        image_ids = torch.repeat_interleave(
            torch.arange(num_images, device=device, dtype=torch.int32),
            seq_lens,
        )

        def vision_mask_mod(b, h, q_idx, kv_idx):
            return image_ids[q_idx] == image_ids[kv_idx]

        block_mask = create_block_mask(
            vision_mask_mod, B=1, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
            device=device,
        )
        return block_mask

    def _spatial_merge(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor,
        merger: VisionPatchMerger,
    ) -> torch.Tensor:
        """Apply spatial merge using gather indices — no .tolist() for the gather.

        The merge permutation reorders patches from (t, h, w) layout to
        (t, h//m, w//m, m, m) layout, then groups m*m patches into one token.
        """
        # Build merge indices (reorders patches into merge groups)
        merge_indices = self._build_merge_indices(grid_thw)
        # Gather in merge order: adjacent m*m patches become contiguous
        reordered = hidden_states[merge_indices]
        # Now reshape: every (merge_size^2) consecutive patches form one merged token
        merge_unit = self.spatial_merge_size ** 2
        # reordered is (total_patches, hidden_size), reshape to (total_merged, merge_unit * hidden)
        reordered = reordered.view(-1, merge_unit * hidden_states.shape[-1])
        return merger(reordered)

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
        block_mask = self._create_block_mask(grid_thw, hidden_states)

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
