# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL MOE Model Implementation - Option B: Extend Qwen3Model

Architecture:
- Qwen3VLTextModel: Extends Qwen3Model, adds DeepStack injection
- Qwen3VLModel: Combines vision encoder + extended text model

Reference HF Implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modular_qwen3_vl.py
"""

from typing import Optional

import torch
from torch import nn

from torchtitan.models.qwen3.model.model import Qwen3Model
from torchtitan.models.qwen3.model.model import \
    apply_rotary_emb as qwen3_apply_rotary_emb

from .args import Qwen3VLModelArgs
from .native_vision import Qwen3VLNativeVisionEncoder as Qwen3VLVisionEncoder


def apply_rotary_emb_mrope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    mrope_section: list[int] = [24, 20, 20],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply interleaved multi-resolution RoPE for Qwen3-VL.

    Matches HF's Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope:
    - Computes per-dimension frequencies from 3D position IDs
    - Interleaves T/H/W frequencies: [THWTHWTHW...TT]
    - mrope_section defines how many frequency pairs per dimension

    For text-only (2D positions), falls back to standard RoPE.
    """
    if positions is not None and positions.dim() == 3:
        assert positions.shape[0] == 3
        bz, seqlen, n_heads, head_dim = xq.shape

        # rope_cache: (max_seq_len, head_dim * 2) with interleaved [cos, sin]
        # Split into cos and sin halves
        half_dim = head_dim // 2  # = 64 for head_dim=128
        # rope_cache layout: (max_seq_len, head_dim) where first half_dim is one set

        # Compute frequencies for each dimension using the shared inv_freq
        # inv_freq has dim head_dim//2 entries, position_ids are (3, bz, seqlen)
        # We need to compute freqs = inv_freq * position_ids for each of 3 dims
        # Then interleave according to mrope_section

        # Extract inv_freq from rope_cache (row 0 and row 1 give us the base)
        # Actually, rope_cache is precomputed as (seq, head_dim*2) where
        # rope_cache[pos] = [cos(freq*pos), sin(freq*pos)] interleaved
        # We need to recompute from position_ids, not use positional lookup

        # Use the same approach as HF: compute freqs from inv_freq * positions
        # inv_freq shape: (head_dim // 2,)
        # We can extract it from rope_cache: at position 1, cos(freq*1) = cos(freq)
        # But it's cleaner to just use the rope_cache lookup per dimension

        # For each position dim, look up rope_cache at those positions
        # rope_cache[pos] gives (head_dim * 2,) = [cos0, sin0, cos1, sin1, ...]
        # We need the first head_dim values (cos) and second head_dim values (sin)

        # Actually rope_cache in TorchTitan Qwen3 is:
        # shape (max_seq, head_dim) with complex-like layout
        # Let me check the actual format...

        # The simplest correct approach: compute cos/sin per dimension,
        # then interleave and apply, matching HF exactly.

        # Step 1: For each T/H/W dimension, gather rope values at those positions
        # positions: (3, bz, seqlen)
        # rope_cache: (max_seq_len, D) where D = head_dim (cos part) or head_dim*2

        # rope_cache: (max_seq, head_dim * 2) where first head_dim = cos, second = sin
        # cos_cache[pos] has head_dim values: cos(freq_0 * pos), cos(freq_1 * pos), ...
        # But rope_cache was built with cat([theta, theta]) before cos/sin,
        # so cos_cache[pos] = [cos(f0*p), cos(f1*p), ..., cos(f63*p), cos(f0*p), ..., cos(f63*p)]
        # i.e., it's doubled: first half_dim entries = second half_dim entries
        cos_cache = rope_cache[:, :head_dim]   # (max_seq, head_dim=128)
        sin_cache = rope_cache[:, head_dim:]   # (max_seq, head_dim=128)
        # Only need half_dim unique values (first 64 of the 128)
        half_dim = head_dim // 2
        cos_cache_half = cos_cache[:, :half_dim]  # (max_seq, 64)
        sin_cache_half = sin_cache[:, :half_dim]

        # Step 2: Gather per-dimension frequencies at position IDs
        freqs_cos = []
        freqs_sin = []
        for d in range(3):
            pos_d = positions[d].long()  # (bz, seqlen)
            freqs_cos.append(cos_cache_half[pos_d])  # (bz, seqlen, 64)
            freqs_sin.append(sin_cache_half[pos_d])

        # Step 3: Interleave per HF's apply_interleaved_mrope
        # In the half_dim=64 frequency array:
        #   Start with T (dim 0), then overwrite H (dim 1) and W (dim 2) at interleaved positions
        cos_interleaved = freqs_cos[0].clone()
        sin_interleaved = freqs_sin[0].clone()
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            cos_interleaved[..., idx] = freqs_cos[dim_idx][..., idx]
            sin_interleaved[..., idx] = freqs_sin[dim_idx][..., idx]

        # Step 4: Apply rotary embedding
        # Expand to full head_dim: [cos_half, cos_half] to match rotate_half pattern
        cos_full = torch.cat([cos_interleaved, cos_interleaved], dim=-1)  # (bz, seqlen, head_dim=128)
        sin_full = torch.cat([sin_interleaved, sin_interleaved], dim=-1)
        cos_full = cos_full.unsqueeze(2)  # (bz, seqlen, 1, head_dim)
        sin_full = sin_full.unsqueeze(2)

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        xq_out = (xq * cos_full) + (rotate_half(xq) * sin_full)
        xk_out = (xk * cos_full) + (rotate_half(xk) * sin_full)

        return xq_out, xk_out
    else:
        # Standard RoPE for text-only
        return qwen3_apply_rotary_emb(xq, xk, rope_cache, positions)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_placeholder_mask(
    input_ids: torch.LongTensor,
    inputs_embeds: torch.FloatTensor,
    image_features: Optional[torch.FloatTensor] = None,
    video_features: Optional[torch.FloatTensor] = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Find positions of image/video tokens and create scatter masks.
    
    HF Reference: Lines 1221-1250 in modeling_qwen3_vl.py
    """
    # Find image token positions — no CPU sync needed
    special_image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    # Find video token positions — no CPU sync needed
    special_video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    return special_image_mask, special_video_mask


def get_rope_index(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    spatial_merge_size: int = 2,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate 3D position IDs for vision tokens and 1D for text tokens.
    
    Qwen3VL uses timestamps to separate video frames, and 3D RoPE (T, H, W) for vision.
    
    HF Reference: Lines 1078-1175 in modeling_qwen3_vl.py
    """
    # Handle video frame splitting - use timestamps rather than absolute time
    # Each frame of a video is treated separately with T=1
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1  # Each frame treated separately
    
    mrope_position_deltas = []
    
    # If no vision inputs, fall back to simple 1D position IDs
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        # Initialize position_ids: (3, batch_size, seq_len) for T, H, W dimensions
        batch_size, seq_len = input_ids.shape
        position_ids = torch.ones(
            3, batch_size, seq_len,
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(input_ids.device)
        
        image_index, video_index = 0, 0
        
        # Process each sample in batch
        for i in range(batch_size):
            # Get valid tokens for this sample
            sample_input_ids = input_ids[i][attention_mask[i] == 1]
            
            # Count images and videos in this sample
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            
            # Build position IDs by scanning through sequence
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            
            # Process each vision token
            for _ in range(image_nums + video_nums):
                # Find next image token position
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                
                # Find next video token position
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                
                # Determine which comes first
                if ed_image < ed_video:
                    # Process image
                    t, h, w = image_grid_thw[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    # Process video
                    t, h, w = video_grid_thw[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                
                # Calculate grid dimensions after spatial merge
                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                
                # Add text positions before vision token
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
                
                # Add 3D vision positions (T, H, W)
                # t_index is always 0 because llm_grid_t is always 1 for each frame
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                
                # Move past the vision token
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            
            # Add remaining text positions
            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
            
            # Concatenate all positions
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            
            # Calculate position delta (for generation)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[i]))
        
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        
        return position_ids, mrope_position_deltas
    
    else:
        # No vision inputs - simple 1D positions
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        
        return position_ids, mrope_position_deltas


# ============================================================================
# EXTENDED TEXT MODEL WITH DEEPSTACK
# ============================================================================

class Qwen3VLTextModel(Qwen3Model):
    """
    Extends Qwen3Model to add DeepStack visual feature injection.
    
    Inherits all Qwen3Model functionality and adds:
    - _deepstack_process(): Injects visual features at specific positions
    - Modified forward(): Accepts visual_pos_masks and deepstack_visual_embeds
    - Uses multi-resolution RoPE (MroPE) for 3D positional encoding
    
    This matches HF's Qwen3VLTextModel which extends Qwen3Model.
    HF Reference: Lines ~650-750 in modular_qwen3_vl.py
    """
    
    def __init__(self, model_args: Qwen3VLModelArgs):
        # Initialize parent Qwen3Model
        super().__init__(model_args)
        
        # Patch all attention layers to use MroPE instead of standard RoPE
        self._patch_attention_layers()
    
    def _patch_attention_layers(self):
        """Patch attention layers to use apply_rotary_emb_mrope instead of apply_rotary_emb."""
        from torchtitan.models.qwen3.model import model as qwen3_model_module

        # Temporarily replace the apply_rotary_emb function in the qwen3 module
        # This way, all attention layers will use MroPE
        qwen3_model_module.apply_rotary_emb = apply_rotary_emb_mrope
    
    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject visual features into decoder hidden states at masked positions.
        
        Args:
            hidden_states: Decoder hidden states (batch_size, seq_len, hidden_size)
            visual_pos_masks: Boolean mask indicating visual token positions (batch_size, seq_len)
            visual_embeds: Visual features to inject (num_visual_tokens, hidden_size)
            
        Returns:
            Updated hidden states with visual features injected
            
        HF Reference: Lines ~680-690 in modular_qwen3_vl.py
        """
        # Cast to correct device/dtype
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        # Add visual features at masked positions.
        # Use masked_scatter to place flat visual_embeds at mask positions, then add.
        # (masked_scatter is the most efficient primitive for flat→sparse scatter)
        mask_3d = visual_pos_masks.unsqueeze(-1).expand_as(hidden_states)
        visual_add = hidden_states.new_zeros(hidden_states.shape)
        visual_add.masked_scatter_(mask_3d, visual_embeds)

        return hidden_states + visual_add
    
    def forward(
        self,
        tokens: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks,
        positions: Optional[torch.Tensor] = None,
        # NEW PARAMETERS FOR VISION:
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ):
        """
        Extended forward pass with DeepStack visual feature injection.
        
        Calls parent Qwen3Model.forward() but injects visual features after
        specific decoder layers as specified by deepstack_visual_embeds.
        
        Args:
            tokens: Input token IDs (or None if using embeddings directly)
            rope_cache: Precomputed RoPE frequencies
            attention_masks: Attention masks
            positions: Position IDs for RoPE
            visual_pos_masks: Boolean mask for visual token positions (NEW)
            deepstack_visual_embeds: List of visual features per layer (NEW)
            
        Returns:
            Model outputs (logits or hidden states)
            
        HF Reference: Lines ~810-870 in modular_qwen3_vl.py
        """
        # Step 1: Get embeddings
        # In VL mode, tokens are already embeddings (from Qwen3VLModel.forward)
        # In text-only mode, tokens would be IDs but VL always provides embeddings
        h = tokens
        
        # Step 2: Process through layers with DeepStack injection
        for layer_idx, layer in enumerate(self.layers.values()):
            # Standard layer forward
            h = layer(h, rope_cache, attention_masks, positions)
            
            # NEW: Inject visual features after specific layers
            # deepstack_visual_embeds is a list where each element corresponds to a layer
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                h = self._deepstack_process(
                    h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
        
        # Step 3: Apply final norm and output projection
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        
        return output


# ============================================================================
# MAIN VL MODEL
# ============================================================================

class Qwen3VLModel(nn.Module):
    """
    Main Qwen3 VL model combining vision encoder with extended text model.
    
    Components:
    - visual: Qwen3VLVisionEncoder for image/video encoding
    - language_model: Qwen3VLTextModel (extended with DeepStack)
    
    HF Reference: Lines ~750-900 in modular_qwen3_vl.py
    """
    
    def __init__(self, model_args: Qwen3VLModelArgs):
        super().__init__()
        self.model_args = model_args
        
        # Vision encoder
        self.visual = Qwen3VLVisionEncoder(model_args.vision_config)
        
        # Extended text decoder (Qwen3VLTextModel, not base Qwen3Model!)
        self.language_model = Qwen3VLTextModel(model_args)
        
        # Cache for generation
        self.rope_deltas = None
    
    def get_input_embeddings(self):
        return self.language_model.tok_embeddings
    
    def set_input_embeddings(self, value):
        self.language_model.tok_embeddings = value
    
    def init_weights(self, buffer_device=None):
        """
        Initialize model weights. Delegates to language model's init_weights.
        Vision encoder weights are typically loaded from pretrained checkpoint.
        """
        if hasattr(self.language_model, 'init_weights'):
            self.language_model.init_weights(buffer_device=buffer_device)
    
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Encode images via vision encoder and return embeddings + deepstack features.
        
        HF Reference: Lines 1196-1202 in modeling_qwen3_vl.py
        """
        # Cast to vision encoder dtype
        pixel_values = pixel_values.type(self.visual.dtype)

        # Mark dim 0 as dynamic before entering compiled vision encoder.
        # Tells compiler: total_patches varies, but hidden dims are static.
        torch._dynamo.maybe_mark_dynamic(pixel_values, 0)
        torch._dynamo.maybe_mark_dynamic(image_grid_thw, 0)

        # Forward through vision encoder - returns (embeddings, deepstack_features)
        image_embeds, deepstack_image_embeds = self.visual(
            pixel_values, grid_thw=image_grid_thw
        )
        
        # Return flat embeddings directly (caller will use as-is)
        # No need to split by grid sizes — the flat tensor is ready for masked_scatter
        return image_embeds, deepstack_image_embeds
    
    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        """Videos processed same as images."""
        return self.get_image_features(pixel_values_videos, video_grid_thw)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Main forward pass combining vision and text.
        
        Steps:
        1. Get text embeddings
        2. Encode vision (images/videos) → get embeddings + deepstack features
        3. Scatter vision embeddings into text positions (get_placeholder_mask)
        4. Calculate position IDs (get_rope_index)
        5. Forward through Qwen3VLTextModel with vision parameters
        
        HF Reference: Lines 1253-1334 in modeling_qwen3_vl.py
        """
        # Step 1: Get text embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        
        # Step 2: Encode and scatter vision features
        image_mask = None
        video_mask = None
        deepstack_visual_embeds = None
        
        # Process images if provided
        if pixel_values is not None:
            # Encode images through vision encoder
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )
            # Ensure correct device/dtype
            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            
            # Get scatter mask for image positions
            image_mask, _ = get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds
            )
            
            # Scatter image embeddings into text embedding positions
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            deepstack_visual_embeds = deepstack_image_embeds
        
        # Process videos if provided
        if pixel_values_videos is not None:
            # Encode videos (same as images)
            video_embeds, deepstack_video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            # Ensure correct device/dtype
            video_embeds = video_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            
            # Get scatter mask for video positions
            _, video_mask = get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds
            )
            
            # Scatter video embeddings into text embedding positions
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        
        # Step 3: Combine visual masks and deepstack features
        visual_pos_masks = None
        
        if image_mask is not None and video_mask is not None:
            # Both images and videos present - need to merge
            image_mask = image_mask[..., 0]  # Remove embedding dim
            video_mask = video_mask[..., 0]  # Remove embedding dim
            visual_pos_masks = image_mask | video_mask  # Combine masks
            
            # Merge deepstack features from both modalities
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                # Create joint embedding tensor
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        
        elif image_mask is not None:
            # Only images
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            # deepstack_visual_embeds already set from images
        
        elif video_mask is not None:
            # Only videos
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds
        
        # Step 4: Calculate position IDs if not provided
        if position_ids is None:
            position_ids, rope_deltas = get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
                spatial_merge_size=self.model_args.vision_config.spatial_merge_size,
                image_token_id=self.model_args.image_token_id,
                video_token_id=self.model_args.video_token_id,
                vision_start_token_id=self.model_args.vision_start_token_id,
            )
            self.rope_deltas = rope_deltas
        
        # Step 5: Forward through Qwen3VLTextModel with DeepStack
        # NOTE: Qwen3 doesn't support attention_masks, it uses causal masking internally
        outputs = self.language_model(
            tokens=inputs_embeds,  # Pass embeddings directly (not token IDs)
            rope_cache=self.language_model.rope_cache,
            attention_masks=None,  # Qwen3 uses causal masking, doesn't accept external masks
            positions=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        
        return outputs


# ============================================================================
# NEXT SESSION: IMPLEMENTATION GUIDE
# ============================================================================
"""
PHASE 4 - OPTION B IMPLEMENTATION PLAN
======================================

Why Option B (Extend Qwen3Model)?
1. ✅ Matches HF architecture exactly
2. ✅ Memory efficient (layer-wise injection)
3. ✅ Clean DeepStack implementation
4. ✅ Backward compatible (optional params)
5. ✅ Easier to debug and maintain

Implementation Order:
--------------------

Step 1: Implement Qwen3VLTextModel (60 min)
   a. _deepstack_process() - ~10 lines
      - Clone hidden_states
      - Add visual_embeds at visual_pos_masks positions
      
   b. forward() - ~50 lines  
      - Copy Qwen3Model.forward() logic
      - Add DeepStack injection in layer loop
      - Keep all Qwen3Model behavior intact

Step 2: Implement Helper Functions (60 min)
   a. get_placeholder_mask() - ~30 lines
   b. get_rope_index() - ~80 lines (most complex!)
   c. get_image_features() - ~15 lines

Step 3: Implement Qwen3VLModel.forward() (90 min)
   - Vision encoding and scattering
   - Position calculation
   - Call to Qwen3VLTextModel
   
Step 4: Testing (30 min)
   - Dummy inputs
   - Shape validation
   - Error checking

TOTAL: ~3.5 hours

Key Architecture:
----------------
```
Qwen3Model (base text decoder)
    ↓ extends
Qwen3VLTextModel (+ DeepStack injection)
    ↑ used by
Qwen3VLModel (vision + text integration)
```

Success Criteria:
----------------
✓ Qwen3VLTextModel extends Qwen3Model
✓ DeepStack injection works correctly
✓ Forward pass compiles
✓ Shapes validated
✓ Backward compatible
"""
