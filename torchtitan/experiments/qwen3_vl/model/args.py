# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL MOE Model Arguments

Configuration classes for the Qwen3 VL MOE model, including:
- Vision encoder configuration (SigLIP-2)
- Text decoder configuration (Qwen3 with MOE)
- Special tokens for multimodal inputs
"""

from dataclasses import dataclass, field

import torch.nn as nn

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.utils import get_moe_model_nparams_and_flops


@dataclass
class Qwen3VLVisionArgs:
    """
    Vision encoder configuration with field names matching HF Qwen3VLVisionConfig.
    
    This alignment minimizes conversion overhead when initializing the HF vision model.
    All field names and defaults match HuggingFace's Qwen3VLVisionConfig exactly.
    """
    
    # Architecture
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    hidden_act: str = "gelu_pytorch_tanh"
    
    # Patching
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    
    # Output
    out_hidden_size: int = 3584  # Dimension after merger, fed to projector
    num_position_embeddings: int = 2304  # 48x48 grid
    
    # DeepStack - intermediate layer feature extraction
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])
    
    # Training
    initializer_range: float = 0.02


@dataclass
class SpecialTokens:
    """Special tokens for vision-language model inputs."""
    
    img_token: str
    img_id: int
    video_token: str
    video_id: int
    vision_start_token: str
    vision_start_id: int
    vision_end_token: str
    vision_end_id: int
    pad_token: str
    pad_id: int
    ignore_id: int = -100  # PyTorch cross_entropy default
    
    @classmethod
    def from_tokenizer(cls, tokenizer: HuggingFaceTokenizer):
        """
        Extract special token IDs from a HuggingFace tokenizer.
        
        Args:
            tokenizer: HuggingFaceTokenizer instance
            
        Returns:
            SpecialTokens instance with IDs extracted from tokenizer
        """
        SPECIAL_TOKENS_MAP = {
            "img": "<|image|>",
            "video": "<|video|>",
            "vision_start": "<|vision_start|>",
            "vision_end": "<|vision_end|>",
            "pad": "<|pad|>",
        }
        added_tokens = tokenizer.tokenizer.get_added_tokens_decoder()
        token_to_id = {tok.content: tok_id for tok_id, tok in added_tokens.items()}
        special_tokens_dict = {}
        for prefix, tok in SPECIAL_TOKENS_MAP.items():
            special_tokens_dict[f"{prefix}_token"] = tok
            special_tokens_dict[f"{prefix}_id"] = token_to_id[tok]
        return cls(**special_tokens_dict)


@dataclass
class Qwen3VLModelArgs(Qwen3ModelArgs):
    """
    Extended Qwen3 model arguments with vision support.
    
    Inherits text model configuration from Qwen3ModelArgs (including moe_enabled flag)
    and adds vision encoder, projector, and multimodal-specific parameters.
    MOE vs dense is controlled by the inherited moe_enabled flag.
    """
    
    # Vision encoder configuration with HF-aligned field names
    vision_config: Qwen3VLVisionArgs = field(default_factory=Qwen3VLVisionArgs)
    
    # Special token IDs for multimodal inputs
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    
    # DeepStack configuration
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        """Override to include vision encoder FLOPs in MFU calculation.

        The base Qwen3 calculation only counts language model FLOPs.
        We add the vision encoder's FLOPs amortized per text token, estimated as:
        - vision_flops_per_patch = 6 * (vision_params - vision_embed_params)
        - avg_patches_per_token ≈ 0.5 (empirical: ~1000 patches per ~2000 text tokens in packed seq)
        - vision_flops_per_text_token = vision_flops_per_patch * avg_patches_per_token
        """
        # Get base language model FLOPs (from Qwen3 MOE calculation)
        nparams, lang_flops_per_token = get_moe_model_nparams_and_flops(
            self, model, 2 * self.head_dim, seq_len
        )

        # Estimate vision encoder FLOPs per patch
        vc = self.vision_config
        # Vision encoder: 27 blocks × (attention + MLP) per patch
        # Attention: 4 × hidden² (QKV + output projections)
        # MLP: 2 × hidden × intermediate
        vision_flops_per_patch = vc.depth * (
            4 * vc.hidden_size * vc.hidden_size  # attention
            + 2 * vc.hidden_size * vc.intermediate_size  # MLP
        )
        # Merger FLOPs: merge_dim → out_hidden (applied once after all blocks)
        merge_dim = vc.hidden_size * (vc.spatial_merge_size ** 2)
        merger_flops = 2 * merge_dim * merge_dim + 2 * merge_dim * vc.out_hidden_size
        # DeepStack mergers (3 of them)
        deepstack_merger_flops = len(vc.deepstack_visual_indexes) * merger_flops

        # Total vision FLOPs per patch (×6 for fwd+bwd: 2 for matmul fwd, 4 for bwd)
        total_vision_flops_per_patch = 6 * (vision_flops_per_patch + merger_flops // vc.depth + deepstack_merger_flops // vc.depth)

        # Amortize over text tokens: ~0.5 vision patches per text token (empirical)
        # In packed VQAv2 sequences: ~5-10 images × ~260 merged patches / ~8000 text tokens
        avg_patches_per_token = 0.3
        vision_flops_per_text_token = int(total_vision_flops_per_patch * avg_patches_per_token)

        num_flops_per_token = lang_flops_per_token + vision_flops_per_text_token
        return nparams, num_flops_per_token
