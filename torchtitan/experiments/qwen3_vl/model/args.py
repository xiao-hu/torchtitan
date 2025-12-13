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

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs


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
    # Specifies which vision encoder layers to extract features from for injection into text decoder
    # In HF config, this is part of vision_config, but since we reuse Siglip2ModelArgs, we keep it here
    # Default [8, 16, 24] for Qwen3-VL-30B-A3B matches HF implementation
    # These vision layer features are injected into corresponding early text decoder layers
    deepstack_visual_indexes: list[int] = field(default_factory=lambda: [8, 16, 24])
