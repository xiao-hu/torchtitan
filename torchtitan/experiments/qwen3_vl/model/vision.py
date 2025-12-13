# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL Vision Encoder

Thin wrapper around HuggingFace's Qwen3VLVisionModel for TorchTitan integration.
Uses proven HF implementation with minimal adaptations for TorchTitan compatibility.
"""

import torch

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionConfig, Qwen3VLVisionModel)
except ImportError:
    raise ImportError(
        "Qwen3 VL vision encoder requires transformers library. "
        "Install with: pip install transformers>=4.37.0"
    )

from .args import Qwen3VLVisionArgs


def convert_vision_config_to_hf(vision_config) -> Qwen3VLVisionConfig:
    """
    Convert Qwen3VLVisionArgs to HF Qwen3VLVisionConfig.
    
    Since field names are aligned, this is a trivial conversion.
    
    Args:
        vision_config: Qwen3VLVisionArgs with HF-aligned field names
        
    Returns:
        Qwen3VLVisionConfig for HuggingFace
    """
    # Field names match perfectly - just convert dataclass to HF config
    return Qwen3VLVisionConfig(**vision_config.__dict__)


class Qwen3VLVisionEncoder(Qwen3VLVisionModel):
    """
    TorchTitan wrapper for HuggingFace Qwen3VLVisionModel.
    
    This is a thin integration layer that:
    - Converts torchtitan config to HF config
    - Provides compatible forward signature
    - Maintains HF's proven implementation (3D patching, merging, DeepStack)
    
    No parallelism is applied to the vision encoder as it's relatively small.
    """
    
    def __init__(self, args: Qwen3VLVisionArgs):
        """
        Initialize vision encoder from torchtitan config.
        
        Args:
            args: Qwen3VLVisionArgs with vision encoder configuration
        """
        # Convert torchtitan config to HF config
        hf_vision_config = convert_vision_config_to_hf(args)
        
        # Store deepstack indexes from args (may differ from default)
        if hasattr(args, 'deepstack_visual_indexes'):
            hf_vision_config.deepstack_visual_indexes = args.deepstack_visual_indexes
        
        # Initialize parent HF model
        super().__init__(hf_vision_config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through vision encoder.
        
        Args:
            hidden_states: Patchified pixel values [seq_len, hidden_size]
            grid_thw: Grid dimensions [num_images, 3] (temporal, height, width)
            **kwargs: Additional arguments (ignored for compatibility)
        
        Returns:
            Tuple of (final_features, deepstack_features_list)
            - final_features: [seq_len, out_hidden_size]
            - deepstack_features_list: List of intermediate features for DeepStack
        """
        # HF model returns (hidden_states, deepstack_features_list)
        return super().forward(hidden_states, grid_thw, **kwargs)


# Convenience: Export HF components for use in other parts of the model
__all__ = [
    'Qwen3VLVisionEncoder',
    'Qwen3VLVisionModel',
    'Qwen3VLVisionConfig',
    'convert_vision_config_to_hf',
]
