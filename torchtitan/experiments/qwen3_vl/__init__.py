# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL MOE - Vision-Language Model with Mixture of Experts

This package implements Qwen3 VLM to train "Qwen/Qwen3-VL-30B-A3B-Instruct"
combining:
- Qwen3 text-only model with MOE support
- SigLIP-2 vision encoder
- DeepStack visual feature integration
- Multi-dimensional RoPE for vision position encoding
"""

from torchtitan.models.moe import MoEArgs

from .model import Qwen3VLModel, Qwen3VLTextModel
from .model.args import Qwen3VLModelArgs, Qwen3VLVisionArgs, SpecialTokens

__all__ = [
    "Qwen3VLModel",
    "Qwen3VLTextModel",
    "Qwen3VLModelArgs",
    "Qwen3VLVisionArgs",
    "SpecialTokens",
    "qwen3_vl_args",
]

# Model configurations for different Qwen3 VL variants
qwen3_vl_args = {
    "debugmodel": Qwen3VLModelArgs(
        # Text model config (small debug model)
        vocab_size=2048,
        max_seq_len=4096,
        head_dim=128,
        dim=256,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=3072,
        rope_theta=1000000,
        enable_weight_tying=True,
        moe_enabled=False,
        # Vision config (small for debug)
        vision_config=Qwen3VLVisionArgs(
            depth=12,
            hidden_size=512,
            intermediate_size=2048,
            num_heads=8,
            out_hidden_size=256,
            deepstack_visual_indexes=[2, 4, 6],
        ),
    ),
    "30B-A3B": Qwen3VLModelArgs(
        # Text model config (from Qwen3 30B-A3B)
        vocab_size=151936,
        max_seq_len=262144,
        head_dim=128,
        dim=2048,
        n_layers=48,
        n_heads=32,
        n_kv_heads=4,
        qk_norm=True,
        hidden_dim=6144,
        rope_theta=1000000,
        moe_enabled=True,
        moe_inter_dim=768,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=0,
            top_k=8,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
        ),
        # Vision config (SigLIP-2 based encoder)
        vision_config=Qwen3VLVisionArgs(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            hidden_act="gelu_pytorch_tanh",
            in_channels=3,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=2048,  # Matches text dim for projector
            num_position_embeddings=2304,
            deepstack_visual_indexes=[8, 16, 24],
            initializer_range=0.02,
        ),
        # Special token IDs for multimodal
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        deepstack_visual_indexes=[8, 16, 24],
    ),
}
