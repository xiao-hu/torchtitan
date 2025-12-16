# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL Model Parallelization

This file applies parallelisms to the Qwen3-VL model which consists of:
1. Vision encoder (model.visual) - Simple FSDP2 wrapping (no TP/EP/AC)
2. Language model (model.language_model) - Full Qwen3 parallelization
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchtitan.config import TORCH_DTYPE_MAP, JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.qwen3.infra.parallelize import parallelize_qwen3
from torchtitan.tools.logging import logger


def parallelize_qwen3_vl(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply parallelization to Qwen3-VL model.
    
    Strategy:
    - Vision encoder: Simple FSDP wrapping (no TP/EP/AC)
    - Language model: Delegate to parallelize_qwen3 (works because Qwen3VLTextModel extends Qwen3Model)
    
    Key insight: Qwen3VLTextModel extends Qwen3Model with identical structure (tok_embeddings, layers, 
    norm, output), so we can reuse parallelize_qwen3 directly. This ensures proper MOE mesh handling
    and avoids gradient clipping mesh mismatch errors.
    
    Args:
        model: Qwen3VLModel instance with .visual and .language_model
        parallel_dims: Parallelism configuration
        job_config: Training job configuration
    """
    world_mesh = parallel_dims.world_mesh
    language_model = model.language_model
    
    # ========================================================================
    # STEP 1: Wrap Vision Encoder with Simple FSDP (before language model parallelization)
    # ========================================================================
    if parallel_dims.fsdp_enabled:
        # Determine DP mesh
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        
        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )
        
        # Simple FSDP wrapping for vision encoder layers
        for transformer_block in model.visual.blocks:
            fully_shard(
                transformer_block,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
            )
        
        logger.info("Applied FSDP to vision encoder")
    
    # ========================================================================
    # STEP 2: Apply Full Qwen3 Parallelization to Language Model
    # ========================================================================
    # This handles TP, EP, AC, Compile, and FSDP for the language model
    # Works because Qwen3VLTextModel extends Qwen3Model with same structure
    parallelize_qwen3(
        language_model,
        parallel_dims,
        job_config,
    )
    
    # ========================================================================
    # STEP 3: Optional Compile for Vision Encoder
    # ========================================================================
    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    if model_compile_enabled:
        # Vision encoder has different structure (.blocks instead of .layers)
        # Apply torch.compile directly to each block
        for block in model.visual.blocks:
            block.forward = torch.compile(block.forward, backend=job_config.compile.backend)
        logger.info("Applied torch.compile to vision encoder blocks")
    
    # ========================================================================
    # STEP 4: Wrap Whole Model at Root Level
    # ========================================================================
    if parallel_dims.fsdp_enabled:
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
        logger.info("Applied root-level FSDP to complete Qwen3-VL model")
    
    return model
