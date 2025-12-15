# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL Model Parallelization

This file applies parallelisms to the Qwen3-VL model which consists of:
1. Vision encoder (model.visual) - Uses default FSDP2
2. Language model (model.language_model) - Uses Qwen3 parallelization

The language model is a Qwen3VLTextModel (extends Qwen3Model) which contains
the transformer layers that need TP/EP/CP treatment.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
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
    - Vision encoder (model.visual): Default FSDP2 (no TP/EP)
    - Language model (model.language_model): Full Qwen3 parallelization (TP/EP/CP/FSDP)
    
    Args:
        model: Qwen3VLModel instance with .visual and .language_model
        parallel_dims: Parallelism configuration
        job_config: Training job configuration
    """
    world_mesh = parallel_dims.world_mesh
    
    # ========================================================================
    # STEP 1: Apply FSDP to Vision Encoder
    # ========================================================================
    # Vision encoder uses default FSDP2 without any special parallelism
    if parallel_dims.fsdp_enabled:
        # Determine DP mesh dimensions
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        
        # Apply FSDP to vision encoder using fully_shard directly
        # (vision encoder doesn't have the same structure as language model)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )
        
        fully_shard(
            model.visual,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )
        logger.info("Applied FSDP to vision encoder")
    
    # ========================================================================
    # STEP 2: Apply Full Qwen3 Parallelization to Language Model
    # ========================================================================
    # Reuse the existing parallelize_qwen3 function for the language model
    # This handles TP, EP, AC, Compile, and FSDP for the transformer layers
    parallelize_qwen3(
        model.language_model,
        parallel_dims,
        job_config,
    )
    logger.info("Applied Qwen3 parallelization to language model")
    
    return model
