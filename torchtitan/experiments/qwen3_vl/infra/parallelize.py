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
    - Optimization: Create dp_mesh once and share between vision and language models
    """
    world_mesh = parallel_dims.world_mesh

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
    )

    # ========================================================================
    # Create dp_mesh once for reuse (avoids duplicate mesh slicing)
    # ========================================================================
    dp_mesh = None
    if parallel_dims.fsdp_enabled:
        if parallel_dims.dp_replicate_enabled:
            # HSDP case: Need both dp_replicate and dp_shard_cp dimensions
            # No pre-flattened mesh exists for this combination
            # Must use tuple slicing (triggers deprecation warning until PT 2.11)
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
            dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]
        else:
            # Standard FSDP: Use pre-flattened mesh (warning-free)
            # This mesh was created during build_mesh() with _flatten()
            dp_mesh = world_mesh["dp_shard_cp"]

    # ========================================================================
    # STEP 1: Vision Encoder - Comprehensive FSDP Wrapping
    # ========================================================================
    
    if parallel_dims.fsdp_enabled:
        # Convert reshard_after_forward config to boolean
        # "default" means True (reshard for memory efficiency)
        reshard_config = job_config.parallelism.fsdp_reshard_after_forward
        if isinstance(reshard_config, str):
            vision_reshard = reshard_config.lower() != "never"
            # try this: vision_reshard = reshard_config.lower() == "always"
        else:
            vision_reshard = reshard_config
        
        # Wrap all vision encoder components
        # ModuleLists (blocks, deepstack_merger_list) need individual wrapping
        for name, module in model.visual.named_children():
            if len(list(module.parameters())) == 0:
                continue  # Skip modules without parameters
            
            if isinstance(module, nn.ModuleList):
                # ModuleList: Wrap each sub-module individually
                for sub_module in module:
                    fully_shard(
                        sub_module,
                        mesh=dp_mesh,
                        mp_policy=mp_policy,
                        reshard_after_forward=vision_reshard,
                    )
                logger.info(f"Applied FSDP to {len(module)} modules in vision component: {name}")
            else:
                # Regular module: Wrap directly
                fully_shard(
                    module,
                    mesh=dp_mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=vision_reshard,
                )
                logger.info(f"Applied FSDP to vision component: {name}")

    # ========================================================================
    # STEP 2: Apply Full Qwen3 Parallelization to Language Model
    # ========================================================================
    # Pass shared dp_mesh to avoid duplicate mesh creation
    parallelize_qwen3(
        model.language_model,
        parallel_dims,
        job_config,
    )

    # ========================================================================
    # STEP 3: Optional Compile for Vision Encoder
    # ========================================================================
    visual_compile_enabled = (
        job_config.compile.enable and "visual" in job_config.compile.components
    )
    if visual_compile_enabled:
        # try fullgraph=True, use TORCH_LOGS="recompiles,guards" to debug, use TORCH_LOGS="dynamic" to check the code forcing recompile
        # advanced: rewrite a Qwen-style view operation so that it is "compiler-friendly" and stops triggering recompiles
        model.visual = torch.compile(model.visual, backend=job_config.compile.backend)
        logger.info("Applied torch.compile to entire vision encoder")

        # compile the core part only, does not work
        # for i, block in enumerate(model.visual.blocks):
        #     model.visual.blocks[i] = torch.compile(block)
        # logger.info("Applied torch.compile to vision encoder blocks")

        # dynamic compile, try mark_dynamic(input_tensor, 1)/maybe_mark_dynamic(grid_thw, 0) before the forward pass instead
        # try mode="max-autotune-no-cudagraphs"
        # model.visual = torch.compile(model.visual, backend=job_config.compile.backend, dynamic=True)
        # logger.info("Applied torch.compile to entire vision encoder")

    # ========================================================================
    # STEP 4: Wrap Whole Model at Root Level
    # ========================================================================
    if parallel_dims.fsdp_enabled:
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
        logger.info("Applied root-level FSDP to complete Qwen3-VL model")
    
    return model
