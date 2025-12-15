# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TrainSpec for Qwen3-VL Vision-Language Model.

Integrates Qwen3-VL with TorchTitan's training loop by providing:
- Model instantiation (Qwen3VLModel)
- Tokenizer/Processor loading (Qwen3VLProcessor)
- Dataloader construction with VL-specific preprocessing
- Model converter for checkpoint conversion
"""

from transformers import Qwen3VLProcessor

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.validate import build_validator
from torchtitan.config import JobConfig
from torchtitan.experiments.qwen3_vl import Qwen3VLModel, qwen3_vl_args
from torchtitan.experiments.qwen3_vl.datasets import (
    DataCollatorForSupervisedDataset, build_vl_dataloader,
    preprocess_qwen_visual)
from torchtitan.experiments.qwen3_vl.infra.parallelize import \
    parallelize_qwen3_vl
from torchtitan.experiments.qwen3_vl.model.state_dict_adapter import \
    Qwen3VLStateDictAdapter
from torchtitan.protocols.train_spec import TrainSpec


def build_qwen3vl_tokenizer(job_config: JobConfig):
    """
    Load Qwen3-VL processor (tokenizer + image processor).
    
    Note: Returns Qwen3VLProcessor (not just tokenizer) which includes:
    - tokenizer: For text processing
    - image_processor: For vision input processing
    
    Args:
        job_config: TorchTitan job configuration
        
    Returns:
        Qwen3VLProcessor instance
    """
    processor = Qwen3VLProcessor.from_pretrained(
        job_config.model.hf_assets_path,
        trust_remote_code=True,
    )
    
    # Ensure padding side is correct for training
    processor.tokenizer.padding_side = "right"
    
    return processor


def build_qwen3vl_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,  # Actually a Qwen3VLProcessor
    job_config: JobConfig,
):
    """
    Build VL dataloader with Qwen3-VL specific preprocessing.
    
    This function encapsulates:
    - Processor (tokenizer + image processor)
    - Model-specific preprocessing (preprocess_qwen_visual)
    - Collator for batching (DataCollatorForSupervisedDataset)
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Qwen3VLProcessor (not just tokenizer!)
        job_config: TorchTitan job configuration
        
    Returns:
        ParallelAwareDataloader with VL-specific components
    """

    # Create collator with just the tokenizer component
    collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer.tokenizer
    )
    
    # Build dataloader with model-specific preprocessing
    return build_vl_dataloader(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        processor=tokenizer,
        preprocess_fn=preprocess_qwen_visual,
        collate_fn=collator,
        job_config=job_config,
        infinite=True,  # Loop dataset infinitely for training
    )


# Define TrainSpec for Qwen3-VL
qwen3_vl_train_spec = TrainSpec(
    model_cls=Qwen3VLModel,
    model_args=qwen3_vl_args,
    parallelize_fn=parallelize_qwen3_vl,  # Use Qwen3-VL parallelization
    pipelining_fn=None,  # No pipeline parallelism for now
    build_optimizers_fn=build_optimizers,
    build_lr_schedulers_fn=build_lr_schedulers,
    build_dataloader_fn=build_qwen3vl_dataloader,
    build_tokenizer_fn=build_qwen3vl_tokenizer,
    build_loss_fn=build_cross_entropy_loss,
    build_validator_fn=build_validator,
    state_dict_adapter=Qwen3VLStateDictAdapter,
)
