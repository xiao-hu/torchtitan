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
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import JobConfig
from torchtitan.experiments.qwen3_vl import Qwen3VLModel, qwen3_vl_args
from torchtitan.experiments.qwen3_vl.datasets.cached_vl_datasets import (
    VL_CACHED_DATASETS, build_cached_vl_dataloader)
from torchtitan.experiments.qwen3_vl.datasets.utils import (
    collate_vl_batch, preprocess_qwen_visual_pil)
from torchtitan.experiments.qwen3_vl.datasets.vl_datasets import (
    VL_DATASETS, build_vl_dataloader)
from torchtitan.experiments.qwen3_vl.infra.parallelize import \
    parallelize_qwen3_vl
from torchtitan.experiments.qwen3_vl.model.state_dict_adapter import \
    Qwen3VLStateDictAdapter
from torchtitan.hf_datasets.text_datasets import DATASETS as TEXT_DATASETS
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec


def build_qwen3vl_tokenizer(job_config: JobConfig):
    """
    Load appropriate tokenizer based on dataset type.
    
    For text-only datasets (c4, pile, etc.):
        Returns HuggingFaceTokenizer with encode(text, add_bos, add_eos) interface
        
    For vision-language datasets (vqav2, etc.):
        Returns Qwen3VLProcessor (tokenizer + image processor)
    
    Args:
        job_config: TorchTitan job configuration
        
    Returns:
        HuggingFaceTokenizer or Qwen3VLProcessor depending on dataset
    """
    dataset_name = job_config.training.dataset.lower()
    
    # Text-only datasets: use HuggingFaceTokenizer
    if dataset_name in TEXT_DATASETS:
        return HuggingFaceTokenizer(job_config.model.hf_assets_path)
    
    # Vision-language datasets: use Qwen3VLProcessor
    else:
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
    Build dataloader with auto-detection for text-only vs vision-language datasets.
    
    Supports:
    - Text-only datasets (c4, pile, etc.) → Uses HuggingFaceTextDataset
    - Vision-language datasets (vqav2, etc.) → Uses HuggingFaceVLDataset
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Qwen3VLProcessor (includes tokenizer + image processor)
        job_config: TorchTitan job configuration
        
    Returns:
        ParallelAwareDataloader with appropriate preprocessing
    """
    from torchtitan.tools.logging import logger
    
    dataset_name = job_config.training.dataset.lower()
    
    # Route 1: Text-only datasets (reuse existing text infrastructure)
    if dataset_name in TEXT_DATASETS:
        logger.info(f"Using text-only dataset: {dataset_name}")
        # tokenizer is already HuggingFaceTokenizer from build_qwen3vl_tokenizer
        return build_text_dataloader(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            job_config=job_config,
            infinite=True,
        )
    
    # Route 2: Cached VL datasets (fast cache loading)
    elif dataset_name in VL_CACHED_DATASETS:
        logger.info(f"Using cached VL dataset: {dataset_name}")
        
        # Create collator with tokenizer captured in closure
        def collator(instances):
            return collate_vl_batch(instances, tokenizer)
        
        # Build dataloader using cached dataset
        # Cache resolution happens inside build_cached_vl_dataloader
        return build_cached_vl_dataloader(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            collate_fn=collator,
            dataset_name=dataset_name,
            job_config=job_config,
            infinite=True,
        )
    
    # Route 3: Vision-language datasets (use VL-specific preprocessing)
    elif dataset_name in VL_DATASETS:
        logger.info(f"Using vision-language dataset: {dataset_name}")
        
        # Create collator with tokenizer captured in closure
        def collator(instances):
            return collate_vl_batch(instances, tokenizer)
        
        # Build dataloader with model-specific preprocessing
        return build_vl_dataloader(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            processor=tokenizer,
            preprocess_fn=preprocess_qwen_visual_pil,
            collate_fn=collator,
            job_config=job_config,
            infinite=True,  # Loop dataset infinitely for training
        )
    
    # Route 4: Unknown dataset
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported text datasets: {list(TEXT_DATASETS.keys())}. "
            f"Supported VL datasets: {list(VL_DATASETS.keys())}. "
            f"Supported cached VL datasets: {list(VL_CACHED_DATASETS.keys())}"
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
