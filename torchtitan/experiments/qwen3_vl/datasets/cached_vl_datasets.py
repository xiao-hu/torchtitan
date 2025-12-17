# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Cached Vision-Language datasets for TorchTitan.

Provides fast loading of cached VL datasets using TensorDict memory-mapped cache.
"""

import hashlib
from pathlib import Path
from typing import Callable

from tensordict import LazyStackedTensorDict, TensorDict
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

# ============================================================================
# Helper Functions
# ============================================================================

def get_cache_key(
    dataset_name: str,
    seq_len: int,
    buffer_size: int,
    model_path: str,
) -> str:
    """Generate deterministic cache key from config."""
    model_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
    return f"{dataset_name}_seq{seq_len}_buf{buffer_size}_{model_hash}"


# ============================================================================
# Cached VL Datasets Registry
# ============================================================================

# Maps dataset names to their cached directories
VL_CACHED_DATASETS = {
    "vqav2_cached": "/data/xxie-sandbox/preprocessed_cache_td/",
    # Add more cached datasets here:
    # "coco_cached": "/path/to/coco/cache",
    # etc.
}


# ============================================================================
# Cached Dataset Class
# ============================================================================

class CachedVLDataset(IterableDataset, Stateful):
    """
    Load cached and packed VL samples from TensorDict memmap cache.
    
    Much faster than HuggingFaceVLDataset since:
    - No PIL decoding
    - No tokenization
    - No online packing
    - Memory-mapped TensorDict (instant init, zero-copy, native tensors)
    
    TensorDict benefits:
    - Native tensor storage (no conversion overhead)
    - Memory-mapped format (instant loading)
    - Simple sharding (just slice: td[rank::world_size])
    - Zero-copy access
    """
    
    def __init__(
        self,
        cache_dir: str,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ):
        cache_path = Path(cache_dir)
        samples_dir = cache_path / "samples"
        
        # Load all samples using LazyStackedTensorDict (instant, seamless access!)
        logger.info(f"Loading preprocessed samples from {samples_dir}...")
        sample_paths = sorted(samples_dir.glob("sample_*"))
        
        if not sample_paths:
            raise ValueError(f"No samples found in {samples_dir}")
        
        # Load each sample as memmap
        samples = [TensorDict.load_memmap(str(path)) for path in sample_paths]
        
        # Stack lazily (no memory overhead, seamless indexing!)
        full_dataset = LazyStackedTensorDict(*samples, stack_dim=0)
        
        # Apply DP sharding (simple slice!)
        self.dataset = full_dataset[dp_rank::dp_world_size]
        
        logger.info(f"Loaded {len(full_dataset)} samples")
        logger.info(f"DP shard: {len(self.dataset)} samples for rank {dp_rank}/{dp_world_size}")
        logger.info("Format: LazyStackedTensorDict (memory-mapped, zero-copy, native tensors)")
        
        self.infinite = infinite
        self._sample_idx = 0
    
    def __iter__(self):
        while True:
            for i in range(self._sample_idx, len(self.dataset)):
                self._sample_idx = i + 1
                # Return dict (TensorDict behaves like dict)
                yield dict(self.dataset[i])  # Memory-mapped access - zero copy!
            
            if not self.infinite:
                logger.info("Cached dataset epoch complete")
                break
            else:
                self._sample_idx = 0
                logger.info("Cached dataset restarting epoch")
    
    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
    
    def state_dict(self):
        return {"sample_idx": self._sample_idx}


# ============================================================================
# Dataloader Builder
# ============================================================================

def build_cached_vl_dataloader(
    dp_world_size: int,
    dp_rank: int,
    collate_fn: Callable,
    dataset_name: str,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build dataloader for cached VL dataset.
    
    Loads data from cached TensorDict cache (fast, memory-mapped).
    Automatically resolves cache directory from VL_CACHED_DATASETS registry.
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        collate_fn: Collator for batching
        dataset_name: Dataset name from VL_CACHED_DATASETS registry
        job_config: Job configuration (for seq_len, model_path, batch_size)
        infinite: Whether to loop dataset infinitely
        
    Returns:
        ParallelAwareDataloader with CachedVLDataset
    """
    # Get cache base from registry
    if dataset_name not in VL_CACHED_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in VL_CACHED_DATASETS. "
            f"Available: {list(VL_CACHED_DATASETS.keys())}"
        )
    
    cache_base = Path(VL_CACHED_DATASETS[dataset_name])
    
    # It's a base directory - find the actual cache using get_cache_key
    seq_len = job_config.training.seq_len
    buffer_size = max(50, seq_len // 40)
    model_path = job_config.model.hf_assets_path
    
    # Strip "_cached" suffix to get base dataset name
    base_dataset = dataset_name.replace("_cached", "")
    cache_key = get_cache_key(
        dataset_name=base_dataset,
        seq_len=seq_len,
        buffer_size=buffer_size,
        model_path=model_path,
    )
    
    cache_dir = cache_base / cache_key
    logger.info(f"Resolved cache directory: {cache_dir}")
    
    # Verify cache exists
    if not cache_dir.exists():
        raise ValueError(
            f"Cached dataset not found: {cache_dir}\n"
            f"Dataset '{dataset_name}' is in VL_CACHED_DATASETS but cache is missing."
        )
    
    samples_dir = cache_dir / "samples"
    if not samples_dir.exists() or not list(samples_dir.glob("sample_*")):
        raise ValueError(
            f"Invalid cache directory: {cache_dir}\n"
            f"Expected samples/ directory with sample_* files."
        )
    
    # Load cached dataset
    logger.info(f"Loading CachedVLDataset from: {cache_dir}")
    vl_ds = CachedVLDataset(
        cache_dir=str(cache_dir),
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )
    
    return ParallelAwareDataloader(
        dataset=vl_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
        collate_fn=collate_fn,
    )
