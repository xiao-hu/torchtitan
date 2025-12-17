# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Vision-Language datasets for TorchTitan.

Follows the same pattern as text_datasets.py with DatasetConfig,
but for multimodal (vision + text) data.
"""

import json
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from tensordict import LazyStackedTensorDict, TensorDict
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.config import JobConfig
from torchtitan.experiments.qwen3_vl.datasets.packing import VLSamplePacker
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger

# ============================================================================
# Dataset Loaders
# ============================================================================

def _load_hf_dataset(dataset_path: str, split: str):
    """Load HuggingFace dataset with default settings (non-streaming)."""
    return load_dataset(dataset_path, split=split, streaming=False)


# ============================================================================
# Sample Processors
# ============================================================================

def format_vqav2_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process VQAv2 sample into Qwen3-VL conversation format.
    
    Input (VQAv2 format):
        {
            "image": PIL.Image,
            "question": str,
            "answers": [
                {"answer": str, "answer_confidence": str, "answer_id": int},
                ...
            ],
            ...
        }
    
    Output (Qwen3-VL conversation format):
        {
            "conversations": [
                {"from": "human", "value": "<image>What color is the car?"},
                {"from": "gpt", "value": "Red"}
            ],
            "image": [PIL.Image],
            "data_path": ""
        }
    """
    # Get the most common answer (VQAv2 has multiple annotations)
    # answers is a list of dicts: [{"answer": str, "answer_confidence": str, "answer_id": int}, ...]
    answers_list = sample.get("answers", [])
    if answers_list and isinstance(answers_list, list):
        # Extract the "answer" field from the first annotation
        answer = answers_list[0].get("answer", "unknown") if isinstance(answers_list[0], dict) else "unknown"
    else:
        answer = "unknown"
    
    # Format question with image placeholder
    question = sample["question"]
    
    # Build conversation in Qwen3-VL format
    return {
        "conversations": [
            {
                "from": "human",
                "value": f"<image>{question}"
            },
            {
                "from": "gpt",
                "value": answer
            }
        ],
        "image": [sample["image"]],  # PIL Image - processor will handle it
        "data_path": ""  # Empty since we're providing PIL images directly
    }


# ============================================================================
# Dataset Registry
# ============================================================================

VL_DATASETS = {
    "vqav2": DatasetConfig(
        path="lmms-lab/VQAv2",
        loader=partial(_load_hf_dataset, split="validation"),
        sample_processor=format_vqav2_sample,
    ),
    "vqav2_validation": DatasetConfig(
        path="lmms-lab/VQAv2",
        loader=partial(_load_hf_dataset, split="test"),
        sample_processor=format_vqav2_sample,
    ),
    # Add more VL datasets here following the same pattern:
    # "coco_caption": DatasetConfig(...),
    # "nocaps": DatasetConfig(...),
    # etc.
}


def _validate_vl_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate VL dataset name and path."""
    if dataset_name not in VL_DATASETS:
        raise ValueError(
            f"VL dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(VL_DATASETS.keys())}"
        )

    config = VL_DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} VL dataset from {path}")
    return path, config.loader, config.sample_processor


# ============================================================================
# Generic VL Dataset Class
# ============================================================================

class HuggingFaceVLDataset(IterableDataset, Stateful):
    """
    HuggingFace Vision-Language Dataset with optional sample packing.
    
    Follows the same pattern as mm_datasets.py for proven reliability.
    Sample-based packing maintains strict sample boundaries and works
    with models that don't support external attention masks.
    
    Args:
        dataset_name: Name of dataset from VL_DATASETS registry
        dataset_path: Optional override path
        processor: Vision-language processor (tokenizer + image processor)
        preprocess_fn: Function to preprocess samples
        batch_size: Batch size for sample packing
        seq_len: Maximum sequence length
        packing_buffer_size: Buffer size for packing (0=disabled)
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        infinite: Whether to loop dataset infinitely
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        processor: Any,
        preprocess_fn: Callable,
        batch_size: int = 1,
        seq_len: int = 2048,
        packing_buffer_size: int = 0,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, sample_processor = _validate_vl_dataset(
            dataset_name, dataset_path
        )
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._processor = processor
        self._preprocess_fn = preprocess_fn
        self._sample_processor = sample_processor
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.infinite = infinite
        self.enable_packing = packing_buffer_size > 0
        
        if self.enable_packing:
            self.packer = VLSamplePacker(
                max_seq_length=seq_len,
                buffer_size=packing_buffer_size,
                batch_size=batch_size,
            )
            logger.info(
                f"Sample packing enabled: buffer_size={packing_buffer_size}, "
                f"seq_len={seq_len}, batch_size={batch_size}"
            )

        # Variables for checkpointing
        self._sample_idx = 0
        self.error_count = 0

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                try:
                    self._sample_idx += 1
                    
                    # Format and preprocess sample
                    formatted_sample = self._sample_processor(sample)
                    processed = self._preprocess_fn([formatted_sample], self._processor)
                    
                    if processed is None:
                        continue

                    # Check sequence length
                    if processed["input_ids"].shape[0] > self.seq_len:
                        logger.warning(
                            f"Sample length {processed['input_ids'].shape[0]} > "
                            f"training seq_len={self.seq_len}. Skipping."
                        )
                        continue

                    if self.enable_packing:
                        self.packer.add_sample(processed)
                        
                        if self.packer.has_batch_ready():
                            batch = self.packer.get_next_batch()
                            if batch:
                                yield from batch
                    else:
                        yield processed  # individual sample

                except Exception as e:
                    error_msg = str(e)
                    # Critical errors that should stop execution immediately
                    critical_errors = [
                        "Tensors must have same number of dimensions",
                        "shape mismatch",
                        "dimension mismatch",
                    ]
                    
                    if any(critical in error_msg for critical in critical_errors):
                        logger.error(f"CRITICAL ERROR in packing/processing: {e}")
                        raise  # Re-raise critical errors
                    
                    # Only swallow minor preprocessing errors
                    logger.warning(f"Error processing VL sample: {e}")
                    self.error_count += 1
                    if self.error_count > 10:
                        logger.error("Too many errors. Stopping.")
                        raise
                    continue

            # Flush remaining packed samples at end of epoch
            if self.enable_packing:
                while True:
                    batch = self.packer.get_next_batch()
                    if batch:
                        yield from batch
                    else:
                        break

            if not self.infinite:
                logger.info(f"VL dataset {self.dataset_name} epoch complete")
                break
            else:
                self._sample_idx = 0
                logger.info(f"VL dataset {self.dataset_name} restarting epoch")

    def _get_data_iter(self):
        """Get iterator with proper state management."""
        try:
            # Check if we're at the end of non-streaming dataset
            if not hasattr(self._data, "iterable_dataset"):
                if isinstance(self._data, Dataset) and (
                    self._sample_idx == len(self._data)
                ):
                    return iter([])

            it = iter(self._data)

            # Skip to current position for checkpoint resumption
            if self._sample_idx > 0:
                for _ in range(self._sample_idx):
                    next(it)

            return it
        except Exception as e:
            logger.error(f"Error in _get_data_iter: {e}")
            return iter([])

    def load_state_dict(self, state_dict):
        """Load dataset state."""
        self._sample_idx = state_dict["sample_idx"]

        # Restore packer state if available
        if (
            self.enable_packing
            and hasattr(self, "packer")
            and "packer_state" in state_dict
        ):
            packer_state = state_dict["packer_state"]
            self.packer.sample_buffer.clear()
            self.packer.packed_samples.clear()
            self.packer.sample_buffer.extend(packer_state["sample_buffer"])
            self.packer.packed_samples.extend(packer_state["packed_samples"])

    def state_dict(self):
        """Save dataset state."""
        state = {"sample_idx": self._sample_idx}

        # Save packer state if packing is enabled
        if self.enable_packing and hasattr(self, "packer"):
            state["packer_state"] = {
                "sample_buffer": list(self.packer.sample_buffer),
                "packed_samples": list(self.packer.packed_samples),
            }

        return state


# ============================================================================
# Preprocessed Dataset Class
# ============================================================================

class PreprocessedVLDataset(IterableDataset, Stateful):
    """
    Load preprocessed and packed VL samples from TensorDict memmap cache.
    
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
                logger.info("Preprocessed dataset epoch complete")
                break
            else:
                self._sample_idx = 0
                logger.info("Preprocessed dataset restarting epoch")
    
    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
    
    def state_dict(self):
        return {"sample_idx": self._sample_idx}


# ============================================================================
# Dataloader Builder
# ============================================================================

def build_vl_dataloader(
    dp_world_size: int,
    dp_rank: int,
    processor: Any,  # VL processor
    preprocess_fn: Callable,  # Preprocessing function (e.g., preprocess_qwen_visual_pil from train_spec)
    collate_fn: Callable,  # Collator for batching
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build a data loader for Vision-Language datasets with optional sample packing.
    
    Follows the proven pattern from mm_datasets.py.
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        processor: Vision-language processor (e.g., Qwen3VLProcessor)
        preprocess_fn: Function to preprocess samples
        collate_fn: Collator for batching (e.g., DataCollatorForSupervisedDataset)
        job_config: TorchTitan job configuration
        infinite: Whether to loop dataset infinitely
        
    Returns:
        ParallelAwareDataloader instance
        
    Sample Packing:
        - Enable via packing_buffer_size > 0 in config
        - Maintains sample boundaries for correct gradient computation
        - Works with models that don't support external attention masks
        
    Example:
        >>> from transformers import Qwen3VLProcessor
        >>> from torchtitan.experiments.qwen3_vl.datasets import (
        ...     preprocess_qwen_visual,
        ...     DataCollatorForSupervisedDataset
        ... )
        >>> 
        >>> processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
        >>> collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
        >>> 
        >>> dataloader = build_vl_dataloader(
        ...     dp_world_size=8,
        ...     dp_rank=0,
        ...     processor=processor,
        ...     preprocess_fn=preprocess_qwen_visual,
        ...     collate_fn=collator,
        ...     job_config=job_config
        ... )
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    # Derive packing buffer size from seq_len
    # Logic: Assuming average VQA sample is ~400 tokens (question + answer + vision tokens)
    # - seq_len / 400 ≈ number of samples that fit in one packed sequence
    # - We want to buffer 10x that to get good packing efficiency
    # - So: buffer_size = 10 * (seq_len / 400) = seq_len / 40
    # Example: seq_len=4096 → buffer_size ≈ 102 samples
    packing_buffer_size = max(50, seq_len // 40)
    logger.info(f"Packing enabled with buffer_size={packing_buffer_size} (derived from seq_len={seq_len})")

    hf_vl_ds = HuggingFaceVLDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        processor=processor,
        preprocess_fn=preprocess_fn,
        batch_size=batch_size,
        seq_len=seq_len,
        packing_buffer_size=packing_buffer_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    return ParallelAwareDataloader(
        dataset=hf_vl_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
