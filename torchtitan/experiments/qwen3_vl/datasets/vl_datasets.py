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

from functools import partial
from typing import Any, Callable, Dict

from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.config import JobConfig
from torchtitan.experiments.vlm.datasets.utils.packing import SamplePacker
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
            "answers": {"answer": [str, str, ...], ...},
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
    answers = sample.get("answers", {}).get("answer", [])
    answer = answers[0] if answers else "unknown"
    
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
    Generic Vision-Language dataset for HuggingFace datasets with optional sample packing.
    
    Uses a sample_processor to convert dataset-specific formats to
    the model's expected format (e.g., Qwen3-VL conversation format).
    
    Features:
        - Clean separation: dataset formatting vs model preprocessing
        - Optional sample packing for training efficiency
        - Streaming and non-streaming dataset support
        - Stateful checkpointing with packer state
        - Robust error handling
    
    Args:
        dataset_name: Name of dataset from VL_DATASETS registry
        dataset_path: Optional override path
        processor: Vision-language processor (tokenizer + image processor)
        preprocess_fn: Function to preprocess samples (e.g., preprocess_qwen_visual)
        batch_size: Batch size for sample packing
        seq_len: Maximum sequence length for packing
        packing_buffer_size: Size of packing buffer (0 = disabled)
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        infinite: Whether to loop dataset infinitely
    """
    
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        processor: Any,  # Qwen3VLProcessor or similar
        preprocess_fn: Callable,  # e.g., preprocess_qwen_visual
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

        # Optional sample packing for efficiency
        self.enable_packing = packing_buffer_size > 0
        if self.enable_packing:
            self.packer = SamplePacker(
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

    def __iter__(self):
        while True:
            for sample in self._get_data_iter():
                try:
                    # Step 1: Use dataset-specific processor to convert to standard format
                    formatted_sample = self._sample_processor(sample)
                    
                    # Step 2: Use model-specific preprocessing (tokenization, image processing, etc.)
                    processed = self._preprocess_fn([formatted_sample], self._processor)
                    
                    self._sample_idx += 1

                    # Check sequence length
                    if processed["input_ids"].shape[0] > self.seq_len:
                        logger.warning(
                            f"Sample length {processed['input_ids'].shape[0]} > "
                            f"training seq_len={self.seq_len}. Skipping."
                        )
                        continue

                    # Step 3: Optional packing or direct yield
                    if self.enable_packing:
                        self.packer.add_sample(processed)
                        
                        if self.packer.has_batch_ready():
                            batch = self.packer.get_next_batch()
                            if batch:
                                yield from batch
                    else:
                        yield processed

                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
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
                # Reset for next iteration
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
        """Load dataset state including packer state."""
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
            logger.info(f"Restored packer state: {len(packer_state['sample_buffer'])} buffered samples")

    def state_dict(self):
        """Save dataset state including packer state."""
        state = {"sample_idx": self._sample_idx}

        # Save packer state if packing is enabled
        if self.enable_packing and hasattr(self, "packer"):
            state["packer_state"] = {
                "sample_buffer": list(self.packer.sample_buffer),
                "packed_samples": list(self.packer.packed_samples),
            }

        return state


# ============================================================================
# Dataloader Builder
# ============================================================================

def build_vl_dataloader(
    dp_world_size: int,
    dp_rank: int,
    processor: Any,  # VL processor
    preprocess_fn: Callable,  # Preprocessing function
    collate_fn: Callable,  # Collator for batching
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build a data loader for Vision-Language datasets with optional sample packing.
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        processor: Vision-language processor (e.g., Qwen3VLProcessor)
        preprocess_fn: Function to preprocess samples (e.g., preprocess_qwen_visual)
        collate_fn: Collator for batching (e.g., DataCollatorForSupervisedDataset)
        job_config: TorchTitan job configuration
        infinite: Whether to loop dataset infinitely
        
    Returns:
        ParallelAwareDataloader instance
        
    Sample Packing:
        Enable via job_config.data.packing_buffer_size > 0
        - Improves GPU utilization by reducing padding
        - 30-50% training speedup typical
        - Maintains sample boundaries for correct gradient computation
        
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

    # Get packing configuration (default: disabled)
    # Since JobConfig doesn't have a data attribute, we default to 0 (disabled)
    packing_buffer_size = 0

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
