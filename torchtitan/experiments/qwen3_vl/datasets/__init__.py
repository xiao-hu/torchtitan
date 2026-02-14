# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL dataset utilities (EXPERIMENTAL).

This module contains all dataset infrastructure for Qwen3-VL:
1. Generic VL dataset infrastructure (DatasetConfig pattern)
2. Batch-free preprocessing and collation (optimized for TorchTitan)

Architecture:
    torchtitan/experiments/qwen3_vl/datasets/
    ├── vl_datasets.py          # Generic VL infrastructure (EXPERIMENTAL)
    ├── utils.py                # Batch-free preprocessing & collation
    ├── packing.py              # Sample packing utilities
    ├── data_processor.py       # Reference: Original Qwen3-VL preprocessing (UNUSED)
    └── __init__.py             # This file

Note: Once finalized and tested, vl_datasets.py may be promoted to
      torchtitan/hf_datasets/ for use by other VL models.

Usage Example:
    >>> from transformers import Qwen3VLProcessor
    >>> from torchtitan.experiments.qwen3_vl.datasets import (
    ...     build_vl_dataloader,
    ...     preprocess_qwen_visual_pil,
    ...     collate_vl_batch,
    ... )
    >>> 
    >>> processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
    >>> 
    >>> # Works with ANY dataset in VL_DATASETS registry!
    >>> dataloader = build_vl_dataloader(
    ...     dp_world_size=8,
    ...     dp_rank=0,
    ...     processor=processor,
    ...     preprocess_fn=preprocess_qwen_visual_pil,
    ...     collate_fn=lambda instances: collate_vl_batch(instances, processor),
    ...     job_config=job_config
    ... )
"""
# Batch-free preprocessing and collation utilities
from .utils import collate_vl_batch, preprocess_qwen_visual_pil
# Generic VL infrastructure (EXPERIMENTAL - may move to torchtitan/hf_datasets/)
from .vl_datasets import (VL_DATASETS, HuggingFaceVLDataset,
                          build_vl_dataloader, format_vqav2_sample)

__all__ = [
    # Generic VL infrastructure (EXPERIMENTAL)
    "VL_DATASETS",
    "HuggingFaceVLDataset",
    "build_vl_dataloader",
    "format_vqav2_sample",
    # Batch-free preprocessing and collation (optimized for TorchTitan)
    "preprocess_qwen_visual_pil",
    "collate_vl_batch",
]
