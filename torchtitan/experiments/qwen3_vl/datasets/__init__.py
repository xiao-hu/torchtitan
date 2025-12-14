# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL dataset utilities (EXPERIMENTAL).

This module contains all dataset infrastructure for Qwen3-VL:
1. Generic VL dataset infrastructure (DatasetConfig pattern)
2. Qwen3-VL specific preprocessing (from official implementation)

Architecture:
    torchtitan/experiments/qwen3_vl/datasets/
    ├── vl_datasets.py                          # Generic VL infrastructure (EXPERIMENTAL)
    ├── rope2d.py                               # 3D RoPE (from official Qwen3-VL)
    ├── data_processor.py                       # Preprocessing (from official Qwen3-VL)
    └── __init__.py                             # This file

Note: Once finalized and tested, vl_datasets.py may be promoted to
      torchtitan/hf_datasets/ for use by other VL models.

Usage Example:
    >>> from transformers import Qwen3VLProcessor
    >>> from torchtitan.experiments.qwen3_vl.datasets import (
    ...     build_vl_dataloader,
    ...     preprocess_qwen_visual,
    ...     DataCollatorForSupervisedDataset,
    ... )
    >>> 
    >>> processor = Qwen3VLProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
    >>> collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
    >>> 
    >>> # Works with ANY dataset in VL_DATASETS registry!
    >>> dataloader = build_vl_dataloader(
    ...     dp_world_size=8,
    ...     dp_rank=0,
    ...     processor=processor,
    ...     preprocess_fn=preprocess_qwen_visual,  # Qwen3-VL specific
    ...     collate_fn=collator,                   # Qwen3-VL specific
    ...     job_config=job_config
    ... )
"""

# Generic VL infrastructure (EXPERIMENTAL - may move to torchtitan/hf_datasets/)
from .vl_datasets import (
    VL_DATASETS,
    HuggingFaceVLDataset,
    build_vl_dataloader,
    format_vqav2_sample,
)

# Qwen3-VL specific preprocessing (verbatim from official implementation)
from .data_processor import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    VIDEO_TOKEN_INDEX,
    DataCollatorForSupervisedDataset,
    preprocess_qwen_visual,
)
from .rope2d import get_rope_index_3

__all__ = [
    # Generic VL infrastructure (EXPERIMENTAL)
    "VL_DATASETS",
    "HuggingFaceVLDataset",
    "build_vl_dataloader",
    "format_vqav2_sample",
    # 3D RoPE position encoding (Qwen3-VL specific)
    "get_rope_index_3",
    # Preprocessing (Qwen3-VL specific)
    "preprocess_qwen_visual",
    "DataCollatorForSupervisedDataset",
    # Constants
    "IGNORE_INDEX",
    "IMAGE_TOKEN_INDEX",
    "VIDEO_TOKEN_INDEX",
]
