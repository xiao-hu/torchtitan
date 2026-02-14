# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL MOE Model Implementation

Contains the core model architecture combining:
- Vision encoder (SigLIP-2)
- Text decoder (Qwen3 with MOE)
- Projector for vision-to-text feature mapping
- DeepStack integration for multi-layer visual features
"""

from .args import Qwen3VLModelArgs, Qwen3VLVisionArgs, SpecialTokens
from .model import Qwen3VLModel, Qwen3VLTextModel
from .state_dict_adapter import Qwen3VLStateDictAdapter
from .vision import Qwen3VLVisionEncoder

__all__ = [
    "Qwen3VLModel",
    "Qwen3VLTextModel",
    "Qwen3VLModelArgs",
    "Qwen3VLVisionArgs",
    "Qwen3VLVisionEncoder",
    "Qwen3VLStateDictAdapter",
    "SpecialTokens",
]
