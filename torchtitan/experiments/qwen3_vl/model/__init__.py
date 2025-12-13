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
from .vision import Qwen3VLVisionEncoder
from .model import Qwen3VLModel, Qwen3VLTextModel

__all__ = [
    'Qwen3VLModelArgs',
    'Qwen3VLVisionArgs',
    'SpecialTokens',
    'Qwen3VLVisionEncoder',
    'Qwen3VLModel',
    'Qwen3VLTextModel',
]
