# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample packing utilities for Qwen3-VL multimodal datasets.

This implementation extends the base SamplePacker with VL-specific field handling:
- Properly aggregates image_grid_thw across packed samples
- Concatenates position_ids for 3D RoPE
- Handles batch-free tensor formats throughout
"""

from typing import Any, Dict, List

import torch
from torchtitan.experiments.vlm.datasets.utils.packing import SamplePacker


class VLSamplePacker(SamplePacker):
    """
    Vision-Language sample packer extending base SamplePacker.
    
    Adds VL-specific handling for:
    - image_grid_thw: Concatenates [n, 3] tensors across samples
    - position_ids: Concatenates [3, seq] tensors for 3D RoPE
    - pixel_values: Inherits list aggregation from base class
    
    All other buffer management logic reused from SamplePacker.
    """

    def __init__(
        self,
        max_seq_length: int,
        buffer_size: int = 100,
        batch_size: int = 8,
    ):
        """
        Initialize VL sample packer.
        
        Args:
            max_seq_length: Maximum sequence length after packing
            buffer_size: Number of samples to buffer before packing
            batch_size: Target batch size
        """
        super().__init__(max_seq_length, buffer_size, batch_size)
        
        # VL-specific statistics
        self.total_samples_processed = 0
        self.total_samples_packed = 0

    def _pack_buffered_samples(self) -> List[Dict[str, Any]]:
        """
        Pack buffered samples with VL-specific field handling.
        
        Overrides base class to use _create_packed_sample for VL fields.
        
        Note: Unlike base class, this does NOT skip samples exceeding max_seq_length.
        Assumes upstream filtering handles oversized samples.
        
        Returns:
            List of packed sample dicts
        """
        if not self.sample_buffer:
            return []

        # Track samples being packed
        self.total_samples_processed += len(self.sample_buffer)

        # Sort samples by length for better packing (longest first)
        samples = sorted(
            self.sample_buffer,
            key=lambda x: len(x["input_ids"]),
            reverse=True
        )

        packed_sequences = []
        current_sequence = []
        current_length = 0

        for sample in samples:
            sample_length = len(sample["input_ids"])

            # Check if adding this sample would exceed max length
            if current_sequence and (current_length + sample_length > self.max_seq_length):
                # Current sequence is full - pack it
                packed_sequences.append(self._create_packed_sample(current_sequence))
                current_sequence = []
                current_length = 0

            # Add sample to current sequence
            current_sequence.append(sample)
            current_length += sample_length

        # Handle remaining sequence
        if current_sequence:
            packed_sequences.append(self._create_packed_sample(current_sequence))

        self.sample_buffer.clear()
        self.total_samples_packed += len(packed_sequences)
        
        return packed_sequences

    def _create_packed_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a single packed sample from multiple samples.
        
        Args:
            samples: List of sample dicts to pack
            
        Returns:
            Packed sample dict with batch dimensions preserved
        """
        if len(samples) == 1:
            return samples[0]
        
        # Concatenate directly without padding (like text-only packer)
        # All inputs are batch-free tensors
        packed_input_ids = torch.cat([s["input_ids"] for s in samples], dim=0)  # [seq] → [total_seq]
        packed_labels = torch.cat([s["labels"] for s in samples], dim=0)  # [seq] → [total_seq]
        
        # Aggregate vision data (filter out None values)
        pixel_values_list = [s["pixel_values"] for s in samples if s["pixel_values"] is not None]
        image_grid_thw_list = [s["image_grid_thw"] for s in samples if s["image_grid_thw"] is not None]
        
        # Concatenate vision tensors (already batch-free)
        packed_pixel_values = None
        if pixel_values_list:
            # Concatenate along patches dimension: [p1, f] + [p2, f] → [total_p, f]
            packed_pixel_values = torch.cat(pixel_values_list, dim=0)
        
        packed_image_grid_thw = None
        if image_grid_thw_list:
            # Concatenate along image dimension: [n1, 3] + [n2, 3] → [total_n, 3]
            packed_image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        
        # Concatenate position_ids along sequence dimension
        # [3, seq1] + [3, seq2] + ... → [3, total_seq]
        packed_position_ids = None
        pos_ids_list = [s["position_ids"] for s in samples if s.get("position_ids") is not None]
        if pos_ids_list:
            packed_position_ids = torch.cat(pos_ids_list, dim=1)
        
        # All outputs are batch-free - collator will add batch dims
        return {
            "input_ids": packed_input_ids,      # [total_seq]
            "labels": packed_labels,             # [total_seq]
            "pixel_values": packed_pixel_values,  # [total_patches, feat]
            "image_grid_thw": packed_image_grid_thw,  # [total_imgs, 3]
            "position_ids": packed_position_ids,  # [3, total_seq]
        }


    def get_stats(self) -> Dict[str, Any]:
        """Get packing statistics."""
        avg_samples_per_pack = (
            self.total_samples_processed / max(1, self.total_samples_packed)
        )
        
        return {
            "total_samples_processed": self.total_samples_processed,
            "total_samples_packed": self.total_samples_packed,
            "avg_samples_per_pack": avg_samples_per_pack,
            "buffer_size": len(self.sample_buffer),
            "packed_queue_size": len(self.packed_samples),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"VLSamplePacker("
            f"processed={stats['total_samples_processed']}, "
            f"packed={stats['total_samples_packed']}, "
            f"avg_per_pack={stats['avg_samples_per_pack']:.2f}, "
            f"buffer={stats['buffer_size']})"
        )
