# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample packing utilities for Qwen3-VL multimodal datasets.

This implementation handles variable-length VL samples by:
1. Padding samples to a common length before concatenation
2. Properly handling VL-specific fields (pixel_values, image_grid_thw, position_ids)
3. Maintaining correct masking for padded regions
"""

from collections import deque
from typing import Any, Dict, List, Optional

import torch


class VLSamplePacker:
    """
    Packs multiple VL samples together with proper padding.
    
    Key differences from generic packing:
    - Pads input_ids and labels to common length
    - Aggregates pixel_values and image_grid_thw lists
    - Recomputes position_ids for packed sequences
    - Handles attention masks correctly
    """

    def __init__(
        self,
        max_seq_length: int,
        buffer_size: int = 100,
        batch_size: int = 8,
        pad_token_id: int = 151643,  # Qwen3 pad token
        ignore_index: int = -100,
    ):
        """
        Initialize sample packer.
        
        Args:
            max_seq_length: Maximum sequence length after packing
            buffer_size: Number of samples to buffer before packing
            batch_size: Target batch size
            pad_token_id: Token ID for padding
            ignore_index: Label value for ignored tokens
        """
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        # Initialize buffers
        self.sample_buffer: deque = deque()
        self.packed_samples: deque = deque()
        
        # Statistics
        self.total_samples_processed = 0
        self.total_samples_packed = 0
        self.total_tokens_padded = 0

    def _pad_sample(
        self, 
        sample: Dict[str, Any], 
        target_length: int
    ) -> Dict[str, Any]:
        """
        Pad a single sample to target length.
        
        Args:
            sample: Sample dict with input_ids, labels, etc.
            target_length: Target sequence length
            
        Returns:
            Padded sample dict with batch dimensions preserved
        """
        # Work with batch dimensions directly - no squeeze needed!
        input_ids = sample["input_ids"]  # [1, seq]
        labels = sample["labels"]        # [1, seq]
        
        current_length = input_ids.shape[1]  # Sequence is on dim=1
        pad_size = target_length - current_length
        self.total_tokens_padded += pad_size
        
        # Pad along sequence dimension (dim=1)
        input_ids = torch.cat([
            input_ids,
            torch.full((1, pad_size), self.pad_token_id, dtype=torch.long)
        ], dim=1)
        labels = torch.cat([
            labels,
            torch.full((1, pad_size), self.ignore_index, dtype=torch.long)
        ], dim=1)
        
        # Pad position_ids along sequence dimension (dim=2)
        position_ids = None
        if sample.get("position_ids") is not None:
            pos_ids = sample["position_ids"]  # [3, 1, seq]
            # Pad along last dimension (sequence)
            pad_positions = torch.zeros((3, 1, pad_size), dtype=torch.long)
            position_ids = torch.cat([pos_ids, pad_positions], dim=2)
        
        return {
            "input_ids": input_ids,      # [1, padded_seq]
            "labels": labels,              # [1, padded_seq]
            "pixel_values": sample["pixel_values"],
            "image_grid_thw": sample["image_grid_thw"],
            "position_ids": position_ids,  # [3, 1, padded_seq]
        }

    def _pack_buffered_samples(self) -> List[Dict[str, Any]]:
        """
        Pack buffered samples into optimal sequences with padding.
        
        Returns:
            List of packed sample dicts
        """
        if not self.sample_buffer:
            return []

        # Sort samples by length for better packing (longest first)
        samples = sorted(
            self.sample_buffer, 
            key=lambda x: x["input_ids"].shape[1],  # Sequence length on dim=1
            reverse=True
        )

        packed_sequences = []
        current_sequence = []
        current_length = 0

        for sample in samples:
            sample_length = sample["input_ids"].shape[1]  # Sequence length on dim=1

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
        
        # Find max length and pad all samples
        max_length = max(s["input_ids"].shape[1] for s in samples)  # Sequence on dim=1
        padded_samples = [self._pad_sample(s, max_length) for s in samples]
        
        # Concatenate text data along sequence dimension (dim=1)
        # [1, seq1] + [1, seq2] + ... → [1, total_seq]
        packed_input_ids = torch.cat([s["input_ids"] for s in padded_samples], dim=1)
        packed_labels = torch.cat([s["labels"] for s in padded_samples], dim=1)
        
        # Aggregate vision data (filter out None values)
        pixel_values_list = [s["pixel_values"] for s in padded_samples if s["pixel_values"] is not None]
        image_grid_thw_list = [s["image_grid_thw"] for s in padded_samples if s["image_grid_thw"] is not None]
        
        # Concatenate vision tensors
        packed_pixel_values = None
        if pixel_values_list:
            # Squeeze first, then cat: [1,p,f] → [p,f], then cat along dim=0
            # (feature dim can vary per sample, can't cat along dim=1)
            squeezed = [pv.squeeze(0) for pv in pixel_values_list]
            packed_pixel_values = torch.cat(squeezed, dim=0)  # [total_p, f]
        
        packed_image_grid_thw = None
        if image_grid_thw_list:
            # Cat along image dimension: [n1,3] + [n2,3] → [total_n,3]
            packed_image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        
        # Concatenate position_ids along sequence dimension (dim=2)
        # [3, 1, seq1] + [3, 1, seq2] + ... → [3, 1, total_seq]
        packed_position_ids = None
        pos_ids_list = [s["position_ids"] for s in padded_samples if s.get("position_ids") is not None]
        if pos_ids_list:
            packed_position_ids = torch.cat(pos_ids_list, dim=2)
        
        # No unsqueeze needed - dimensions already correct!
        return {
            "input_ids": packed_input_ids,      # [1, total_seq]
            "labels": packed_labels,             # [1, total_seq]
            "pixel_values": packed_pixel_values,  # [total_patches, feat]
            "image_grid_thw": packed_image_grid_thw,  # [total_imgs, 3]
            "position_ids": packed_position_ids,  # [3, 1, total_seq]
        }

    def add_sample(self, sample: Dict[str, Any]) -> None:
        """
        Add a sample to the packing buffer.
        
        Args:
            sample: Sample dict to add
        """
        self.sample_buffer.append(sample)
        self.total_samples_processed += 1

        # Pack when buffer is full
        if len(self.sample_buffer) >= self.buffer_size:
            packed = self._pack_buffered_samples()
            self.packed_samples.extend(packed)

    def has_batch_ready(self) -> bool:
        """Check if a full batch of packed samples is ready."""
        return len(self.packed_samples) >= self.batch_size

    def get_next_batch(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get next batch of packed samples if available.
        
        Returns:
            List of packed sample dicts, or None if not enough samples
        """
        if not self.has_batch_ready():
            # Try to pack remaining samples
            if self.sample_buffer:
                packed = self._pack_buffered_samples()
                self.packed_samples.extend(packed)

            if not self.has_batch_ready():
                return None

        # Extract batch
        batch = []
        for _ in range(self.batch_size):
            if not self.packed_samples:
                break
            batch.append(self.packed_samples.popleft())

        return batch

    def flush(self) -> List[Dict[str, Any]]:
        """
        Flush remaining samples from buffer.
        
        Returns:
            List of remaining packed samples
        """
        if self.sample_buffer:
            packed = self._pack_buffered_samples()
            self.packed_samples.extend(packed)
        
        remaining = list(self.packed_samples)
        self.packed_samples.clear()
        return remaining

    def get_stats(self) -> Dict[str, Any]:
        """Get packing statistics."""
        avg_samples_per_pack = (
            self.total_samples_processed / max(1, self.total_samples_packed)
        )
        avg_padding_per_token = (
            self.total_tokens_padded / max(1, self.total_samples_processed)
        )
        
        return {
            "total_samples_processed": self.total_samples_processed,
            "total_samples_packed": self.total_samples_packed,
            "total_tokens_padded": self.total_tokens_padded,
            "avg_samples_per_pack": avg_samples_per_pack,
            "avg_padding_per_token": avg_padding_per_token,
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
