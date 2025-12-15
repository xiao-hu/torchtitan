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
from torchtitan.tools.logging import logger


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

    def _squeeze_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Remove batch dimension [1, ...] -> [...]"""
        while tensor.dim() > 1 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

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
            Padded sample dict
        """
        # Squeeze batch dims: [1, seq_len] -> [seq_len]
        input_ids = self._squeeze_batch_dim(sample["input_ids"])
        labels = self._squeeze_batch_dim(sample["labels"])
        
        current_length = input_ids.shape[0]
        pad_size = target_length - current_length
        self.total_tokens_padded += pad_size
        
        # Pad input_ids and labels
        input_ids = torch.cat([
            input_ids,
            torch.full((pad_size,), self.pad_token_id, dtype=torch.long)
        ])
        labels = torch.cat([
            labels,
            torch.full((pad_size,), self.ignore_index, dtype=torch.long)
        ])
        
        # Pad position_ids: [3, 1, seq_len] -> [3, seq_len] + padding
        position_ids = None
        if sample.get("position_ids") is not None:
            pos_ids = sample["position_ids"]
            # Squeeze batch dim: [3, 1, seq_len] -> [3, seq_len]
            if pos_ids.dim() == 3 and pos_ids.shape[1] == 1:
                pos_ids = pos_ids.squeeze(1)
            # Pad along sequence dimension
            pad_positions = torch.zeros((3, pad_size), dtype=torch.long)
            position_ids = torch.cat([pos_ids, pad_positions], dim=1)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": sample["pixel_values"],
            "image_grid_thw": sample["image_grid_thw"],
            "position_ids": position_ids,
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
            key=lambda x: self._squeeze_batch_dim(x["input_ids"]).shape[0],
            reverse=True
        )

        packed_sequences = []
        current_sequence = []
        current_length = 0

        for sample in samples:
            sample_length = self._squeeze_batch_dim(sample["input_ids"]).shape[0]

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

    def _flatten_vision_data(self, data_list: List, is_pixel_values: bool = False) -> List:
        """
        Flatten nested lists/tensors of vision data.
        
        Args:
            data_list: List of vision data (pixel_values or image_grid_thw)
            is_pixel_values: If True, keep 2D tensors intact; else split into rows
        """
        flattened = []
        for item in data_list:
            if item is None:
                continue
            elif isinstance(item, list):
                flattened.extend(self._flatten_vision_data(item, is_pixel_values))
            elif torch.is_tensor(item):
                # Squeeze batch dim: [1, ...] -> [...]
                if item.dim() == 3 and item.shape[0] == 1:
                    item = item.squeeze(0)
                
                if item.dim() == 2:
                    if is_pixel_values:
                        # pixel_values: keep as 2D [num_patches, embed_dim]
                        flattened.append(item)
                    else:
                        # image_grid_thw: split [num_images, 3] -> multiple [3]
                        flattened.extend([item[i] for i in range(item.shape[0])])
                else:
                    flattened.append(item)
        return flattened

    def _create_packed_sample(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a single packed sample from multiple samples.
        
        Args:
            samples: List of sample dicts to pack
            
        Returns:
            Packed sample dict
        """
        if len(samples) == 1:
            return samples[0]
        
        # Find max length and pad all samples
        max_length = max(self._squeeze_batch_dim(s["input_ids"]).shape[0] for s in samples)
        padded_samples = [self._pad_sample(s, max_length) for s in samples]
        
        # Concatenate text data
        packed_input_ids = torch.cat([s["input_ids"] for s in padded_samples])
        packed_labels = torch.cat([s["labels"] for s in padded_samples])
        
        # Aggregate vision data
        all_pixel_values = [s["pixel_values"] for s in padded_samples if s["pixel_values"] is not None]
        all_image_grid_thw = [s["image_grid_thw"] for s in padded_samples if s["image_grid_thw"] is not None]
        
        # Flatten into lists of tensors
        packed_pixel_values_list = self._flatten_vision_data(all_pixel_values, is_pixel_values=True) if all_pixel_values else []
        packed_image_grid_thw_list = self._flatten_vision_data(all_image_grid_thw, is_pixel_values=False) if all_image_grid_thw else []
        
        # Concatenate lists into single tensors for collator
        packed_pixel_values = None
        if packed_pixel_values_list:
            # Concatenate 2D tensors [num_patches, embed_dim] along batch dim -> [total_patches, embed_dim]
            packed_pixel_values = torch.cat(packed_pixel_values_list, dim=0)
        
        packed_image_grid_thw = None
        if packed_image_grid_thw_list:
            # Stack 1D tensors [3] along batch dim -> [num_images, 3]
            packed_image_grid_thw = torch.stack(packed_image_grid_thw_list, dim=0)
        
        # Concatenate position_ids: [3, seq_len] + [3, seq_len] -> [3, total_seq_len]
        packed_position_ids = None
        pos_ids_list = [s["position_ids"] for s in padded_samples if s.get("position_ids") is not None]
        if pos_ids_list:
            # All position_ids are already [3, seq_len] from _pad_sample
            packed_position_ids = torch.cat(pos_ids_list, dim=1)
        
        # Restore batch dimensions for collator
        return {
            "input_ids": packed_input_ids.unsqueeze(0),  # [seq] -> [1, seq]
            "labels": packed_labels.unsqueeze(0),  # [seq] -> [1, seq]
            "pixel_values": packed_pixel_values,
            "image_grid_thw": packed_image_grid_thw,
            "position_ids": packed_position_ids.unsqueeze(1) if packed_position_ids is not None else None,  # [3, seq] -> [3, 1, seq]
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
