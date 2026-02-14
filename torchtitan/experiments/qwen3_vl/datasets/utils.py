# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch-free preprocessing and collation utilities for Qwen3-VL.

These functions work with batch-free tensors throughout the pipeline:
1. Preprocessing: Squeeze batch dims from processor output
2. Packing: Concatenate batch-free tensors
3. Collation: Add batch dims back when creating training batches

This design matches the text-only packer and provides dimensional consistency.
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from torchtitan.experiments.qwen3_vl.model.model import get_rope_index


def preprocess_qwen_visual_pil(sources, processor):
    """
    PIL-aware preprocessing for Qwen3-VL models with batch-free output.
    
    Expects sources to be formatted with processor-ready messages:
        sources[0]["messages"] = [
            {"role": "user", "content": [{"type": "image", "image": PIL.Image}, {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": "..."}
        ]
    
    Args:
        sources: List of formatted samples with "messages" field
        processor: Qwen3VLProcessor instance
        
    Returns:
        Dict with batch-free tensors:
        - input_ids: [seq]
        - labels: [seq]
        - pixel_values: [patches, feat]
        - image_grid_thw: [imgs, 3]
        - position_ids: [3, seq]
    """
    IGNORE_INDEX = -100
    
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")
    
    source = sources[0]
    
    # Extract processor-ready messages (already formatted by format_vqav2_sample)
    messages = source.get("messages", [])
    
    # Extract PIL images from messages for processor
    pil_images = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    pil_images.append(item["image"])
    
    # Use processor to handle PIL images
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_result = processor(
        text=[text],
        images=pil_images if pil_images else None,
        return_tensors="pt",
    )
    
    # Processor returns tensors with batch dimension since we pass text=[text]
    # Squeeze to get batch-free tensors for packing
    
    # Ensure correct dtypes and squeeze batch dimension
    input_ids = full_result["input_ids"].long().squeeze(0)  # [1, seq] → [seq]
    
    # Ensure image_grid_thw has correct dtype (already [num_images, 3], no batch dim)
    if "image_grid_thw" in full_result and full_result["image_grid_thw"] is not None:
        full_result["image_grid_thw"] = full_result["image_grid_thw"].long()
    
    # Create labels (mask everything except assistant responses)
    labels = torch.full_like(input_ids, IGNORE_INDEX)  # [seq]
    
    # Find assistant response tokens (between <|im_start|>assistant and <|im_end|>)
    # Token IDs: <|im_start|>=151644, user=872, assistant=77091, <|im_end|>=151645, \n=198
    input_ids_flat = input_ids.tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L - 1:
        # Look for <|im_start|> followed by assistant
        if input_ids_flat[pos] == 151644 and input_ids_flat[pos + 1] == 77091:
            # Found assistant turn
            ans_start = pos + 2  # Skip <|im_start|> and "assistant"
            
            # Skip the newline if present (it's part of the template prompt, not model output)
            if ans_start < L and input_ids_flat[ans_start] == 198:  # 198 is '\n'
                ans_start += 1
            
            ans_end = ans_start
            # Find the closing <|im_end|>
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                # Unmask assistant response including <|im_end|>
                labels[ans_start : ans_end + 1] = input_ids[ans_start : ans_end + 1]
                pos = ans_end
        pos += 1
    
    # Calculate position_ids for 3D RoPE
    position_ids, _ = get_rope_index(
        input_ids=input_ids.unsqueeze(0),  # [seq] → [1, seq] for get_rope_index
        image_grid_thw=full_result.get("image_grid_thw"),
        video_grid_thw=full_result.get("video_grid_thw"),
        attention_mask=None,
        spatial_merge_size=2,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
    )  # Returns [3, 1, seq]
    
    # Squeeze batch dimension from position_ids for consistency
    position_ids = position_ids.squeeze(1)  # [3, 1, seq] → [3, seq]
    
    # Return batch-free tensors - collator will add batch dims when batching
    full_result["input_ids"] = input_ids  # [seq]
    full_result["labels"] = labels        # [seq]
    full_result["position_ids"] = position_ids  # [3, seq]
    
    return full_result


def collate_vl_batch(instances, tokenizer):
    """
    Custom collator that handles batch-free tensors from preprocessing/packing.
    
    Input shapes (per instance):
    - input_ids: [seq]
    - labels: [seq]
    - pixel_values: [patches, feat]
    - image_grid_thw: [imgs, 3]
    - position_ids: [3, seq]
    
    Output shapes (batched):
    - input: [batch, max_seq]
    - labels: [batch, max_seq]
    - pixel_values: [total_patches, feat]
    - image_grid_thw: [total_imgs, 3]
    - position_ids: [3, batch, max_seq]
    - attention_mask: [batch, max_seq]
    
    Args:
        instances: List of sample dicts from dataset
        tokenizer: Qwen3VLProcessor with tokenizer
        
    Returns:
        Tuple of (batch_dict, labels) for TorchTitan training loop
    """
    IGNORE_INDEX = -100
    
    # Extract fields from instances
    input_ids = [inst["input_ids"] for inst in instances]
    labels = [inst["labels"] for inst in instances]
    position_ids = [inst["position_ids"] for inst in instances]
    
    # Pad sequences to max length in batch, rounded up to multiple of 512
    # This significantly reduces torch.compile recompilations while keeping memory overhead low
    PADDING_MULTIPLE = 256
    
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.tokenizer.pad_token_id
    )
    labels = pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    
    # Round up to nearest multiple of PADDING_MULTIPLE
    max_len = input_ids.shape[1]
    max_len = ((max_len + PADDING_MULTIPLE - 1) // PADDING_MULTIPLE) * PADDING_MULTIPLE
    
    # Apply padding if needed
    if input_ids.shape[1] < max_len:
        pad_len = max_len - input_ids.shape[1]
        input_ids = torch.nn.functional.pad(
            input_ids, (0, pad_len), value=tokenizer.tokenizer.pad_token_id
        )
        labels = torch.nn.functional.pad(
            labels, (0, pad_len), value=IGNORE_INDEX
        )
    padded_pos_ids = []
    for pos_id in position_ids:
        # pos_id is [3, seq], pad to [3, max_len]
        pad_len = max_len - pos_id.shape[1]
        if pad_len > 0:
            padded = torch.nn.functional.pad(pos_id, (0, pad_len), value=0)
        else:
            padded = pos_id[:, :max_len]
        padded_pos_ids.append(padded)
    # Stack to [batch, 3, max_len], then permute to [3, batch, max_len]
    position_ids = torch.stack(padded_pos_ids, dim=0).permute(1, 0, 2)
    
    # Truncate to model_max_length if needed
    if hasattr(tokenizer.tokenizer, 'model_max_length'):
        max_model_len = tokenizer.tokenizer.model_max_length
        input_ids = input_ids[:, :max_model_len]
        labels = labels[:, :max_model_len]
        position_ids = position_ids[:, :, :max_model_len]
    
    # Build batch dict
    batch = {
        "input": input_ids,  # TorchTitan expects "input"
        "attention_mask": input_ids.ne(tokenizer.tokenizer.pad_token_id),
        "position_ids": position_ids,
    }
    
    # Aggregate vision data (already batch-free, just concatenate)
    pixel_values_list = [
        inst["pixel_values"] for inst in instances 
        if "pixel_values" in inst and inst["pixel_values"] is not None
    ]
    image_grid_thw_list = [
        inst["image_grid_thw"] for inst in instances
        if "image_grid_thw" in inst and inst["image_grid_thw"] is not None
    ]
    
    if pixel_values_list:
        batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        batch["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
    else:
        batch["pixel_values"] = None
        batch["image_grid_thw"] = None
    
    # TorchTitan expects (input_dict, labels)
    return batch, labels
