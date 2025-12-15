# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TrainSpec for Qwen3-VL Vision-Language Model.

Integrates Qwen3-VL with TorchTitan's training loop by providing:
- Model instantiation (Qwen3VLModel)
- Tokenizer/Processor loading (Qwen3VLProcessor)
- Dataloader construction with VL-specific preprocessing
- Model converter for checkpoint conversion
"""

from transformers import Qwen3VLProcessor

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.validate import build_validator
from torchtitan.config import JobConfig
from torchtitan.experiments.qwen3_vl import Qwen3VLModel, qwen3_vl_args
from torchtitan.experiments.qwen3_vl.datasets import (
    DataCollatorForSupervisedDataset, build_vl_dataloader)
from torchtitan.experiments.qwen3_vl.infra.parallelize import \
    parallelize_qwen3_vl
from torchtitan.experiments.qwen3_vl.model.model import get_rope_index
from torchtitan.experiments.qwen3_vl.model.state_dict_adapter import \
    Qwen3VLStateDictAdapter
from torchtitan.protocols.train_spec import TrainSpec


def preprocess_qwen_visual_pil(sources, processor):
    """
    PIL-aware preprocessing for Qwen3-VL models.
    
    This is an alternative to preprocess_qwen_visual from data_processor.py that
    handles PIL Images directly instead of requiring file paths. Use this for datasets
    like VQAv2 that provide PIL Images in memory.
    
    Args:
        sources: List of formatted samples with PIL Images in 'image' field
        processor: Qwen3VLProcessor instance
        
    Returns:
        Dict with input_ids, pixel_values, image_grid_thw, and labels
    """
    import torch
    
    IGNORE_INDEX = -100
    
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")
    
    source = sources[0]
    
    # Extract PIL images
    pil_images = source.get("image", [])
    if not isinstance(pil_images, list):
        pil_images = [pil_images]
    
    # Build messages for processor (using PIL Images directly)
    conversations = source.get("conversations", [])
    messages = []
    
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]
        
        if role == "user":
            # Build content with PIL images
            content = []
            
            # Handle <image> placeholders in text
            text_parts = text.split("<image>")
            
            image_idx = 0
            for i, part in enumerate(text_parts):
                if i > 0 and image_idx < len(pil_images):
                    # Insert image before this text part
                    content.append({"type": "image", "image": pil_images[image_idx]})
                    image_idx += 1
                
                if part.strip():
                    content.append({"type": "text", "text": part.strip()})
            
            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": text})
    
    # Use processor to handle PIL images
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_result = processor(
        text=[text],
        images=pil_images if pil_images else None,
        return_tensors="pt",
    )
    
    # Ensure correct dtypes and shapes
    input_ids = full_result["input_ids"].long()  # Must be Long for embedding
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    
    # Ensure pixel_values has correct shape (add batch dim if missing)
    if "pixel_values" in full_result and full_result["pixel_values"] is not None:
        pixel_values = full_result["pixel_values"]
        if pixel_values.dim() == 2:
            # Missing batch dim - add it
            pixel_values = pixel_values.unsqueeze(0)
        full_result["pixel_values"] = pixel_values
    
    # Ensure image_grid_thw has correct dtype
    # Processor always returns [num_images, 3] for image_grid_thw
    if "image_grid_thw" in full_result and full_result["image_grid_thw"] is not None:
        full_result["image_grid_thw"] = full_result["image_grid_thw"].long()
    
    # Create labels (mask everything except assistant responses)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    
    # Find assistant response tokens (between <|im_start|>assistant and <|im_end|>)
    # Token IDs: <|im_start|>=151644, user=872, assistant=77091, <|im_end|>=151645, \n=198
    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L - 1:
        # Look for <|im_start|> followed by assistant
        if input_ids_flat[pos] == 151644 and input_ids_flat[pos + 1] == 77091:
            # Found assistant turn
            # The chat template adds '\n' after '<|im_start|>assistant' as part of generation prompt
            # So we should unmask from AFTER the newline (what model actually generates)
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
                labels[0, ans_start : ans_end + 1] = input_ids[0, ans_start : ans_end + 1]
                pos = ans_end
        pos += 1
    
    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    
    # Calculate position_ids for 3D RoPE (required by collator)
    # This is done during data loading for efficiency (computed once, reused across epochs)
    position_ids, _ = get_rope_index(
        input_ids=input_ids,
        image_grid_thw=full_result.get("image_grid_thw"),  # Already normalized above
        video_grid_thw=full_result.get("video_grid_thw"),
        attention_mask=None,  # Will be computed from input_ids
        spatial_merge_size=2,  # Default from model config
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
    )
    full_result["position_ids"] = position_ids
    
    return full_result


def build_qwen3vl_tokenizer(job_config: JobConfig):
    """
    Load Qwen3-VL processor (tokenizer + image processor).
    
    Note: Returns Qwen3VLProcessor (not just tokenizer) which includes:
    - tokenizer: For text processing
    - image_processor: For vision input processing
    
    Args:
        job_config: TorchTitan job configuration
        
    Returns:
        Qwen3VLProcessor instance
    """
    processor = Qwen3VLProcessor.from_pretrained(
        job_config.model.hf_assets_path,
        trust_remote_code=True,
    )
    
    # Ensure padding side is correct for training
    processor.tokenizer.padding_side = "right"
    
    return processor


def build_qwen3vl_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer,  # Actually a Qwen3VLProcessor
    job_config: JobConfig,
):
    """
    Build VL dataloader with Qwen3-VL specific preprocessing.
    
    This function encapsulates:
    - Processor (tokenizer + image processor)
    - Model-specific preprocessing (preprocess_qwen_visual)
    - Collator for batching (DataCollatorForSupervisedDataset)
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Qwen3VLProcessor (not just tokenizer!)
        job_config: TorchTitan job configuration
        
    Returns:
        ParallelAwareDataloader with VL-specific components
    """

    # Create base collator
    base_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer.tokenizer
    )
    
    # Wrap collator to match TorchTitan's expected format: (input_dict, labels)
    def collator(instances):
        """
        Wrapper that adapts Qwen3-VL collator output to TorchTitan format.
        
        Qwen3-VL collator returns: dict with labels inside
        TorchTitan expects: tuple of (input_dict, labels)
        """
        batch = base_collator(instances)
        # Extract labels and rename input_ids to input
        labels = batch.pop("labels")
        batch["input"] = batch.pop("input_ids")
        return batch, labels
    
    # Build dataloader with model-specific preprocessing
    return build_vl_dataloader(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        processor=tokenizer,
        preprocess_fn=preprocess_qwen_visual_pil,
        collate_fn=collator,
        job_config=job_config,
        infinite=True,  # Loop dataset infinitely for training
    )


# Define TrainSpec for Qwen3-VL
qwen3_vl_train_spec = TrainSpec(
    model_cls=Qwen3VLModel,
    model_args=qwen3_vl_args,
    parallelize_fn=parallelize_qwen3_vl,  # Use Qwen3-VL parallelization
    pipelining_fn=None,  # No pipeline parallelism for now
    build_optimizers_fn=build_optimizers,
    build_lr_schedulers_fn=build_lr_schedulers,
    build_dataloader_fn=build_qwen3vl_dataloader,
    build_tokenizer_fn=build_qwen3vl_tokenizer,
    build_loss_fn=build_cross_entropy_loss,
    build_validator_fn=build_validator,
    state_dict_adapter=Qwen3VLStateDictAdapter,
)
