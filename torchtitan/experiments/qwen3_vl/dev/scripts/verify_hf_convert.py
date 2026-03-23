#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert HuggingFace Qwen3-VL checkpoint to TorchTitan format.

Usage:
    python convert_hf_checkpoint.py \
        --hf-checkpoint /checkpoints/xxie-sandbox/Qwen/Qwen3-VL-30B-A3B-Instruct \
        --output-path /path/to/output/checkpoint.pt \
        --model-flavor 30B-A3B

This script:
1. Loads HF Qwen3-VL checkpoint
2. Converts to TorchTitan format using Qwen3VLStateDictAdapter
3. Saves as PyTorch checkpoint
4. Validates conversion (optional)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, Qwen3VLProcessor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from torchtitan.experiments.qwen3_vl import (Qwen3VLModel, Qwen3VLModelArgs,
                                             Qwen3VLStateDictAdapter,
                                             qwen3_vl_args)


def detect_model_flavor(checkpoint_path: str) -> str:
    """Detect the model flavor from HF checkpoint config."""
    from transformers import AutoConfig
    
    print("Detecting model flavor from checkpoint config...")
    config = AutoConfig.from_pretrained(checkpoint_path)
    
    # Check language model config
    lang_config = config.text_config
    n_layers = lang_config.num_hidden_layers
    hidden_size = lang_config.hidden_size
    
    # Map to TorchTitan flavors
    if n_layers == 48 and hidden_size == 2048:
        flavor = "30B-A3B"
    elif n_layers == 8 and hidden_size == 256:
        flavor = "debugmodel"
    else:
        # Unknown config - default to 30B-A3B
        print(f"  ⚠️ Unknown config: {n_layers} layers, {hidden_size} hidden size")
        print("  Defaulting to '30B-A3B' flavor")
        flavor = "30B-A3B"
    
    print(f"✓ Detected flavor: {flavor} ({n_layers} layers, {hidden_size} hidden size)")
    return flavor


def load_hf_checkpoint(checkpoint_path: str):
    """Load HuggingFace checkpoint."""
    print(f"\n{'=' * 80}")
    print("LOADING HUGGINGFACE CHECKPOINT")
    print(f"{'=' * 80}\n")
    
    print(f"Loading from: {checkpoint_path}")
    
    # Load model using AutoModelForVision2Seq to handle MOE models
    print("Loading model (this may take a while for 30B model)...")
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        # No device_map = loads to CPU by default
    )
    
    # Get state dict
    state_dict = model.state_dict()
    
    print(f"✓ Loaded HF model with {len(state_dict)} keys")
    print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Show some key statistics
    vision_keys = [k for k in state_dict.keys() if "visual" in k]
    text_keys = [k for k in state_dict.keys() if "language_model" in k]
    moe_keys = [k for k in state_dict.keys() if "mlp.experts" in k or "mlp.gate" in k]
    
    print("\nKey distribution:")
    print(f"  Vision keys: {len(vision_keys)}")
    print(f"  Text keys: {len(text_keys)}")
    print(f"  MOE keys: {len(moe_keys)}")
    
    return model, state_dict


def convert_to_torchtitan(
    hf_state_dict: dict,
    model_args: Qwen3VLModelArgs,
    hf_assets_path: str,
):
    """Convert HF state dict to TorchTitan format."""
    print(f"\n{'=' * 80}")
    print("CONVERTING TO TORCHTITAN FORMAT")
    print(f"{'=' * 80}\n")
    
    # Create adapter
    adapter = Qwen3VLStateDictAdapter(model_args, hf_assets_path)
    
    # Convert
    print("Converting state dict...")
    tt_state_dict = adapter.from_hf(hf_state_dict)
    
    print(f"✓ Converted to {len(tt_state_dict)} TorchTitan keys")
    
    # Show key samples
    vision_keys = [k for k in tt_state_dict.keys() if k.startswith("visual.")]
    text_keys = [k for k in tt_state_dict.keys() if k.startswith("language_model.")]
    
    print("\nTorchTitan key distribution:")
    print(f"  Vision keys: {len(vision_keys)}")
    print(f"  Text keys: {len(text_keys)}")
    
    print("\nSample TorchTitan keys:")
    for i, key in enumerate(list(tt_state_dict.keys())[:10]):
        print(f"  {key}")
    if len(tt_state_dict) > 10:
        print(f"  ... ({len(tt_state_dict) - 10} more keys)")
    
    return tt_state_dict


def save_checkpoint(state_dict: dict, output_path: str, model_args: Qwen3VLModelArgs):
    """Save TorchTitan checkpoint."""
    print(f"\n{'=' * 80}")
    print("SAVING TORCHTITAN CHECKPOINT")
    print(f"{'=' * 80}\n")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Prepare checkpoint
    checkpoint = {
        "model": state_dict,
        "model_args": model_args.__dict__,
    }
    
    # Save
    print(f"Saving to: {output_path}")
    torch.save(checkpoint, output_path)
    
    # Get file size
    file_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"✓ Checkpoint saved ({file_size_gb:.2f} GB)")


def validate_conversion(
    hf_state_dict: dict,
    tt_state_dict: dict,
    model_args: Qwen3VLModelArgs,
    hf_assets_path: str,
):
    """Validate conversion by converting back to HF format."""
    print(f"\n{'=' * 80}")
    print("VALIDATING CONVERSION")
    print(f"{'=' * 80}\n")
    
    # Create adapter
    adapter = Qwen3VLStateDictAdapter(model_args, hf_assets_path)
    
    # Convert back to HF
    print("Converting TorchTitan → HuggingFace...")
    reconverted_hf = adapter.to_hf(tt_state_dict)
    
    print(f"✓ Reconverted to {len(reconverted_hf)} HF keys")
    
    # Compare keys (excluding weight-tied lm_head)
    original_keys = set(hf_state_dict.keys())
    reconverted_keys = set(reconverted_hf.keys())
    
    # lm_head might be missing due to weight tying
    missing_keys = original_keys - reconverted_keys
    extra_keys = reconverted_keys - original_keys
    
    print("\nKey comparison:")
    print(f"  Original HF keys: {len(original_keys)}")
    print(f"  Reconverted keys: {len(reconverted_keys)}")
    
    if missing_keys:
        print(f"  Missing keys: {len(missing_keys)}")
        for key in list(missing_keys)[:5]:
            print(f"    - {key}")
        if len(missing_keys) > 5:
            print(f"    ... ({len(missing_keys) - 5} more)")
    
    if extra_keys:
        print(f"  Extra keys: {len(extra_keys)}")
        for key in list(extra_keys)[:5]:
            print(f"    - {key}")
    
    # Compare shapes for common keys
    common_keys = original_keys & reconverted_keys
    shape_mismatches = []
    
    for key in common_keys:
        orig_shape = hf_state_dict[key].shape
        reconv_shape = reconverted_hf[key].shape
        if orig_shape != reconv_shape:
            shape_mismatches.append((key, orig_shape, reconv_shape))
    
    if shape_mismatches:
        print(f"\n❌ Shape mismatches: {len(shape_mismatches)}")
        for key, orig, reconv in shape_mismatches[:5]:
            print(f"  {key}: {orig} → {reconv}")
    else:
        print(f"\n✓ All {len(common_keys)} common keys have matching shapes!")
    
    # Validate weight tying if lm_head is missing
    if "lm_head.weight" in missing_keys and model_args.enable_weight_tying:
        print("\n✓ lm_head.weight missing due to weight tying (expected)")
    
    return len(shape_mismatches) == 0


def test_model_inference(
    tt_state_dict: dict,
    model_args: Qwen3VLModelArgs,
    hf_assets_path: str,
):
    """Test that the converted model can run inference with a real VQAv2 sample."""
    print(f"\n{'=' * 80}")
    print("TESTING MODEL INFERENCE")
    print(f"{'=' * 80}\n")
    
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create TorchTitan model
        print("Creating TorchTitan model...")
        model = Qwen3VLModel(model_args)
        
        # Load converted weights
        print("Loading converted weights...")
        model.load_state_dict(tt_state_dict)
        
        # Move model to device
        print(f"Moving model to {device}...")
        model = model.to(device)
        model.eval()
        
        print("✓ Model loaded successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Load a real sample using HuggingFaceVLDataset with PIL-aware preprocessing
        print("\nLoading real VQAv2 sample using HuggingFaceVLDataset...")
        from torchtitan.experiments.qwen3_vl.datasets import \
            HuggingFaceVLDataset
        from torchtitan.experiments.qwen3_vl.train_spec import \
            preprocess_qwen_visual_pil

        # Load processor
        processor = Qwen3VLProcessor.from_pretrained(hf_assets_path)
        
        # Create dataset with PIL-aware preprocessing (handles PIL Images from VQAv2)
        vl_dataset = HuggingFaceVLDataset(
            dataset_name="vqav2",
            dataset_path="lmms-lab/VQAv2",
            processor=processor,
            preprocess_fn=preprocess_qwen_visual_pil,  # PIL-aware version
            batch_size=1,
            seq_len=2048,
            packing_buffer_size=0,  # No packing for test
            dp_rank=0,
            dp_world_size=1,
            infinite=False,
        )
        
        # Get one processed sample
        processed = next(iter(vl_dataset))
        
        print("  Loaded sample from VQAv2 using dataset pipeline")
        
        # Extract tensors (already preprocessed by HuggingFaceVLDataset)
        input_ids = processed["input_ids"]
        pixel_values = processed["pixel_values"]
        image_grid_thw = processed["image_grid_thw"]
        
        print(f"  input_ids shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        print(f"  pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
        print(f"  image_grid_thw shape: {image_grid_thw.shape}, dtype: {image_grid_thw.dtype}")
        
        # Move data to device
        print(f"\nMoving data to {device}...")
        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device)
        image_grid_thw = image_grid_thw.to(device)
        
        # Run forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        
        # Check outputs
        print("✓ Forward pass successful!")
        print(f"  Output shape: {outputs.shape}")
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        print(f"  Expected: ({batch_size}, {seq_len}, {model_args.vocab_size})")
        
        # Verify output shape
        expected_shape = (batch_size, seq_len, model_args.vocab_size)
        if outputs.shape == expected_shape:
            print("✓ Output shape matches expected shape!")
        else:
            print("❌ Output shape mismatch!")
            print(f"  Expected: {expected_shape}")
            print(f"  Got: {outputs.shape}")
            return False
        
        # Check for NaN or Inf
        if torch.isnan(outputs).any():
            print("❌ Output contains NaN values!")
            return False
        if torch.isinf(outputs).any():
            print("❌ Output contains Inf values!")
            return False
        
        print("✓ Output values are valid (no NaN/Inf)")
        
        print("\n" + "=" * 80)
        print("✅ INFERENCE TEST PASSED WITH REAL DATA!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print("\n❌ Inference test failed!")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF Qwen3-VL checkpoint to TorchTitan format"
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=str,
        required=True,
        help="Path to HF checkpoint directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save TorchTitan checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model-flavor",
        type=str,
        default=None,
        choices=["debugmodel", "30B-A3B"],
        help="Model flavor to use for args (default: auto-detect from checkpoint)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate conversion by converting back to HF",
    )
    parser.add_argument(
        "--skip-save",
        action="store_true",
        help="Skip saving checkpoint (for testing)",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'=' * 80}")
    print("QWEN3-VL CHECKPOINT CONVERSION")
    print(f"{'=' * 80}\n")
    print(f"HF Checkpoint: {args.hf_checkpoint}")
    print(f"Output Path: {args.output_path}")
    print(f"Validate: {args.validate}")
    
    # Auto-detect model flavor if not specified
    if args.model_flavor is None:
        print("\nAuto-detecting model flavor...")
        model_flavor = detect_model_flavor(args.hf_checkpoint)
    else:
        model_flavor = args.model_flavor
        print(f"\nUsing specified model flavor: {model_flavor}")
    
    # Get model args
    model_args = qwen3_vl_args[model_flavor]
    
    print(f"\nUsing TorchTitan model flavor: {model_flavor}")
    print("Model configuration:")
    print(f"  Layers: {model_args.n_layers}")
    print(f"  Hidden size: {model_args.dim}")
    print(f"  Heads: {model_args.n_heads}")
    print(f"  Vocab size: {model_args.vocab_size}")
    print(f"  MOE enabled: {model_args.moe_enabled}")
    if model_args.moe_enabled:
        print(f"  Num experts: {model_args.moe_args.num_experts}")
        print(f"  Top-k: {model_args.moe_args.top_k}")
    
    # Load HF checkpoint
    hf_model, hf_state_dict = load_hf_checkpoint(args.hf_checkpoint)
    
    # Convert to TorchTitan
    tt_state_dict = convert_to_torchtitan(
        hf_state_dict,
        model_args,
        args.hf_checkpoint,
    )
    
    # Test model inference (always run this)
    print("\n" + "=" * 80)
    print("STEP 1: Testing model inference with converted weights")
    print("=" * 80)
    inference_ok = test_model_inference(
        tt_state_dict,
        model_args,
        args.hf_checkpoint,
    )
    
    if not inference_ok:
        print("\n❌ Inference test failed! Model forward pass has issues.")
        return 1
    
    # Validate if requested
    if args.validate:
        print("\n" + "=" * 80)
        print("STEP 2: Validating conversion (optional)")
        print("=" * 80)
        valid = validate_conversion(
            hf_state_dict,
            tt_state_dict,
            model_args,
            args.hf_checkpoint,
        )
        if not valid:
            print("\n❌ Validation failed! Check shape mismatches above.")
            return 1
    
    # Save checkpoint
    if not args.skip_save:
        save_checkpoint(tt_state_dict, args.output_path, model_args)
    else:
        print("\n⚠️  Skipping checkpoint save (--skip-save flag)")
    
    print(f"\n{'=' * 80}")
    print("CONVERSION COMPLETE!")
    print(f"{'=' * 80}\n")
    
    print("Next steps:")
    print("  1. Load checkpoint in TorchTitan:")
    print(f"     checkpoint = torch.load('{args.output_path}')")
    print("     model.load_state_dict(checkpoint['model'])")
    print("  2. Start training with TorchTitan")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
