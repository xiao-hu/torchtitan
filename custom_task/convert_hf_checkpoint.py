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
import json
import os
import sys
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from torchtitan.experiments.qwen3_vl import (
    Qwen3VLModelArgs,
    Qwen3VLStateDictAdapter,
    qwen3_vl_args,
)


def load_hf_checkpoint(checkpoint_path: str):
    """Load HuggingFace checkpoint."""
    print(f"\n{'=' * 80}")
    print("LOADING HUGGINGFACE CHECKPOINT")
    print(f"{'=' * 80}\n")
    
    print(f"Loading from: {checkpoint_path}")
    
    # Load model
    print("Loading model (this may take a while for 30B model)...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
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
    
    print(f"\nKey distribution:")
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
    
    print(f"\nTorchTitan key distribution:")
    print(f"  Vision keys: {len(vision_keys)}")
    print(f"  Text keys: {len(text_keys)}")
    
    print(f"\nSample TorchTitan keys:")
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
    
    print(f"\nKey comparison:")
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
        default="30B-A3B",
        choices=["debugmodel", "30B-A3B"],
        help="Model flavor to use for args",
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
    print(f"Model Flavor: {args.model_flavor}")
    print(f"Validate: {args.validate}")
    
    # Get model args
    model_args = qwen3_vl_args[args.model_flavor]
    
    print(f"\nModel configuration:")
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
    
    # Validate if requested
    if args.validate:
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
    print(f"  1. Load checkpoint in TorchTitan:")
    print(f"     checkpoint = torch.load('{args.output_path}')")
    print(f"     model.load_state_dict(checkpoint['model'])")
    print(f"  2. Start training with TorchTitan")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
