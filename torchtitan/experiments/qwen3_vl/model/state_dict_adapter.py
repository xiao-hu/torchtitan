# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3 VL State Dict Adapter

Handles checkpoint conversion between HuggingFace and TorchTitan formats for Qwen3-VL.

Extends Qwen3StateDictAdapter to add vision encoder mappings while reusing
all text model and MOE handling from the base class.

HF checkpoint structure:
- model.visual.* (vision encoder - HF Qwen3VLVisionModel)
- model.* (text decoder with MOE - same as Qwen3)

TorchTitan structure:
- visual.* (vision encoder wrapper)
- language_model.* (text decoder - Qwen3VLTextModel extends Qwen3Model)

Key differences from base Qwen3:
1. Vision encoder uses HF directly (no key mapping needed for visual.*)
2. Text model is nested under language_model.* instead of direct layers.*
"""

from typing import Any

from torchtitan.models.qwen3.model.state_dict_adapter import \
    Qwen3StateDictAdapter

from .args import Qwen3VLModelArgs


class Qwen3VLStateDictAdapter(Qwen3StateDictAdapter):
    """
    State dict adapter for Qwen3-VL multimodal model.
    
    Extends Qwen3StateDictAdapter to handle vision encoder while reusing
    text model and MOE conversion logic.
    """
    
    def __init__(self, model_args: Qwen3VLModelArgs, hf_assets_path: str | None):
        # Initialize parent with text model args
        super().__init__(model_args, hf_assets_path)
        
        # Add Qwen3-VL specific mappings
        # Vision encoder: We use HF directly, so these keys pass through unchanged
        # Text model: Add language_model prefix to existing mappings
        self._update_mappings_for_vl()
    
    def _update_mappings_for_vl(self):
        """
        Update key mappings to account for Qwen3-VL structure.
        
        Vision keys: HF model used directly, keys pass through
        Text keys: Handled by from_hf/to_hf methods with language_model prefix
        """
        # Vision encoder keys: HF model used directly
        # Pattern: model.visual.* -> visual.*
        vision_keys = [
            "model.visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.bias",
        ]
        for vkey in vision_keys:
            self.from_hf_map[vkey] = vkey.replace("model.", "")
    
    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert TorchTitan state dict to HuggingFace format.
        
        Handles:
        1. Vision encoder keys (pass through with model. prefix)
        2. Text model keys (remove language_model., use parent logic, add model.language_model.)
        3. MOE expert weights (handled by parent)
        
        Args:
            state_dict: TorchTitan model state dict
            
        Returns:
            HuggingFace-formatted state dict
        """
        # Separate vision and text model keys
        vision_dict = {}
        text_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("visual."):
                # Vision encoder: add model. prefix
                hf_key = f"model.{key}"
                vision_dict[hf_key] = value
            elif key.startswith("language_model."):
                # Text model: remove language_model. prefix for parent processing
                stripped_key = key.replace("language_model.", "", 1)
                text_dict[stripped_key] = value
            else:
                # Unknown key, pass through
                text_dict[key] = value
        
        # Convert text model using parent logic (returns model.* keys)
        hf_text_dict = super().to_hf(text_dict)
        
        # Add language_model infix to text model keys: model.* -> model.language_model.*
        final_text_dict = {}
        for key, value in hf_text_dict.items():
            if key.startswith("model."):
                # Insert language_model after model.
                new_key = key.replace("model.", "model.language_model.", 1)
                final_text_dict[new_key] = value
            else:
                # Keys like lm_head.weight stay as is
                final_text_dict[key] = value
        
        # Merge vision and text dictionaries
        hf_state_dict = {**vision_dict, **final_text_dict}
        
        return hf_state_dict
    
    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert HuggingFace state dict to TorchTitan format.
        
        Handles:
        1. Vision encoder keys (strip model. prefix)
        2. Text model keys (strip model.language_model., convert GroupedExperts directly, use parent for non-MOE)
        3. GroupedExperts weights (convert directly without splitting - both HF and TorchTitan use GroupedExperts)
        
        Note: Qwen3-VL-MOE HF checkpoint uses GroupedExperts format where all experts are in a single tensor.
        TorchTitan also uses GroupedExperts, so we directly convert keys without splitting/concatenating.
        
        Args:
            hf_state_dict: HuggingFace model state dict
            
        Returns:
            TorchTitan-formatted state dict
        """
        import torch

        # Separate vision, text, and grouped expert keys
        vision_dict = {}
        text_dict = {}
        grouped_dict = {}  # GroupedExperts keys - direct conversion, bypass parent
        
        # HF Qwen3-VL-MOE uses GroupedExperts format - convert directly to TorchTitan format
        for key, value in hf_state_dict.items():
            if key.startswith("model.visual."):
                # Vision encoder: strip model. prefix
                tt_key = key.replace("model.", "", 1)
                vision_dict[tt_key] = value
                
            elif key.startswith("model.language_model."):
                # Check if this is a GroupedExperts key - convert directly
                if ".mlp.experts.down_proj" in key:
                    # GroupedExperts down_proj [num_experts, intermediate, hidden] → w2 [num_experts, hidden, intermediate]
                    # Transpose last 2 dims: [128, 768, 2048] → [128, 2048, 768]
                    tt_key = key.replace("model.language_model.", "").replace(".mlp.experts.down_proj", ".moe.experts.w2")
                    grouped_dict[tt_key] = value.transpose(-2, -1)
                    
                elif ".mlp.experts.gate_up_proj" in key:
                    # GroupedExperts fused gate_up_proj [num_experts, hidden, 2*intermediate]
                    # Split into w1 and w3, transpose to [num_experts, intermediate, hidden]
                    w1, w3 = value.chunk(2, dim=-1)  # Split: [128, 2048, 1536] → 2x [128, 2048, 768]
                    # Transpose last 2 dims: [128, 2048, 768] → [128, 768, 2048]
                    base_key = key.replace("model.language_model.", "").replace(".mlp.experts.gate_up_proj", ".moe.experts")
                    grouped_dict[f"{base_key}.w1"] = w1.transpose(-2, -1)  # gate_proj → w1
                    grouped_dict[f"{base_key}.w3"] = w3.transpose(-2, -1)  # up_proj → w3
                    
                    # Add expert_bias (not in HF checkpoint, initialize with zeros)
                    # Shape: [num_experts] - one bias value per expert for load balancing
                    num_experts = w1.shape[0]
                    expert_bias_key = base_key.replace(".moe.experts", ".moe.expert_bias")
                    grouped_dict[expert_bias_key] = torch.zeros(num_experts, dtype=value.dtype, device=value.device)
                    
                elif ".mlp.gate.weight" in key:
                    # Router gate - direct conversion
                    # model.language_model.layers.N.mlp.gate.weight → layers.N.moe.router.gate.weight
                    tt_key = key.replace("model.language_model.", "").replace(".mlp.gate.weight", ".moe.router.gate.weight")
                    grouped_dict[tt_key] = value
                    
                else:
                    # Non-GroupedExperts text model keys: strip model.language_model. → model. for parent
                    parent_key = key.replace("model.language_model.", "model.", 1)
                    text_dict[parent_key] = value
            else:
                # Other keys (like lm_head.weight) pass through to parent
                text_dict[key] = value
        
        # Convert non-GroupedExperts text model using parent logic
        tt_text_dict = super().from_hf(text_dict)
        
        # Add language_model. prefix to all text model keys (both parent-converted and GroupedExperts)
        prefixed_text_dict = {}
        for key, value in tt_text_dict.items():
            prefixed_key = f"language_model.{key}"
            prefixed_text_dict[prefixed_key] = value
        
        # Add language_model. prefix to GroupedExperts keys
        for key, value in grouped_dict.items():
            prefixed_key = f"language_model.{key}"
            prefixed_text_dict[prefixed_key] = value
        
        # Merge vision and text dictionaries
        state_dict = {**vision_dict, **prefixed_text_dict}
        
        return state_dict


__all__ = ["Qwen3VLStateDictAdapter"]
