# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for Qwen3 VL state dict adapter.
"""

import pytest
import torch

from torchtitan.experiments.qwen3_vl import (Qwen3VLStateDictAdapter,
                                             qwen3_vl_args)


class TestQwen3VLStateDictAdapter:
    """Test state dict conversion between HF and TorchTitan formats."""

    def test_adapter_initialization(self):
        """Test that adapter can be initialized."""
        model_args = qwen3_vl_args["debugmodel"]
        adapter = Qwen3VLStateDictAdapter(model_args, hf_assets_path=None)
        
        assert adapter is not None
        assert hasattr(adapter, "from_hf")
        assert hasattr(adapter, "to_hf")

    def test_vision_encoder_key_mapping_to_hf(self):
        """Test vision encoder keys are mapped correctly from TorchTitan to HF."""
        model_args = qwen3_vl_args["debugmodel"]
        adapter = Qwen3VLStateDictAdapter(model_args, hf_assets_path=None)
        
        # Create dummy TorchTitan state dict with vision keys
        tt_state_dict = {
            "visual.patch_embed.proj.weight": torch.randn(768, 3, 14, 14),
            "visual.patch_embed.proj.bias": torch.randn(768),
        }
        
        # Convert to HF format
        hf_state_dict = adapter.to_hf(tt_state_dict)
        
        # Vision keys should have model. prefix added
        assert "model.visual.patch_embed.proj.weight" in hf_state_dict
        assert "model.visual.patch_embed.proj.bias" in hf_state_dict
        assert "visual.patch_embed.proj.weight" not in hf_state_dict

    def test_text_model_key_mapping_from_hf(self):
        """Test text model keys are mapped correctly from HF to TorchTitan."""
        model_args = qwen3_vl_args["debugmodel"]
        adapter = Qwen3VLStateDictAdapter(model_args, hf_assets_path=None)
        
        # Create dummy HF state dict with text model keys
        # NOTE: Real HF Qwen3-VL uses model.language_model.* prefix
        hf_state_dict = {
            "model.language_model.embed_tokens.weight": torch.randn(151936, 256),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
            "model.language_model.norm.weight": torch.randn(256),
            "lm_head.weight": torch.randn(151936, 256),
        }
        
        # Convert to TorchTitan format
        tt_state_dict = adapter.from_hf(hf_state_dict)
        
        # Text model keys should have language_model. prefix
        assert "language_model.tok_embeddings.weight" in tt_state_dict
        assert "language_model.layers.0.attention.wq.weight" in tt_state_dict
        assert "language_model.norm.weight" in tt_state_dict
        assert "language_model.output.weight" in tt_state_dict

    def test_combined_vision_and_text_from_hf(self):
        """Test conversion of combined vision and text model from HF."""
        model_args = qwen3_vl_args["debugmodel"]
        adapter = Qwen3VLStateDictAdapter(model_args, hf_assets_path=None)
        
        # Create dummy HF state dict with both vision and text keys
        # NOTE: Real HF Qwen3-VL uses model.language_model.* prefix for text
        hf_state_dict = {
            # Vision keys
            "model.visual.patch_embed.proj.weight": torch.randn(768, 3, 14, 14),
            # Text keys (with correct model.language_model.* prefix)
            "model.language_model.embed_tokens.weight": torch.randn(151936, 256),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(256, 256),
            "lm_head.weight": torch.randn(151936, 256),
        }
        
        # Convert to TorchTitan format
        tt_state_dict = adapter.from_hf(hf_state_dict)
        
        # Check both vision and text keys are present
        assert "visual.patch_embed.proj.weight" in tt_state_dict
        assert "language_model.tok_embeddings.weight" in tt_state_dict
        assert "language_model.layers.0.attention.wq.weight" in tt_state_dict
        assert "language_model.output.weight" in tt_state_dict



if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])
