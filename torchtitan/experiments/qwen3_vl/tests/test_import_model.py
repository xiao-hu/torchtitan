# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test suite for Qwen3 VL model imports and configuration.
"""

import pytest
import torch


def _has_transformers():
    """Check if transformers library is available."""
    try:
        import transformers
        return True
    except ImportError:
        return False


class TestQwen3VLImports:
    """Test that Qwen3 VL model can be imported and configured."""

    def test_imports(self):
        """Test that all main components can be imported."""
        from torchtitan.experiments.qwen3_vl import (
            Qwen3VLModel,
            Qwen3VLModelArgs,
            Qwen3VLTextModel,
            Qwen3VLVisionArgs,
            SpecialTokens,
            qwen3_vl_args,
        )

        # Verify all exports are accessible
        assert Qwen3VLModel is not None
        assert Qwen3VLModelArgs is not None
        assert Qwen3VLTextModel is not None
        assert Qwen3VLVisionArgs is not None
        assert SpecialTokens is not None
        assert qwen3_vl_args is not None

    def test_model_configs_available(self):
        """Test that model configurations are properly defined."""
        from torchtitan.experiments.qwen3_vl import qwen3_vl_args

        # Check expected configurations exist
        assert "30B-A3B" in qwen3_vl_args
        assert "debugmodel" in qwen3_vl_args

    @pytest.mark.skipif(
        not _has_transformers(),
        reason="transformers library not available"
    )
    def test_model_instantiation_with_transformers(self):
        """Test model instantiation when transformers is available."""
        from torchtitan.experiments.qwen3_vl import Qwen3VLModel, qwen3_vl_args

        model_args = qwen3_vl_args["debugmodel"]  # Use small model for testing
        
        try:
            model = Qwen3VLModel(model_args)
            assert model is not None
            
            # Verify model components exist
            assert hasattr(model, "visual")
            assert hasattr(model, "language_model")
            
            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            assert num_params > 0
            
        except Exception as e:
            pytest.fail(f"Model instantiation failed: {e}")


if __name__ == "__main__":
    # Allow running as script for quick testing
    pytest.main([__file__, "-v"])
