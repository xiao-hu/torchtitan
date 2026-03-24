"""
Verify numerical equivalence between HF Qwen3VLVisionModel and native vision encoder.

Loads the same checkpoint into both, runs the same input, compares outputs.
Run on a single GPU (no distributed).
"""

import torch
import sys

def main():
    device = "cuda:0"
    dtype = torch.float32  # Use fp32 for comparison precision

    # =========================================================================
    # 1. Load HF vision encoder
    # =========================================================================
    print("Loading HF Qwen3VLVisionModel...")
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel, Qwen3VLVisionConfig,
    )
    from transformers import Qwen3VLConfig

    hf_config = Qwen3VLConfig.from_pretrained(
        "/checkpoints/xxie-sandbox/Qwen/Qwen3-VL-30B-A3B-Instruct"
    )
    hf_vision = Qwen3VLVisionModel(hf_config.vision_config).to(device, dtype)
    hf_vision.eval()
    print(f"  HF params: {sum(p.numel() for p in hf_vision.parameters()):,}")

    # =========================================================================
    # 2. Load native vision encoder with same weights
    # =========================================================================
    print("Loading native Qwen3VLNativeVisionEncoder...")
    from torchtitan.experiments.qwen3_vl.model.native_vision import (
        Qwen3VLNativeVisionEncoder,
    )
    from torchtitan.experiments.qwen3_vl.model.args import Qwen3VLVisionArgs

    native_args = Qwen3VLVisionArgs(
        depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16,
        in_channels=3, patch_size=16, spatial_merge_size=2,
        temporal_patch_size=2, out_hidden_size=2048,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
    )
    native_vision = Qwen3VLNativeVisionEncoder(native_args).to(device, dtype)
    native_vision.eval()
    print(f"  Native params: {sum(p.numel() for p in native_vision.parameters()):,}")

    # =========================================================================
    # 3. Copy weights from HF to native
    # =========================================================================
    print("Copying weights HF → native...")
    hf_sd = hf_vision.state_dict()
    native_sd = native_vision.state_dict()

    # Build mapping: HF key → native key
    mapped = {}
    unmapped_hf = []
    for hf_key, hf_val in hf_sd.items():
        # Rename MLP layers
        native_key = hf_key.replace(".linear_fc1.", ".fc1.").replace(".linear_fc2.", ".fc2.")

        # Conv3d patch embed → Linear
        if native_key == "patch_embed.proj.weight":
            hf_val = hf_val.reshape(hf_val.shape[0], -1)
            native_key = "patch_embed.weight"
        elif native_key == "patch_embed.proj.bias":
            native_key = "patch_embed.bias"

        if native_key in native_sd:
            if hf_val.shape == native_sd[native_key].shape:
                mapped[native_key] = hf_val
            else:
                print(f"  Shape mismatch: {hf_key} {hf_val.shape} vs {native_key} {native_sd[native_key].shape}")
                unmapped_hf.append(hf_key)
        else:
            # Skip non-parameter keys (like rotary inv_freq buffer)
            if "inv_freq" not in hf_key:
                unmapped_hf.append(hf_key)

    # Check for unmapped native keys
    unmapped_native = [k for k in native_sd if k not in mapped and "inv_freq" not in k]

    if unmapped_hf:
        print(f"  Unmapped HF keys ({len(unmapped_hf)}): {unmapped_hf[:5]}...")
    if unmapped_native:
        print(f"  Unmapped native keys ({len(unmapped_native)}): {unmapped_native[:5]}...")

    # Load mapped weights
    native_vision.load_state_dict(mapped, strict=False)
    print(f"  Mapped {len(mapped)}/{len(native_sd)} native params")

    # =========================================================================
    # 4. Create test input
    # =========================================================================
    print("Creating test input...")
    torch.manual_seed(42)

    # Simulate 2 images: one 1x26x40 and one 1x30x40 (common VQAv2 sizes)
    grid_thw = torch.tensor([[1, 26, 40], [1, 30, 40]], dtype=torch.long, device=device)
    total_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())

    # Random pixel values (already patchified)
    patch_dim = 3 * 2 * 16 * 16  # in_channels * temporal_patch_size * patch_size * patch_size
    pixel_values = torch.randn(total_patches, patch_dim, device=device, dtype=dtype)

    print(f"  grid_thw: {grid_thw.tolist()}")
    print(f"  total_patches: {total_patches}")
    print(f"  pixel_values: {pixel_values.shape}")

    # =========================================================================
    # 5. Run both encoders
    # =========================================================================
    print("\nRunning HF encoder...")
    with torch.no_grad():
        hf_output = hf_vision(pixel_values, grid_thw)
        hf_merged = hf_output.pooler_output
        hf_deepstack = hf_output.deepstack_features
    print(f"  merged: {hf_merged.shape}")
    print(f"  deepstack: {[ds.shape for ds in hf_deepstack]}")

    print("Running native encoder...")
    with torch.no_grad():
        native_merged, native_deepstack = native_vision(pixel_values, grid_thw)
    print(f"  merged: {native_merged.shape}")
    print(f"  deepstack: {[ds.shape for ds in native_deepstack]}")

    # =========================================================================
    # 6. Compare outputs
    # =========================================================================
    print("\n=== COMPARISON ===")

    def compare(name, a, b):
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            return False
        abs_diff = (a - b).abs()
        rel_diff = abs_diff / (b.abs().clamp(min=1e-8))
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        max_rel = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()

        status = "PASS" if cos_sim > 0.999 and max_abs < 0.01 else "FAIL"
        print(f"  {name}: {status}")
        print(f"    shape: {a.shape}")
        print(f"    max_abs_diff: {max_abs:.6f}")
        print(f"    mean_abs_diff: {mean_abs:.6f}")
        print(f"    max_rel_diff: {max_rel:.6f}")
        print(f"    mean_rel_diff: {mean_rel:.6f}")
        print(f"    cosine_similarity: {cos_sim:.8f}")
        return status == "PASS"

    all_pass = True
    all_pass &= compare("merged_features", native_merged, hf_merged)

    for i, (nd, hd) in enumerate(zip(native_deepstack, hf_deepstack)):
        all_pass &= compare(f"deepstack[{i}]", nd, hd)

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
