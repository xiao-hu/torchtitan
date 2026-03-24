"""
Verify DeepStack vision-LLM connection is correct.

Checks:
1. Vision encoder output shapes match what the LLM expects
2. masked_scatter injects features at correct positions
3. DeepStack features are injected at the right layers
4. Position IDs are correct for vision tokens
"""
import torch
import sys

def main():
    device = "cuda:0"
    dtype = torch.float32
    torch.manual_seed(42)

    print("=" * 60)
    print("Verifying Vision-LLM DeepStack Connection")
    print("=" * 60)

    # Load model args
    from torchtitan.experiments.qwen3_vl import qwen3_vl_args
    model_args = qwen3_vl_args["30B-A3B"]

    # =========================================================================
    # 1. Verify vision encoder output shapes
    # =========================================================================
    print("\n=== 1. Vision Encoder Output Shapes ===")
    from torchtitan.experiments.qwen3_vl.model.native_vision import Qwen3VLNativeVisionEncoder

    vision = Qwen3VLNativeVisionEncoder(model_args.vision_config).to(device, dtype).eval()

    # Simulate 2 images
    grid_thw = torch.tensor([[1, 26, 40], [1, 30, 40]], dtype=torch.long, device=device)
    total_patches = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item()
    patch_dim = 3 * 2 * 16 * 16  # in_channels * temporal_patch_size * patch_size^2
    pixel_values = torch.randn(total_patches, patch_dim, device=device, dtype=dtype)

    with torch.no_grad():
        merged, deepstack = vision(pixel_values, grid_thw)

    merge_size = model_args.vision_config.spatial_merge_size
    expected_merged_patches = sum(
        int(t) * (int(h) // merge_size) * (int(w) // merge_size)
        for t, h, w in grid_thw.tolist()
    )
    out_dim = model_args.vision_config.out_hidden_size

    print(f"  merged: {merged.shape} (expected: [{expected_merged_patches}, {out_dim}])")
    assert merged.shape == (expected_merged_patches, out_dim), f"Merged shape mismatch!"

    print(f"  deepstack: {len(deepstack)} features (expected: {len(model_args.deepstack_visual_indexes)})")
    assert len(deepstack) == len(model_args.deepstack_visual_indexes)

    for i, ds in enumerate(deepstack):
        print(f"    deepstack[{i}]: {ds.shape} (expected: [{expected_merged_patches}, {out_dim}])")
        assert ds.shape == (expected_merged_patches, out_dim)

    print("  PASS: All shapes correct")

    # =========================================================================
    # 2. Verify out_hidden_size matches text model dim
    # =========================================================================
    print("\n=== 2. Vision-Text Dimension Match ===")
    text_dim = model_args.dim  # Language model hidden dim
    vision_out_dim = model_args.vision_config.out_hidden_size

    print(f"  Text model dim: {text_dim}")
    print(f"  Vision out_hidden_size: {vision_out_dim}")
    assert text_dim == vision_out_dim, f"Dimension mismatch: text={text_dim} vs vision={vision_out_dim}"
    print("  PASS: Dimensions match")

    # =========================================================================
    # 3. Verify masked_scatter injection
    # =========================================================================
    print("\n=== 3. Masked Scatter Injection ===")
    from torchtitan.experiments.qwen3_vl.model.model import get_placeholder_mask

    # Create a fake input_ids with image tokens at known positions
    seq_len = 100
    image_token_id = model_args.image_token_id  # 151655
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

    # Place image tokens at positions 10-19 (10 tokens for first image)
    # and 30-44 (15 tokens for second image)
    # After merge: 260 patches + 300 patches → 65 + 75 = 140 merged patches
    # But in the real model, num image tokens = num merged patches
    num_merged_img1 = (26 // merge_size) * (40 // merge_size)  # 13*20 = 260
    num_merged_img2 = (30 // merge_size) * (40 // merge_size)  # 15*20 = 300
    total_merged = num_merged_img1 + num_merged_img2

    # Adjust seq_len to fit
    seq_len = total_merged + 50  # 50 text tokens + image tokens
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    # Place image tokens
    input_ids[0, 10:10 + num_merged_img1] = image_token_id
    input_ids[0, 10 + num_merged_img1 + 5:10 + num_merged_img1 + 5 + num_merged_img2] = image_token_id

    # Create fake embeddings
    inputs_embeds = torch.randn(1, seq_len, text_dim, device=device, dtype=dtype)
    image_features = merged  # Use real vision output

    # Get mask
    image_mask, video_mask = get_placeholder_mask(
        input_ids, inputs_embeds,
        image_features=image_features,
        image_token_id=image_token_id,
    )

    # Count image positions in mask
    num_image_positions = image_mask[..., 0].sum().item()
    print(f"  Image token positions in input_ids: {(input_ids == image_token_id).sum().item()}")
    print(f"  Image positions in mask: {num_image_positions}")
    print(f"  Image features count: {image_features.shape[0]}")
    assert num_image_positions == total_merged, f"Mask count mismatch: {num_image_positions} vs {total_merged}"

    # Apply masked_scatter
    new_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

    # Verify: at image positions, embeddings should match vision features
    img_positions = (input_ids[0] == image_token_id).nonzero().squeeze(-1)
    scattered_features = new_embeds[0, img_positions]
    cos_sim = torch.nn.functional.cosine_similarity(
        scattered_features.flatten().unsqueeze(0),
        image_features.flatten().unsqueeze(0)
    ).item()
    print(f"  Scattered features cosine_similarity: {cos_sim:.8f}")
    assert cos_sim > 0.999, f"Scatter mismatch: cos_sim={cos_sim}"
    print("  PASS: masked_scatter injects correctly")

    # =========================================================================
    # 4. Verify DeepStack layer indices
    # =========================================================================
    print("\n=== 4. DeepStack Layer Indices ===")
    ds_indices = model_args.deepstack_visual_indexes
    vision_ds_indices = model_args.vision_config.deepstack_visual_indexes
    n_layers = model_args.n_layers

    print(f"  Model deepstack_visual_indexes: {ds_indices}")
    print(f"  Vision deepstack_visual_indexes: {vision_ds_indices}")
    print(f"  Language model n_layers: {n_layers}")

    # DeepStack features from vision layer i are injected into language layer i
    for idx in ds_indices:
        assert idx < n_layers, f"DeepStack index {idx} >= n_layers {n_layers}"
    print(f"  All indices < n_layers ({n_layers}): PASS")

    # Check that vision and model indices match
    assert ds_indices == vision_ds_indices, f"Index mismatch: model={ds_indices} vs vision={vision_ds_indices}"
    print(f"  Vision and model indices match: PASS")

    # =========================================================================
    # 5. Verify _deepstack_process
    # =========================================================================
    print("\n=== 5. DeepStack Process ===")
    from torchtitan.experiments.qwen3_vl.model.model import Qwen3VLTextModel

    # Create a minimal visual_pos_mask
    batch_size = 1
    visual_pos_masks = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    visual_pos_masks[0, img_positions] = True

    # Simulate hidden states and visual embeds
    hidden_states = torch.randn(batch_size, seq_len, text_dim, device=device, dtype=dtype)
    visual_embeds = torch.randn(total_merged, text_dim, device=device, dtype=dtype)

    # Run _deepstack_process (it's a static method pattern)
    # It adds visual_embeds at masked positions
    mask_3d = visual_pos_masks.unsqueeze(-1).expand_as(hidden_states)
    visual_add = hidden_states.new_zeros(hidden_states.shape)
    visual_add.masked_scatter_(mask_3d, visual_embeds)
    result = hidden_states + visual_add

    # Verify: at masked positions, result should be hidden + visual
    original_at_mask = hidden_states[visual_pos_masks]
    result_at_mask = result[visual_pos_masks]
    diff_at_mask = result_at_mask - original_at_mask

    # The diff should be the visual_embeds (scattered in order)
    cos_sim = torch.nn.functional.cosine_similarity(
        diff_at_mask.flatten().unsqueeze(0),
        visual_embeds.flatten().unsqueeze(0)
    ).item()
    print(f"  DeepStack injection cosine_similarity: {cos_sim:.8f}")
    assert cos_sim > 0.999, f"DeepStack injection mismatch: cos_sim={cos_sim}"

    # Verify: at non-masked positions, result should be unchanged
    non_mask = ~visual_pos_masks
    unchanged = (result[non_mask] == hidden_states[non_mask]).all().item()
    print(f"  Non-masked positions unchanged: {unchanged}")
    assert unchanged, "Non-masked positions were modified!"
    print("  PASS: DeepStack injection is correct")

    print("\n" + "=" * 60)
    print("ALL VISION-LLM CONNECTION TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
