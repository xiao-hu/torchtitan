"""Debug: compare full encoder outputs at each stage for multi-image."""
import torch
import torch.nn.functional as F

device = "cuda:0"
dtype = torch.float32
torch.manual_seed(42)

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers import Qwen3VLConfig
from torchtitan.experiments.qwen3_vl.model.native_vision import Qwen3VLNativeVisionEncoder
from torchtitan.experiments.qwen3_vl.model.args import Qwen3VLVisionArgs

hf_config = Qwen3VLConfig.from_pretrained("/checkpoints/xxie-sandbox/Qwen/Qwen3-VL-30B-A3B-Instruct")
hf_vision = Qwen3VLVisionModel(hf_config.vision_config).to(device, dtype).eval()

native_args = Qwen3VLVisionArgs(
    depth=27, hidden_size=1152, intermediate_size=4304, num_heads=16,
    in_channels=3, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
    out_hidden_size=2048, num_position_embeddings=2304,
    deepstack_visual_indexes=[8, 16, 24],
)
native_vision = Qwen3VLNativeVisionEncoder(native_args).to(device, dtype).eval()

# Copy weights
hf_sd = hf_vision.state_dict()
mapped = {}
for hf_key, hf_val in hf_sd.items():
    native_key = hf_key.replace(".linear_fc1.", ".fc1.").replace(".linear_fc2.", ".fc2.")
    if native_key == "patch_embed.proj.weight":
        hf_val = hf_val.reshape(hf_val.shape[0], -1)
        native_key = "patch_embed.weight"
    elif native_key == "patch_embed.proj.bias":
        native_key = "patch_embed.bias"
    if native_key in native_vision.state_dict():
        mapped[native_key] = hf_val
native_vision.load_state_dict(mapped, strict=False)

# 2 images
grid_thw = torch.tensor([[1, 26, 40], [1, 30, 40]], dtype=torch.long, device=device)
total_patches = 26 * 40 + 30 * 40
patch_dim = 3 * 2 * 16 * 16
pixel_values = torch.randn(total_patches, patch_dim, device=device, dtype=dtype)


def compare(name, a, b):
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    diff = (a - b).abs()
    cos = F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
    print(f"  {name}: max={diff.max():.6f} mean={diff.mean():.6f} cos={cos:.6f}")


with torch.no_grad():
    # Run HF manually step by step
    hf_h = hf_vision.patch_embed(pixel_values)
    hf_pos = hf_vision.fast_pos_embed_interpolate(grid_thw)
    hf_h = hf_h + hf_pos
    hf_rot = hf_vision.rot_pos_emb(grid_thw)
    hf_emb = torch.cat((hf_rot, hf_rot), dim=-1)
    hf_pos_emb = (hf_emb.cos(), hf_emb.sin())
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = F.pad(cu_seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)

    # Run native manually step by step
    native_h = native_vision.patch_embed(pixel_values)
    native_pos = native_vision._compute_position_embeddings(grid_thw)
    native_h = native_h + native_pos
    native_cos, native_sin = native_vision._compute_rotary_pos_emb(grid_thw)
    native_pos_emb = (native_cos, native_sin)
    block_mask = native_vision._create_block_mask(grid_thw, native_h.shape[0], device)

    print("=== Layer-by-layer comparison ===")
    for i in range(27):
        hf_h = hf_vision.blocks[i](hf_h, cu_seqlens=cu_seqlens, position_embeddings=hf_pos_emb)
        native_h = native_vision.blocks[i](native_h, block_mask=block_mask, position_embeddings=native_pos_emb)
        cos = F.cosine_similarity(hf_h.flatten().unsqueeze(0), native_h.flatten().unsqueeze(0)).item()
        diff = (hf_h - native_h).abs()
        if i < 5 or i >= 24 or cos < 0.999:
            print(f"  block {i:2d}: max={diff.max():.6f} mean={diff.mean():.6f} cos={cos:.8f}")

    print(f"\n=== After all blocks ===")
    compare("final_hidden", native_h, hf_h)

    # Spatial merge comparison
    print(f"\n=== Spatial Merge ===")
    hf_merged = hf_vision.merger(hf_h)
    native_merged = native_vision._spatial_merge(native_h, grid_thw, native_vision.merger)
    compare("merged", native_merged, hf_merged)

    # Check: what does HF merger actually receive?
    # HF merger.forward just does norm + MLP on the input
    # But the reshape happens differently...
    print(f"\n  HF merger input shape: {hf_h.shape}")
    print(f"  HF merged output shape: {hf_merged.shape}")
    print(f"  Native merged output shape: {native_merged.shape}")

    # The HF merger receives the raw hidden states and does its own reshape
    # Let's check if the spatial merge reshape is different
    print(f"\n=== Debug: Spatial merge reshape ===")
    merge_size = 2
    # HF's merger forward does: x = self.norm(x.view(-1, hidden_size)).view(-1, merge_dim) then MLP
    # But first HF needs to reshape hidden states for merging...
    # Actually HF's merger is called in the main forward, let me check what HF passes to it

    # In HF Qwen3VLVisionModel.forward:
    #   merged_hidden_states = self.merger(hidden_states)
    # So HF passes the raw flat (2240, 1152) tensor to merger
    # The merger's forward does:
    #   x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x)
    #   .view(-1, self.hidden_size)
    #   x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
    # Where self.hidden_size = 1152 * 4 = 4608

    # So HF merger just views (2240, 1152) as (560, 4608)
    # Our _spatial_merge does per-image permutation then passes to merger

    # The key question: does HF's view(-1, 4608) produce the same grouping as our permutation?
    hf_reshaped = hf_h.view(-1, 1152 * 4)  # HF's simple reshape
    print(f"  HF simple reshape: {hf_h.shape} -> {hf_reshaped.shape}")

    # Our permutation groups by (h//m, w//m, m, m) order
    # HF's simple reshape groups by consecutive patches
    # These are DIFFERENT orderings unless the patches are already in merge order

    # Check: are patches already in merge-permuted order from position embedding?
    # In HF's fast_pos_embed_interpolate, there's a permutation step...
    # Let's verify by checking if hf_reshaped matches our permuted version
    native_permuted = native_vision._spatial_merge.__wrapped__(native_vision, native_h, grid_thw, None)
    # Can't call with None merger... let me just do the permute manually
    grid_thw_list = grid_thw.tolist()
    sizes = [int(t) * int(h) * int(w) for t, h, w in grid_thw_list]
    splits = torch.split(native_h, sizes)
    permuted_chunks = []
    for chunk, (t, h, w) in zip(splits, grid_thw_list):
        t, h, w = int(t), int(h), int(w)
        chunk_p = (
            chunk.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(t * (h // merge_size) * (w // merge_size), -1)
        )
        permuted_chunks.append(chunk_p)
    native_permuted = torch.cat(permuted_chunks, dim=0)

    print(f"  HF reshaped: {hf_reshaped.shape}")
    print(f"  Native permuted: {native_permuted.shape}")
    compare("reshape_vs_permute", native_permuted, hf_reshaped)
