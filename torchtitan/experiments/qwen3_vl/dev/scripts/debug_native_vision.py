"""Debug: compare HF vs native vision encoder at each stage."""

import torch

device = "cuda:0"
dtype = torch.float32
torch.manual_seed(42)

# Load both models
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

# Test input
grid_thw = torch.tensor([[1, 26, 40], [1, 30, 40]], dtype=torch.long, device=device)  # Two images
total_patches = 26 * 40 + 30 * 40
patch_dim = 3 * 2 * 16 * 16
pixel_values = torch.randn(total_patches, patch_dim, device=device, dtype=dtype)


def compare(name, a, b):
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    diff = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()
    print(f"  {name}: max_diff={diff.max().item():.6f}, mean_diff={diff.mean().item():.6f}, cos_sim={cos:.6f}")


with torch.no_grad():
    # === Stage 1: Patch embedding ===
    print("=== Stage 1: Patch Embedding ===")
    hf_embed = hf_vision.patch_embed(pixel_values)
    native_embed = native_vision.patch_embed(pixel_values)
    compare("patch_embed", native_embed, hf_embed)

    # === Stage 2: Position embedding ===
    print("\n=== Stage 2: Position Embedding ===")
    hf_pos = hf_vision.fast_pos_embed_interpolate(grid_thw)
    native_pos = native_vision._compute_position_embeddings(grid_thw)
    print(f"  HF pos shape: {hf_pos.shape}, Native pos shape: {native_pos.shape}")
    if hf_pos.shape == native_pos.shape:
        compare("pos_embed", native_pos, hf_pos)
    else:
        print(f"  Shape mismatch! HF: {hf_pos.shape}, Native: {native_pos.shape}")
        # Compare first few values
        min_len = min(hf_pos.shape[0], native_pos.shape[0])
        compare("pos_embed (truncated)", native_pos[:min_len], hf_pos[:min_len])

    hf_h = hf_embed + hf_pos
    native_h = native_embed + native_pos if hf_pos.shape == native_pos.shape else native_embed + native_pos

    # === Stage 3: Rotary position embedding ===
    print("\n=== Stage 3: Rotary Position Embedding ===")
    hf_rot = hf_vision.rot_pos_emb(grid_thw)
    native_cos, native_sin = native_vision._compute_rotary_pos_emb(grid_thw)
    # HF returns raw freqs, then does cos/sin in forward
    hf_emb = torch.cat((hf_rot, hf_rot), dim=-1)
    hf_cos, hf_sin = hf_emb.cos(), hf_emb.sin()
    print(f"  HF rot shape: {hf_rot.shape}")
    print(f"  Native cos shape: {native_cos.shape}")
    compare("rotary_cos", native_cos, hf_cos)
    compare("rotary_sin", native_sin, hf_sin)

    # === Stage 4: First block ===
    print("\n=== Stage 4: First Transformer Block ===")
    # HF: needs cu_seqlens and position_embeddings
    import torch.nn.functional as F
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    cu_seqlens = F.pad(cu_seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
    hf_position_embeddings = (hf_cos, hf_sin)

    hf_block0_out = hf_vision.blocks[0](hf_h, cu_seqlens=cu_seqlens, position_embeddings=hf_position_embeddings)

    # Native: needs block_mask (multi-image aware)
    from torch.nn.attention.flex_attention import create_block_mask
    seq_len = native_h.shape[0]
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_images = seq_lens.shape[0]
    image_ids = torch.repeat_interleave(
        torch.arange(num_images, device=device, dtype=torch.int32), seq_lens
    )
    def mask_mod(b, h, q_idx, kv_idx):
        return image_ids[q_idx] == image_ids[kv_idx]
    block_mask = create_block_mask(mask_mod, B=1, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)
    native_position_embeddings = (native_cos, native_sin)

    native_block0_out = native_vision.blocks[0](native_h, block_mask=block_mask, position_embeddings=native_position_embeddings)
    compare("block_0_output", native_block0_out, hf_block0_out)

    # === Attention only ===
    print("\n=== Stage 4b: Attention Only (Block 0) ===")
    hf_norm1 = hf_vision.blocks[0].norm1(hf_h)
    native_norm1 = native_vision.blocks[0].norm1(native_h)
    compare("norm1", native_norm1, hf_norm1)

    # QKV
    hf_qkv = hf_vision.blocks[0].attn.qkv(hf_norm1)
    native_qkv = native_vision.blocks[0].attn.qkv(native_norm1)
    compare("qkv", native_qkv, hf_qkv)

    print("\nDone.")
