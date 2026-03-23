# Native Vision Encoder Implementation Plan

**Goal**: Replace HF `Qwen3VLVisionModel` wrapper with a TorchTitan-native implementation that supports `fullgraph=True` torch.compile, eliminating ~1.7s/step CPU overhead.

**Target**: 25-35% MFU (currently 15.5% with ep=2)

---

## Architecture Overview

Qwen3-VL uses a SigLIP-2 based vision encoder with these key components:

```
Input: pixel_values [total_patches, C*temporal_patch_size*patch_size*patch_size]
       grid_thw [num_images, 3]  (temporal, height, width per image)

Pipeline:
1. PatchEmbed: Linear projection → [total_patches, hidden_size]
2. Position: Learned embedding + 2D Rotary (cos/sin for H, W)
3. 27× TransformerBlock:
   - LayerNorm → Attention (with 2D RoPE, per-image masking) → Residual
   - LayerNorm → MLP (SiLU activation) → Residual
   - DeepStack: extract features at layers [8, 16, 24] via merger MLPs
4. PatchMerger: spatial_merge_size×spatial_merge_size → 1 token (LayerNorm + MLP)
5. Output: merged_features [total_merged_patches, out_hidden_size]
         + deepstack_features [3 × [total_merged_patches, out_hidden_size]]
```

### Parameters (30B-A3B config)

| Component | Params | Details |
|-----------|--------|---------|
| PatchEmbed | 1.8M | Linear(C×2×16×16=1536, 1152) |
| Position Embed | 2.7M | Embedding(2304, 1152) |
| Rotary | 0 | Computed, not learned |
| 27× Blocks | 411.5M | Attn(1152, 16 heads) + MLP(1152→4304→1152) |
| PatchMerger | 30.7M | Linear(1152×4, 4304) → SiLU → Linear(4304, 2048) |
| 3× DeepStack Mergers | 92.1M | Same structure as PatchMerger |
| **Total** | **538.8M** | |

---

## Detailed Component Design

### 1. PatchEmbed

HF implementation:
```python
# Reshapes flat input to 3D patches, applies Conv3d, flattens back
hidden_states = hidden_states.view(-1, C, temporal_patch_size, patch_size, patch_size)
hidden_states = self.proj(hidden_states).view(-1, embed_dim)  # proj is Conv3d
```

Native implementation:
```python
# Conv3d is equivalent to a Linear on the flattened patch
# Input is already flattened: [total_patches, C * temporal_patch_size * patch_size * patch_size]
hidden_states = self.proj(hidden_states)  # Linear(1536, 1152)
```

**Note**: The HF checkpoint stores this as a Conv3d weight. The state_dict_adapter already handles the conversion (reshape Conv3d → Linear).

### 2. Position Embedding

Two components:
- **Learned positional embedding**: `Embedding(num_position_embeddings, hidden_size)` = `Embedding(2304, 1152)`
- **2D Rotary position embedding**: Computed from grid_thw, applied per-head in attention

HF computes positions via `fast_pos_embed_interpolate` which:
1. Gets the grid H×W per image from grid_thw
2. Computes position indices for each patch
3. Looks up learned embeddings
4. Adds to patch embeddings

Native approach: Pre-compute position indices from grid_thw (can be done on CPU), then use `F.embedding` lookup. This is compile-friendly.

### 3. 2D Rotary Position Embedding

HF `rot_pos_emb(grid_thw)`:
```python
# For each image, compute h_index and w_index for each patch
# Apply rotary embedding: emb = cat(rotary, rotary) → (cos, sin)
```

Native approach:
- Compute rotary frequencies from grid dimensions
- Apply as (cos, sin) tuple to Q, K in attention
- Key: avoid data-dependent shapes. Use `torch.repeat_interleave` with tensor counts (compile-friendly in recent PyTorch)

### 4. Attention

HF uses cu_seqlens-based attention (flash attention or manual chunking):
```python
# Flash path: pass cu_seqlens to flash_attn
# Non-flash path: split Q,K,V by cu_seqlens, attend per-chunk, cat results
```

**This is the main compile-breaking pattern.** The non-flash path uses `.tolist()` and Python loops over variable-length chunks.

Native options (in order of preference):
1. **SDPA with per-image block mask**: Pad all images to same length, use `F.scaled_dot_product_attention` with an attention mask. Compile-friendly but wastes compute on padding.
2. **FlexAttention with BlockMask**: Like the VLM `siglip2.py` does. Most efficient and compile-friendly. Requires creating a `BlockMask` from `cu_seqlens`.
3. **Flash attention with cu_seqlens**: Would work but `flash_attn` is not always compile-friendly.

**Recommendation**: Use **FlexAttention** (option 2) following the VLM pattern. This is the most TorchTitan-native approach and fully supports `fullgraph=True`.

### 5. MLP

HF:
```python
# SiLU activation (not SwiGLU — single gate, not split gate)
self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))
# act_fn = nn.SiLU() for "gelu_pytorch_tanh" config (misleading name in config)
```

Wait — need to verify the actual activation. The config says `hidden_act="gelu_pytorch_tanh"` but let me double-check:

```python
# HF Qwen3VLVisionMLP.__init__:
self.act_fn = ACT2FN[config.hidden_act]
# "gelu_pytorch_tanh" → nn.GELU(approximate="tanh")
```

So it's actually `GELU(tanh)`, same as VLM siglip2.py. Simple to implement.

### 6. TransformerBlock

```python
# Pre-norm residual
x = x + attn(norm1(x), ...)
x = x + mlp(norm2(x))
```

Standard pre-norm transformer. No complications.

### 7. PatchMerger (Spatial Merge)

Merges `spatial_merge_size × spatial_merge_size` (2×2) adjacent patches into 1 token:

```python
# Reshape: group adjacent patches
# x shape: [total_patches, hidden_size]
# After grouping: [total_patches // 4, hidden_size * 4]
# MLP: Linear(hidden_size * 4, intermediate) → SiLU → Linear(intermediate, out_hidden_size)
```

HF uses a specific reshape pattern based on grid dimensions. Need to handle variable grid sizes.

Native approach: Can use `einops.rearrange` or manual reshape. The key is that this happens **after** the transformer blocks, so the grid structure is known from `grid_thw`.

### 8. DeepStack Feature Extraction

At layers [8, 16, 24], extract hidden states and pass through a merger MLP (same architecture as PatchMerger):

```python
if layer_num in deepstack_visual_indexes:
    deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
    deepstack_feature_lists.append(deepstack_feature)
```

Simple — just need the merger MLPs and the layer indices.

---

## Data Layout Decision

**HF layout**: Flat `(total_seq, D)` — all images concatenated into one sequence, separated by `cu_seqlens`. Efficient for flash attention but requires `cu_seqlens` tracking.

**VLM layout**: Batched `(B, L, D)` — each image padded to max length. Works with standard attention but wastes compute on padding.

**Recommendation**: Use **batched layout `(B, L, D)`** for the native encoder:
- Compatible with FlexAttention and SDPA
- No data-dependent shapes (fixed B, L dimensions per batch)
- The padding overhead is acceptable (vision sequences are short, 200-2000 patches)
- Much more compile-friendly than cu_seqlens

This requires reshaping the flat `pixel_values` input into batched format using `grid_thw`, and reshaping back to flat after the encoder.

---

## State Dict Compatibility

The native encoder must load weights from the HF checkpoint. Key mappings:

| HF Weight | Native Weight | Notes |
|-----------|--------------|-------|
| `visual.patch_embed.proj.weight` (Conv3d) | `visual.patch_embed.proj.weight` (Linear) | Reshape 5D→2D |
| `visual.pos_embed.weight` | `visual.pos_embed.weight` | Direct |
| `visual.blocks.{i}.norm1.weight/bias` | `visual.blocks.{i}.norm1.weight/bias` | Direct |
| `visual.blocks.{i}.attn.qkv.weight/bias` | `visual.blocks.{i}.attn.qkv.weight/bias` | Direct (fused QKV) |
| `visual.blocks.{i}.attn.proj.weight/bias` | `visual.blocks.{i}.attn.proj.weight/bias` | Direct |
| `visual.blocks.{i}.mlp.linear_fc1.*` | `visual.blocks.{i}.mlp.fc1.*` | Rename |
| `visual.blocks.{i}.mlp.linear_fc2.*` | `visual.blocks.{i}.mlp.fc2.*` | Rename |
| `visual.merger.*` | `visual.merger.*` | Direct |
| `visual.deepstack_merger_list.{i}.*` | `visual.deepstack_merger_list.{i}.*` | Direct |

Most weights map directly — the main conversion is Conv3d→Linear for patch embedding.

---

## Implementation Steps

### Phase 1: Core Components (est. ~200 lines)

1. **`Qwen3VLNativeVisionArgs`** — Config dataclass (reuse existing `Qwen3VLVisionArgs`)
2. **`VisionPatchEmbed`** — Linear projection (trivial)
3. **`VisionRotaryEmbedding`** — 2D RoPE computation from grid_thw
4. **`VisionAttention`** — QKV projection + FlexAttention/SDPA + output projection
5. **`VisionMLP`** — FC1 → GELU(tanh) → FC2
6. **`VisionBlock`** — Pre-norm residual (norm → attn → residual, norm → mlp → residual)
7. **`VisionPatchMerger`** — Spatial merge (reshape + LayerNorm + MLP)
8. **`Qwen3VLNativeVisionEncoder`** — Full encoder (embed → blocks → merge → deepstack)

### Phase 2: Integration (~50 lines)

9. Update `model.py` to use native encoder instead of HF wrapper
10. Update `state_dict_adapter.py` for any weight name differences
11. Update `parallelize.py` — FSDP wrapping should be the same

### Phase 3: Verification (~100 lines)

12. Numerical equivalence test: load HF checkpoint, compare outputs
13. Training test: run 20 steps, verify loss trajectory matches
14. Compile test: verify `fullgraph=True` works
15. Profile: compare MFU with/without compile

### Phase 4: Compile Optimization

16. Enable `fullgraph=True` compile on native encoder
17. Test with `dynamic=True` or `mark_dynamic()` for variable image sizes
18. Profile and compare with baseline

---

## Key Compile Patterns to Avoid

1. **No `torch._dynamo.decorators.disable`** — don't use HF utilities
2. **No `.tolist()` or `.item()`** — use tensor ops for all shapes
3. **No Python loops over data-dependent lengths** — use batched ops or FlexAttention
4. **No `torch.split` with dynamic sizes** — use padding + masking instead
5. **No `F.interpolate` in forward** — pre-compute position embeddings if possible

## Open Questions

1. **FlexAttention vs SDPA**: FlexAttention is more efficient (no padding waste) but adds complexity. SDPA with padding is simpler. Start with SDPA, optimize to FlexAttention if needed.
2. **Batched vs flat layout**: Batched is more compile-friendly but adds padding. For VQAv2 with variable image sizes, padding overhead should be small.
3. **Position embedding interpolation**: The HF model uses `fast_pos_embed_interpolate` which does bilinear interpolation per image. This has a Python loop. Options: (a) batch all same-size images, (b) use `F.grid_sample` for batched interpolation, (c) skip interpolation if all images use the default grid size.
