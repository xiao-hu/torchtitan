# Qwen3-VL MFU Improvement Plan

**Date**: 2026-03-22
**Hardware**: 8x H200 (143GB), VQAv2 dataset
**Current MFU**: 2-14% (variable), ~13.5% steady-state
**Target MFU**: 25-30%
**Reference**: Text-only (C4) achieves 33% MFU on same hardware

---

## Profiling Summary

Profiled iteration 20 (steady-state), rank 0, 2 steps (~5s total).

### GPU Utilization: ~24%

The GPU is idle 76% of the time. The compute is starved.

### GPU Kernel Time Breakdown (2.43s across 2 steps)

| Category | Time | % of GPU | Calls |
|----------|------|----------|-------|
| NCCL communication | 0.92s | 37.7% | 783 |
| GEMM/MatMul | 0.48s | 19.8% | 2512 |
| Attention (flash/sdpa) | 0.31s | 12.9% | 2992 |
| Elementwise/Copy | 0.30s | 12.5% | 6672 |
| Index/Gather/Scatter | 0.24s | 9.9% | 1763 |
| Other | 0.15s | 6.1% | 4788 |
| MOE chunk_cat | 0.03s | 1.1% | 131 |

### Key Bottlenecks

1. **Vision encoder BMM backward: 1.005s** — The uncompiled vision encoder's attention backward pass (`BmmBackward0`) is the single most expensive CPU op. This is because we disabled `torch.compile` on the vision encoder (HF transformers 5.x uses `torch._dynamo.decorators.disable` which breaks `fullgraph=True`).

2. **`item()` / CPU-GPU sync: 0.31s (18 calls)** — Two calls take 88ms each. These force the GPU pipeline to stall while the CPU reads scalar values. Likely from `get_rope_index()` or `get_placeholder_mask()` which read tensor values to compute shapes.

3. **`aten::nonzero`: 0.15s (20 calls)** — Data-dependent shape ops that prevent kernel fusion and cause graph breaks.

4. **`aten::index_put_`: 0.19s (487 calls)** — Scatter operations for injecting vision embeddings into text token positions. Many small kernel launches.

5. **NCCL dominates GPU time (38%)** — `all_to_all` for MOE expert dispatch is 0.57s. `all_gather` for FSDP is 0.26s.

6. **No recompilation in steady state** — The MFU variance across steps (0.3% to 14%) is from variable sequence/image sizes, not from torch.compile recompilation. Steps with many/large images have longer vision processing and lower MFU.

### Other Observations

- 11 unique compiled graphs (language model only, vision is uncompiled)
- Data loading: 55-147ms per step (not overlapped with compute)
- Profiler captures 307K events per step

---

## Improvement Plan

### Tier 1: Quick Wins (est. +2-5% MFU)

#### 1.1 Eliminate `item()` calls in forward path — DONE

**Problem**: `.tolist()` and `.sum()`/`.numel()` calls in forward path cause CPU-GPU sync.

**Changes made (2026-03-22)**:
1. Removed `.tolist()` + `torch.split` → `torch.cat` no-op in `get_image_features()` — vision encoder already returns flat embeddings, the split/cat was unnecessary
2. Removed `.sum()` and `.numel()` validation sync in `get_placeholder_mask()` — these were only for error messages

**Result**: +54% TPS on warmup (197→304), +2-5% TPS steady-state (3,262→3,339 at step 20, 3,380→3,547 at step 18). Peak MFU: 14.64% (up from 14.23%). Memory unchanged.

#### 1.2 Replace `aten::nonzero` with mask-based indexing — DONE, MARGINAL

**Problem**: `nonzero` produces dynamic output shapes, preventing fusion. 20 calls costing 0.15s.

**Changes made (2026-03-22)**: Refactored `_deepstack_process` to use `masked_scatter_` + elementwise add instead of double boolean indexing (`h[mask,:] = h[mask,:] + embeds`). Halves nonzero calls per deepstack layer (2→1) and removes `.clone()`.

**Result**: Marginal improvement (~0.2% MFU), within noise of step-to-step variance. The `nonzero` calls are inherent to `masked_scatter_` — fully eliminating them would require pre-computed scatter indices from the dataloader (higher complexity). The remaining `nonzero` calls are dominated by the initial `masked_scatter` for embedding injection and the HF vision encoder internals.

### Tier 2: Medium Effort (est. +5-15% MFU)

#### 2.1 ~~Re-enable visual compile with `fullgraph=False`~~ — TESTED, REJECTED

**Problem**: Vision encoder runs uncompiled. BMM backward alone costs 1s/step.

**Action**: Changed compile to `fullgraph=False` to tolerate HF's `dynamo.disable` decorators.

**Result (2026-03-22)**: **Made things worse.** Steady-state MFU dropped from 13.5% → 12.1% (~10% regression). Memory increased from 116 GiB → 127 GiB (+11 GiB). Early steps much slower due to compilation overhead with many graph breaks.

**Root cause**: HF vision encoder has too many graph breaks (dynamo.disable decorators). Each break creates a separate compiled subgraph, so compile overhead > compute savings for this relatively small model. The correct fix is Tier 3 (native vision encoder) which avoids graph breaks entirely.

#### 2.2 ~~Standardize image resolution~~ — SKIPPED

Dynamic resolution support is a requirement. Not pursuing fixed-resolution approaches.

#### 2.3 Reduce NCCL overhead — DONE (ep=4→ep=2)

**Problem**: NCCL is 38% of GPU kernel time. `all_to_all` for MOE is 0.57s.

**Change (2026-03-22)**: `expert_parallel_degree = 4 → 2`

**Result**:
- **+5-8% MFU** across all steps. Peak MFU: **15.46%** (up from 14.49%)
- GPU utilization doubled: **24% → 53%**
- all_to_all halved (0.497s→0.244s), but FSDP AllGather/ReduceScatter increased (each rank holds 64 experts instead of 32)
- Net NCCL: 0.92s → 0.87s (-5%)
- Memory: 116→117 GiB (slightly higher due to more expert params per rank)

**Profiling with ep=2** (iteration 20, rank0, 2 steps = 4.73s):
- GPU kernel time: 2.50s (53% utilization)
- NCCL: 0.87s (35% of kernels)
- GEMM: 0.52s (21%)
- Attention: 0.34s (13%)
- Remaining idle: ~2.2s = item() sync (0.33s) + nonzero (0.13s) + **CPU overhead (~1.7s)**
- The 1.7s CPU overhead is the dominant bottleneck: eager vision encoder execution, Python interpreter, kernel launch overhead

### Tier 3: Native Vision Encoder — NEXT PRIORITY

Profiling confirms that **~1.7s/step of CPU overhead** from the uncompiled HF vision encoder is the dominant remaining bottleneck (47% GPU idle time with ep=2). A native implementation enabling `fullgraph=True` compile is the highest-impact remaining optimization.

#### 3.1 Native vision encoder implementation

**Problem**: HF's `Qwen3VLVisionModel` is not compile-friendly:
- Uses `torch._dynamo.decorators.disable` → breaks `fullgraph=True`
- Data-dependent shapes in attention → graph breaks
- Eager Python loops → CPU overhead dominates
- `fullgraph=False` tested and **rejected** (overhead > savings)

**Goal**: Rewrite the SigLIP-2 vision encoder in TorchTitan-native PyTorch, enabling `fullgraph=True` torch.compile. Target: eliminate the 1.7s/step CPU overhead.

**Architecture to reimplement** (from HF `Qwen3VLVisionModel`):

```
Input: pixel_values [total_patches, in_channels*temporal_patch_size*patch_size*patch_size]
       grid_thw [num_images, 3]

1. Patch embedding (Conv3D-equivalent via linear projection)
2. Rotary position embedding (2D RoPE for H, W)
3. N transformer blocks (depth=27 for 30B):
   - LayerNorm → Attention (with 2D RoPE) → Residual
   - LayerNorm → MLP (SwiGLU) → Residual
   - Extract deepstack features at specific layers [8, 16, 24]
4. Spatial merge: merge spatial_merge_size×spatial_merge_size patches → 1 token
5. Output projection to text model dimension

Key compile-breaking patterns to avoid:
- No torch._dynamo.decorators.disable
- No data-dependent shapes (use static or mark_dynamic)
- No Python-level loops over variable-length sequences
- Use flash attention or SDPA (not manual attention with bmm)
```

**Implementation plan**:
1. Extract HF model weights and verify architecture dimensions
2. Implement native transformer blocks with standard attention (SDPA)
3. Implement 2D RoPE without dynamic shape ops
4. Implement spatial merge as a static reshape + linear
5. Implement deepstack feature extraction
6. Verify numerical equivalence with HF model
7. Enable `fullgraph=True` compile and benchmark

**Expected impact**: +10-20% MFU (eliminate 1.7s CPU overhead, enable kernel fusion)

#### 3.2 Multi-worker data loading

**Problem**: Data loading (55-147ms) is not overlapped with GPU compute. Single-process loading.

**Action**: Enable multi-process data workers. Blocked by PIL segfault issue with CUDA + multiprocessing (#2073).

**Note**: Currently not the primary bottleneck (data loading is <5% of step time).

#### 3.3 Fused vision token injection

**Problem**: `masked_scatter` calls for embedding injection use `nonzero` internally.

**Action**: Pre-compute scatter indices in the dataloader (CPU) and pass to model. Use `index_put_` with pre-computed indices instead of `masked_scatter`.

**Note**: Would be addressed naturally by 3.1 if the native encoder can output embeddings in the right layout.

---

## Summary of Results

| Optimization | Status | MFU Impact | Cumulative |
|-------------|--------|------------|------------|
| Baseline (post-rebase) | Done | 13.5% | 13.5% |
| 1.1 Eliminate item() calls | Done | +0.5-1% | ~14.3% |
| 1.2 Reduce nonzero | Done | ~0 (marginal) | ~14.3% |
| 2.1 Visual compile fullgraph=False (HF) | Rejected | -1.5% regression | — |
| 2.2 Standardize resolution | Skipped | — | — |
| 2.3 ep=4→ep=2 | Done | +1% | ~15.3% |
| 3.1 Native vision encoder (no compile) | Done | +2-4% steady, +34% warmup | ~15.8% |
| 3.1b Visual compile (native, static) | Blocked | Inductor dynamic shape bug | — |
| 3.1c Visual compile (native, dynamic=True, SDPA) | Done | +6% peak | ~16.8% |
| 3.1d FlexAttention + compiler.disable | Done | Same as 3.1c (~16.4%) | ~16.4% |
| 3.1e Batched tensor ops (gather indices) | Done | Same MFU, cleaner code | ~16.3% |
| 4.1 seq_len 8192→10240 | Done | +18% TPS, peak 19.1% MFU* | **19.1%*** |
| 4.2 MFU calculation fix (include vision FLOPs) | Done | More accurate reporting | — |

\* MFU after 4.2 includes vision encoder FLOPs (~0.3 patches/token). Not directly comparable to earlier numbers. Equivalent old-calc MFU ~16.8%.

**Current best: 19.1% MFU*** (seq_len=10240, native + SDPA + dynamic=True, commit 559e0e07)

**Also tested:**
- seq_len=12288: **21.2% MFU** but OOMs on outlier batches (step 6)
- seq_len=16384: OOMs immediately

### Profiled Gap Analysis (seq_len=8192, iteration 20, rank0, 2 steps = 4.57s)

GPU utilization: **52%** (48% idle). Step time: 2.28s/step.

**GPU kernel time: 2.37s**
| Category | Time | % |
|----------|------|---|
| NCCL | 0.86s | 36.4% |
| GEMM | 0.51s | 21.5% |
| Attention | 0.33s | 14.0% |
| Index/Gather | 0.25s | 10.6% |
| Other | 0.22s | 9.1% |
| Elementwise | 0.20s | 8.4% |

**CPU overhead (the 48% idle = ~2.2s)**
| Source | Time | Notes |
|--------|------|-------|
| BmmBackward | 0.44s | Vision attn backward, uncompiled due to SDPA graph break |
| Compiled region dispatch | 0.93s | 4 regions × 0.23s, framework overhead |
| PythonDispatchMode | 0.48s | FSDP parameter management |
| item() sync | 0.12s | 76 calls (compile guards + FSDP) |
| nonzero | 0.13s | 7 calls (masked_scatter) |
| aten::_to_copy | 0.19s | dtype conversions |

**NCCL detail:**
- AllGather (FSDP): 0.35s — now larger than MOE
- SendRecv (MOE all_to_all): 0.25s
- ReduceScatter: 0.23s

### Completed Steps

1. **seq_len=12288 + fsdp_reshard=always** — DONE (commit 938034fe)
   `fsdp_reshard_after_forward="always"` provided ~5 GiB headroom for seq_len=12288.
   Result: **21.5% peak MFU, 20% steady**. No OOM. Memory: 126 GiB (90%).

2. **NCCL communication overlap** — ALREADY IN PLACE
   `fsdp_reshard_after_forward="always"` enables FSDP AllGather/ReduceScatter overlap.
   Profiling confirms AllGather dropped from 0.35s → 0.23s. MOE SendRecv (0.36s) is now
   the dominant NCCL cost and is inherently synchronous.

3. **Compile vision attention backward** — TESTED, REJECTED
   Padded batched SDPA: regression from 21% → 15% MFU. The pad/unpad overhead + wasted
   compute on padding tokens outweighs the BmmBackward savings. Per-image SDPA chunking
   remains the best approach for now.

### Profiled Bottleneck (seq_len=12288, reshard=always)

GPU utilization: **50%**. Step: 3.14s. BmmBackward now 0.76s (#1 CPU overhead).

| Top CPU costs | Time |
|--------------|------|
| PythonDispatchMode (FSDP) | 0.87s |
| BmmBackward (vision attn bwd) | 0.76s |
| CompiledFunction | 0.53s |
| aten::sort (MOE routing) | 0.46s |
| item() sync | 0.34s |

### Compile Vision Attention Backward — ALL APPROACHES TESTED, NONE IMPROVED MFU

| Approach | Result | Why |
|----------|--------|-----|
| FlexAttention + BlockMask | Same MFU (16.4% vs 16.8%) | BlockMask creation overhead |
| Padded batched SDPA | -6% regression (15% vs 21%) | Pad/unpad + wasted compute |
| Compiled inner SDPA function | -1-5% regression (20% vs 21%) | 270 compile dispatches/step |

The BmmBackward (0.76s) is the cost of per-image attention without padding waste. All alternatives introduce overhead that exceeds the BmmBackward savings.

### mark_dynamic — TESTED, NOT APPLICABLE

| Approach | Result |
|----------|--------|
| `mark_dynamic` inside forward() | Fails: "Attempt to trace forbidden callable" |
| `mark_dynamic` in caller + no `dynamic=True` | OOM: compiler specializes per shape, recompiles 25 variants |
| `maybe_mark_dynamic` + `dynamic=True` | Redundant, same behavior as just `dynamic=True` |

Conclusion: `dynamic=True` on the whole encoder is the correct approach. `mark_dynamic` is designed for standalone compiled functions, not whole-module compile with FSDP.

### Future Directions

1. **Reduce MOE overhead**: `aten::sort` (0.46s) from token routing is a new top cost.

2. **Flash attention with cu_seqlens**: PyTorch's flash attention backend supports `cu_seqlens` natively for variable-length sequences without padding. If `F.scaled_dot_product_attention` with `enable_flash=True` can accept cu_seqlens, this would be the ideal solution — no padding, no per-image loop, compiled backward.

3. **Overlap vision/language compute** (high effort): Async CUDA streams.

4. **Pre-compute vision features** (high effort): For frozen encoder SFT.

3. **Overlap vision/language compute** (high effort): Run vision encoder on separate CUDA stream.

4. **Pre-compute vision features** (high effort): Cache features for frozen encoder during SFT.

## Appendix: Training Results

### VQAv2 (steps 1-20, no visual compile)

```
step:  1  loss: 11.860  tps:   197  mfu:  0.81%  memory:  78GiB (56%)
step:  3  loss:  8.708  tps: 3,544  mfu: 14.62%  memory: 113GiB (81%)
step:  6  loss: 10.585  tps: 3,428  mfu: 14.14%  memory: 116GiB (83%)
step: 10  loss:  7.243  tps: 3,317  mfu: 13.69%  memory: 116GiB (83%)
step: 14  loss:  4.689  tps: 3,448  mfu: 14.23%  memory: 116GiB (83%)
step: 16  loss:  5.038  tps:    77  mfu:  0.32%  memory: 116GiB (83%)  ← large image step
step: 18  loss:  4.818  tps: 3,380  mfu: 13.95%  memory: 116GiB (83%)
step: 20  loss:  3.440  tps: 3,262  mfu: 13.46%  memory: 116GiB (83%)
```

### C4 Text-Only (reference, from previous experiments)

```
step:  5  loss: 12.236  tps: 6,586  mfu: 33.61%  memory: 116GiB (83%)
step: 10  loss: 11.584  tps: 6,526  mfu: 33.31%  memory: 117GiB (83%)
```
