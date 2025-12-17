# Offline Data Preprocessing for Qwen3-VL Training

**Status**: Implementation Plan  
**Goal**: Eliminate data loading bottleneck by preprocessing and packing samples offline  
**Expected Impact**: 3-4x faster training (0.2s vs 0.6s per step), 80-85% GPU utilization (vs 50-70%)

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Solution Architecture](#solution-architecture)
3. [Implementation Plan](#implementation-plan)
4. [Usage Workflow](#usage-workflow)
5. [Performance Projections](#performance-projections)
6. [Maintenance & Updates](#maintenance--updates)

---

## Problem Statement

### Current Bottleneck: Online Data Processing

**Observed Performance** (with online packing):
```
GPU Utilization: 30-40% (idle 60-70% of time!)
CPU Utilization: 97-98% (maxed out)
Step Time: 0.6-0.8s per step
MFU: 30-45%
```

**Root Causes**:
1. **PIL Image Decoding**: 60-100ms per step (~200 images per packed batch)
2. **Sample Packing**: 50-80ms per step (processing ~200 samples, sorting, padding)
3. **Torch.compile Warmup**: 20-40 steps due to variable packed sequence lengths
4. **Tokenization**: 10-20ms per step (text processing)

**Total CPU overhead**: 120-200ms per step (GPU only needs 100-150ms!)

**Note**: Offline preprocessing takes ~1.7s per packed sample (measured)

### Why Workers Don't Help

- Workers offload I/O (PIL decoding, tokenization)
- But they cause **segmentation faults** with CUDA + multiprocessing
- And CPU is bottlenecked on **computation** (packing), not I/O
- Total CPU work remains the same whether in main process or workers

---

## Solution Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Offline Preprocessing                         │
│  (Run ONCE, takes 4-6 hours)                                    │
│                                                                  │
│  1. Load HuggingFace dataset                                    │
│  2. For each sample:                                            │
│     - Decode PIL image                                          │
│     - Tokenize text                                             │
│     - Process vision features                                   │
│  3. Pack samples (buffer_size=75)                               │
│  4. Save each sample individually as TensorDict memmap:         │
│     /checkpoints/xxie-sandbox/preprocessed_cache/               │
│       vqav2_validation_seq8192_buf75_<hash>/                    │
│         ├── samples/                                            │
│         │   ├── sample_000001/  (TensorDict memmap)            │
│         │   ├── sample_000002/  (TensorDict memmap)            │
│         │   └── ...                                             │
│         ├── metadata.json        (config)                      │
│         └── packing_stats.json   (packing stats)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Training (Fast!)                               │
│  (Every run, instant startup)                                   │
│                                                                  │
│  1. Check if cache exists for config                            │
│  2. Load with LazyStackedTensorDict (instant, memory-mapped!)   │
│  3. DP shard: dataset[rank::world_size]                         │
│  4. Start training immediately:                                 │
│     - No PIL decoding                                           │
│     - No tokenization                                           │
│     - No packing                                                │
│     - Variable sequence lengths (natural!)                      │
│  5. GPU Utilization: 80-85%                                     │
│  6. Step Time: 0.2-0.3s                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Cache Key Strategy

Cache is **automatically invalidated** when any config changes:

```python
cache_key = f"{dataset}_{split}_seq{seq_len}_buf{buffer}_model{hash}"

# Examples:
# vqav2_validation_seq8192_buf75_a3f4c2d1
# vqav2_validation_seq4096_buf50_a3f4c2d1  # Different config
```

**Benefits**:
- ✅ Different configs get different caches (no conflicts)
- ✅ Automatic cache selection (no manual management)
- ✅ Safe to experiment (old caches preserved)

---

## Implementation Plan

### Phase 1: Create Preprocessing Infrastructure

#### 1.1 Create Preprocessing Script

**File**: `torchtitan/experiments/qwen3_vl/scripts/preprocess_cached.py`

**Functions**:
- `get_cache_key()` - Generate deterministic cache identifier
- `get_cache_dir()` - Construct cache directory path
- `preprocess_and_cache()` - Main preprocessing function
  - Load dataset from HuggingFace
  - Load Qwen3-VL processor
  - Process samples: PIL decode → tokenize → pack
  - Save to cache with metadata
  - Return statistics

**CLI Interface**:
```bash
# pack the dataset
python -m torchtitan.experiments.qwen3_vl.datasets.preprocess_cached   --dataset vqav2  --seq-len 8192   --buffer-size 64   --model-path /checkpoints/xxie-sandbox/Qwen/Qwen3-VL-30B-A3B-Instruct   --cache-dir /checkpoints/xxie-sandbox/preprocessed_cache --max-samples 2048

# inspect after 
python -m torchtitan.experiments.qwen3_vl.scripts.inspect_cache --cache-dir /data/xxie-sandbox/preprocessed_cache_td/vqav2_seq8192_buf64_08cc5770/
```


#### 1.2 Create Cached Dataset Class

**File**: `torchtitan/experiments/qwen3_vl/datasets/vl_datasets.py` (extend existing)

**Class**: `CachedVLDataset(IterableDataset, Stateful)`

**Features**:
- Load preprocessed samples from cache
- Shard across data parallel ranks
- Support checkpointing (stateful)
- Infinite iteration support

**Methods**:
```python
__init__(cache_dir, dp_rank, dp_world_size, infinite)
__iter__()  # Yield preprocessed samples
load_state_dict()  # Checkpoint support
state_dict()
```

#### 1.3 Modify Dataloader Builder

**File**: `torchtitan/experiments/qwen3_vl/datasets/vl_datasets.py`

**Function**: `build_vl_dataloader()` (modify existing)

**Logic**:
```python
def build_vl_dataloader(...):
    # 1. Determine cache directory from config
    cache_dir = get_cache_dir(...)
    
    # 2. Check if cache exists
    if use_cache and cache_exists:
        # Use fast cached dataset!
        dataset = CachedVLDataset(cache_dir, ...)
    else:
        # Fall back to online preprocessing
        logger.warning("Cache not found, using online preprocessing (slow!)")
        dataset = HuggingFaceVLDataset(...)
    
    # 3. Build dataloader
    return ParallelAwareDataloader(dataset, ...)
```

### Phase 2: Configuration Integration

#### 2.1 Add Config Parameters

**File**: `torchtitan/experiments/qwen3_vl/train_configs/qwen3_vl_30b_moe.toml`

**New Section**:
```toml
[training]
# Existing params...
dataset = "vqav2"
seq_len = 8192

# Offline preprocessing cache settings
use_preprocessed_cache = true  # Enable automatic cache usage
cache_dir = "/checkpoints/xxie-sandbox/preprocessed_cache"
force_preprocess = false  # Set to true to regenerate cache
```

#### 2.2 Update JobConfig

**File**: `torchtitan/config/job_config.py`

**Add Fields** to `Training` class:
```python
@dataclass
class Training:
    # ... existing fields ...
    use_preprocessed_cache: bool = True
    cache_dir: str = "/checkpoints/xxie-sandbox/preprocessed_cache"
    force_preprocess: bool = False
```

## Performance Projections

### Training Speed Comparison

#### Scenario: 3000 steps (6 epochs of VQAv2 validation)

**Option A: Online Packing (Current)**
```
Compilation: Steps 1-40 (variable shapes, slow)
├─ Step 1-40:    40 × 2s = 80s     (compiling)
└─ Step 41-3000: 2960 × 0.6s = 1776s (training)
Total: 1856s = 31 minutes
GPU Utilization: 60% average
Effective GPU Time: 18.6 minutes
```

**Option B: Offline Preprocessing**
```
Preprocessing: 4 hours (ONE TIME)
Compilation: Steps 1-5 (fixed shapes, fast!)
├─ Step 1-5:     5 × 1s = 5s       (compiling)
└─ Step 6-3000:  2995 × 0.2s = 599s (training)
Total Training: 604s = 10 minutes
GPU Utilization: 85% average
Effective GPU Time: 8.5 minutes
```

**Performance Gain**: 3.1x faster per training run (31 min → 10 min)

### Expected Metrics

**After Offline Preprocessing**:
```
Metric              Online    Offline   Improvement
───────────────────────────────────────────────────
Step Time           0.6s      0.2s      3x faster
GPU Utilization     60%       85%       +25%
CPU Utilization     98%       40%       -58% (headroom!)
MFU                 35%       55%       +57%
Compilation Steps   20-40     2-5       8x faster
Memory Usage        Same      Same      No change
```

---

## Appendix: Packed Sample Analysis

### Sample Structure Deep Dive

Based on analysis of actual preprocessed cache (`vqav2_seq8192_buf64_08cc5770`):

#### 1. Pixel Values Shape: [29600, 1536]

**Interpretation**:
- **29,600 total vision patches** from **24 images** packed together
- **1,536** = Qwen3-VL vision encoder embedding dimension
- Images use **dynamic resolution encoding**

**Variable Image Sizes**:
```
Image 0: 1×32×40 = 1,280 patches (portrait)
Image 1: 1×32×40 = 1,280 patches (portrait)
Image 2: 1×40×32 = 1,280 patches (landscape)
Image 3: 1×40×32 = 1,280 patches (landscape)
Image 4: 1×32×40 = 1,280 patches (portrait)
...
Average: ~1,233 patches per image (29,600 ÷ 24)
```

**Why Dynamic Resolution?**
- Qwen3-VL preserves aspect ratios while encoding
- Each 28×28 pixel region → 1 patch
- Images resized to fit token budget efficiently

#### 2. Samples Per Packed Sequence

**Evidence from inspection**:
- **24 images** in `image_grid_thw` tensor
- **48 conversation turns** detected (24 Q + 24 A)
- Each VQA sample = 1 image + 1 Q&A pair

**Packing Efficiency**:
- Original samples: ~200-400 tokens each
- Packed together: **8,280 tokens total**
- **~20x better GPU utilization** vs individual samples

#### 3. Sequence Length = 8280 (Padded, Not Truncated)

**Analysis Results**:
```
Sequence Length: 8,280 tokens
Pad Tokens (151643): 348 found
EOS at end: No (sequence was padded)
Max configured: 8,192
```

**Why exactly 8280?**
- Packer accumulates samples until reaching `max_seq_len=8192`
- Samples exceeding limit saved for next pack
- Result padded to consistent length for efficient batching
- Formula: `8280 = actual_content + 348_pad_tokens`

**This confirms padding, not truncation** ✓

#### 4. Storage Size Analysis

**Per Packed Sample**:
```
pixel_values: ~181 MB  (29,600 patches × 1,536 dim × 4 bytes/float32)
input_ids:    ~0.03 MB (8,280 tokens × 4 bytes/int64)
labels:       ~0.03 MB (8,280 tokens × 4 bytes/int64)
Other fields: ~0.01 MB (position_ids, image_grid_thw)
───────────────────────
Total:        ~181 MB per packed sample
```

**Per Chunk** (64 packed samples):
```
64 samples × 181 MB = ~11.6 GB per chunk
```

**Full Dataset** (VQAv2 validation, 214K samples):
```
214K samples ÷ 24 samples/pack = ~8,917 packed samples
8,917 ÷ 64 per chunk = ~140 chunks
Total storage: ~1.6 TB
```

#### 5. Why `weights_only=False`?

**Technical Explanation**:
- Preprocessed data contains **`transformers.BatchFeature`** objects
- PyTorch 2.6 changed default: `weights_only=False` → `True`
- `BatchFeature` not in safe globals allowlist
- Using `weights_only=False` allows loading custom Python objects

**Security Context**:
- `weights_only=True` prevents arbitrary code execution
- Our preprocessed data is safe (we created it)
- Required for backwards compatibility

**Alternative**:
Could convert `BatchFeature` to pure tensors during preprocessing to avoid this requirement.

#### 6. Loading Performance Optimization

**Current Bottleneck**:
- Loading **~10GB chunks** with vision data is slow
- Each worker loads full chunk into memory

**Optimization Strategies**:

**A. Use New ZIP Serialization**:
```python
torch.save(samples, file, _use_new_zipfile_serialization=True)
```
- Faster loading for large tensors
- Better compression

**B. Async Pre-loading**:
```python
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)
next_future = executor.submit(load_chunk, next_path)
# Process current chunk...
next_chunk = next_future.result()  # Ready!
```

### Summary Statistics

| Metric | Value | Notes |
|--------|-------|-------|
| **Samples/pack** | 24 VQA pairs | Efficient packing |
| **Images/pack** | 24 images | Variable resolution |
| **Vision patches** | 29,600 patches | ~1,233 per image |
| **Sequence length** | 8,280 tokens | Padded (348 pad tokens) |
| **Storage/sample** | ~181 MB | Vision data dominates |
| **Chunk size** | ~11.6 GB | 64 samples × 181MB |
| **Dataset size** | ~1.6 TB | Full VQAv2 validation |

**Key Insight**: Vision encoder outputs (`pixel_values`) dominate storage at ~181MB per packed sample, while text tokens are negligible at ~60KB. Any storage optimization must focus on vision data.
