# Qwen3 VL MOE Implementation

Implementation of Qwen3 VLM (Vision-Language Model) with Mixture of Experts (MOE) to train "Qwen/Qwen3-VL-30B-A3B-Instruct"

## Overview

This implementation combines:
- **Qwen3 text-only model** with MOE support (from `torchtitan/models/qwen3/`)
- **SigLIP-2 vision encoder** (from `torchtitan/experiments/vlm/model/siglip2.py`)
- **DeepStack integration** - Visual features injected into early text decoder layers
- **Multi-dimensional RoPE** (MRoPE) for vision position encoding

## Architecture Components

### Key Features
- **Vision Encoder**: SigLIP-2 based encoder with patch merging
- **Text Decoder**: Qwen3 with MOE layers (sparse expert activation)
- **Projector**: Two-layer MLP mapping vision embeddings to text space
- **DeepStack**: Multi-layer visual feature integration
- **Special Tokens**: Image, video, vision_start, vision_end tokens
- **Position Encoding**: 3D RoPE for temporal, height, width dimensions

### MOE Configuration
- MOE layers inserted at specific decoder layers (`decoder_sparse_step`)
- Load balancing across experts during training
- Expert parallelism support for distributed training

## References

* **Qwen3 (text-only)**: `torchtitan/models/qwen3/`
* **VLM reference**: `torchtitan/experiments/vlm/model/` (Llama3Siglip2Transformer)
* **SigLIP-2 encoder**: `torchtitan/experiments/vlm/model/siglip2.py`
* **VLM datasets**: `torchtitan/experiments/vlm/datasets/`
* **HF implementation**: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py
* **Qwen3-30B-A3B config**: `custom_task/qwen3_30b.toml`

## Implementation Tasks

### Phase 1: Project Setup & Structure ✅
- [x] Create project structure in `torchtitan/experiments/qwen3_vl/`
  - `model/` - Core model implementation directory
  - `datasets/` - Multimodal dataset utilities (TBD)
  - `tests/` - Unit and integration tests (TBD)
  - `train_configs/` - Training configuration TOML files (TBD)
- [x] Create all `__init__.py` files with proper exports

**File Status:**
- ✅ Complete: `__init__.py`, `args.py`, `vision.py`, `README.md`
- ⏳ Scaffold: `model.py`, `state_dict_adapter.py`
- 📋 Planned: `datasets/`, `tests/`, `train_configs/`

### Phase 2: Model Arguments (args.py) ✅
- [x] Create `Qwen3VLVisionArgs` dataclass with HF-aligned field names
  - All fields match `Qwen3VLVisionConfig` exactly: depth, hidden_size, intermediate_size, etc.
  - Enables trivial conversion to HF config (`**config.__dict__`)
  - Includes DeepStack parameters (deepstack_visual_indexes: [8, 16, 24])
  - No dependency on Siglip2ModelArgs (independent, purpose-built config)
- [x] Create `Qwen3VLModelArgs` class extending `Qwen3ModelArgs`
  - Inherits text decoder config (dim, n_layers, n_heads, moe_enabled, etc.)
  - Adds vision encoder config (vision_config: Qwen3VLVisionArgs)
  - Adds special token IDs (image_token_id, video_token_id, vision_start_token_id, vision_end_token_id)
  - Adds DeepStack integration parameter (deepstack_visual_indexes at model level)
- [x] Define `SpecialTokens` dataclass for VLM tokens
  - Structured token management (img, video, vision_start, vision_end, pad)
  - Factory method `from_tokenizer()` for extracting from HuggingFace tokenizer
  - Includes ignore_id for loss masking

### Phase 3: Vision Components ✅
**Implementation**: Import HF Qwen3VLVisionModel with thin wrapper
- [x] Create `Qwen3VLVisionArgs` with HF-aligned field names
  - All fields match `Qwen3VLVisionConfig` exactly (depth, hidden_size, etc.)
  - Eliminates conversion overhead (trivial `**config.__dict__` mapping)
- [x] Import HF `Qwen3VLVisionModel` as-is
  - Proven implementation (3D patching, spatial merging, DeepStack)
  - Zero maintenance burden (upstream bug fixes)
  - No code duplication (~60 lines wrapper vs ~300+ reimplementation)
- [x] Create thin `Qwen3VLVisionEncoder` wrapper
  - Config conversion: TorchTitan `Qwen3VLVisionArgs` → HF `Qwen3VLVisionConfig`
  - Compatible forward signature for TorchTitan integration
  - No parallelism (vision encoder is small, sequential processing)
- [x] Update exports in `__init__.py`

**Benefits**:
- ✅ Proven implementation (all features work: videos, temporal patches, merging, DeepStack)
- ✅ Minimal code (60 lines vs 300-400)
- ✅ Zero conversion overhead (field names aligned)
- ✅ No external dependency at config level (only at runtime)

### Phase 4: Core Model Implementation (model.py) ✅ COMPLETE
**Architecture**
```
Qwen3Model (base text decoder)
    ↓ extends
Qwen3VLTextModel (+ DeepStack injection)
    ↑ used by
Qwen3VLModel (vision + text integration)
```

**Implemented Components**:
- [x] `Qwen3VLTextModel` class (extends Qwen3Model)
  - [x] `_deepstack_process()` - Injects visual features at masked positions
  - [x] Extended `forward()` - Layer-wise DeepStack injection during decoding
  - [x] Backward compatible with optional vision parameters
  
- [x] `Qwen3VLModel` class (main VLM)
  - [x] Combines Qwen3VLVisionEncoder + Qwen3VLTextModel
  - [x] `get_image_features()` - Encodes images and returns embeddings + deepstack features
  - [x] `get_video_features()` - Processes videos (delegates to image features)
  - [x] Main `forward()` - Complete vision-text integration pipeline
  
- [x] Helper functions implemented
  - [x] `get_placeholder_mask()` - Vision token position identification and validation (~30 lines)
  - [x] `get_rope_index()` - 3D position IDs (T, H, W) for vision + 1D for text (~150 lines)
  
**Implementation Details**:
- [x] **Vision token handling**
  - Scatter vision embeddings into text token positions using `masked_scatter`
  - Support both image_token and video_token
  - Handle grid_thw (temporal, height, width) parameters
  
- [x] **DeepStack integration**
  - Vision encoder returns features from multiple layers
  - Inject into text decoder layers after forward pass
  - Maintain visual position masks throughout
  - Support merging image + video features when both present
  
- [x] **Forward pass**
  - Get text embeddings from input_ids
  - Encode vision inputs (images/videos) through vision encoder
  - Scatter vision embeddings into text embedding positions
  - Calculate 3D position IDs via get_rope_index
  - Process through Qwen3VLTextModel with DeepStack injection
  - Return model outputs (logits)

Load Model:
```python
import torchtitan
print(torchtitan.__version__)
from torchtitan.experiments.qwen3_vl import Qwen3VLModel, qwen3_vl_args
model_flavor: str = "30B-A3B"
model_args = qwen3_vl_args[model_flavor]
model = Qwen3VLModel(model_args)
print(f"Created Qwen3VLModel with {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Phase 5: State Dict Adapter (state_dict_adapter.py)
- [ ] Analyze HF checkpoint structure
  - Vision encoder keys: `model.visual.*`
  - Text model keys: `model.language_model.*`
  - Projector keys: `model.projector.*`
  
- [ ] Implement key mappings
  - Vision encoder: `visual.embeddings.*`, `visual.blocks.*`, `visual.merger.*`
  - Text decoder: `language_model.layers.*` (handle MOE layers)
  - MOE experts: `layers.*.mlp.experts.*`, `layers.*.mlp.gate.*`
  - Projector: `projector.w1.weight`, `projector.w2.weight`
  
- [ ] Handle special cases
  - Expert parameters (gate_up_proj, down_proj)
  - Router weights
  - Embedding layers (tok_embeddings, lm_head)
  - Weight tying if enabled
  
- [ ] Test checkpoint conversion
  - Load from HF: `Qwen/Qwen3-VL-30B-A3B-Instruct`
  - Convert to TorchTitan format
  - Verify shapes and numerical accuracy

### Phase 6: Training Configuration
- [ ] Create training config TOML
  - Based on `custom_task/qwen3_30b.toml`
  - Add vision preprocessing parameters
  - Configure MOE parallelism (expert_parallel_degree=2, expert_tensor_parallel_degree=4)
  - Add dataset configuration for multimodal data
  
- [ ] Model parameters
  ```toml
  [model]
  name = "qwen3_vl_moe"
  flavor = "30B-A3B"
  hf_assets_path = "/path/to/Qwen3-VL-30B-A3B-Instruct"
  
  [model.vision]
  patch_size = 14
  temporal_patch_size = 2
  spatial_merge_size = 2
  ```

### Phase 7: Dataset Integration
- [ ] Adapt multimodal datasets from `torchtitan/experiments/vlm/datasets/`
- [ ] Support inputs:
  - `pixel_values`: Image tensors
  - `pixel_values_videos`: Video tensors
  - `image_grid_thw`: Grid dimensions for images
  - `video_grid_thw`: Grid dimensions for videos
- [ ] Handle special tokens in sequences
  - `<|vision_start|>`, `<|image|>`, `<|video|>`, `<|vision_end|>`
- [ ] Implement proper padding and masking

### Phase 8: Testing & Validation
- [ ] Unit tests
  - Vision encoder forward pass
  - Projector forward pass
  - Token scattering logic
  - DeepStack integration
  - MRoPE position encoding
  
- [ ] Integration tests
  - Full model forward pass with dummy data
  - Checkpoint save/load
  - State dict conversion HF ↔ TorchTitan
  
- [ ] Numerical validation
  - Compare with HF implementation
  - Check gradient correctness
  - Verify MOE load balancing

### Phase 9: Optimization & Parallelism
- [ ] Configure tensor parallelism for vision encoder
- [ ] Configure expert parallelism for MOE layers
- [ ] Test parallelism configurations:
  - TP=4, EP=2, FSDP=auto
  - Expert TP=4 for shared experts
- [ ] Enable activation checkpointing (selective mode)
- [ ] Test torch.compile support (note: MOE may have limitations)

### Phase 10: Documentation & Examples
- [x] Update README.md with implementation plan
- [ ] Add usage examples
- [ ] Document model architecture details
- [ ] Add training script example
- [ ] Document special considerations (MRoPE, DeepStack, etc.)

## Technical Challenges

### 1. Multi-dimensional RoPE (MRoPE)
Qwen3 VL uses 3D position encoding for vision inputs:
```python
position_ids.shape = (3, batch_size, seq_len)  # [temporal, height, width]
```
- Must implement `get_rope_index()` for proper position tracking
- Handle different grids for images vs videos
- Maintain position deltas across generation steps

### 2. DeepStack Integration
Visual features from multiple encoder layers injected into decoder:
```python
deepstack_visual_indexes = [0, 1, 2]  # inject into first 3 decoder layers
```
- Extract features from vision encoder layers
- Maintain visual position masks
- Inject at correct decoder layer positions

### 3. Variable Length Vision Sequences
Images and videos have different sizes and temporal dimensions:
```python
grid_thw = [[1, 24, 24], [4, 12, 12]]  # [temporal, height, width]
```
- Proper masking for different sizes
- Position encoding for variable grids
- Efficient batching of mixed sizes

### 4. MOE + Vision Integration
Combining MOE text layers with vision features:
- Ensure vision features propagate through MOE layers
- Load balancing with multimodal inputs
- Proper expert parallelism configuration

### 5. State Dict Conversion
HF checkpoint structure differs from TorchTitan:
- Nested MOE expert weights require careful mapping
- Vision encoder key transformations
- Handle weight tying between embeddings and lm_head

## Project Structure

```
torchtitan/experiments/qwen3_vl/
├── README.md                 # This file
├── __init__.py
├── model/
│   ├── __init__.py
│   ├── args.py              # Model configuration classes
│   ├── model.py             # Main Qwen3VLMoeModel implementation
│   └── state_dict_adapter.py # HF ↔ TorchTitan checkpoint conversion
├── datasets/                # Multimodal dataset utilities (reuse from vlm/)
├── train_configs/           # Training configuration TOML files
└── tests/                   # Unit and integration tests
```

## Usage Example

```python
from torchtitan.experiments.qwen3_vl.model import Qwen3VLMoeModel
from torchtitan.experiments.qwen3_vl.model.args import Qwen3VLModelArgs

# Create model config
model_args = Qwen3VLModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    vocab_size=151936,
    moe_enabled=True,
    num_experts=8,
    # ... other params
)

# Initialize model
model = Qwen3VLMoeModel(model_args)

# Forward pass
outputs = model(
    input_ids=input_ids,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    attention_mask=attention_mask,
)
```

## Dependencies

**Required:**
- PyTorch with CUDA support
- Existing Qwen3 model (`torchtitan/models/qwen3/`)
- SigLIP-2 vision encoder (`torchtitan/experiments/vlm/model/siglip2.py`)
- MOE infrastructure (`torchtitan/models/moe/`)
- VLM dataset utilities (`torchtitan/experiments/vlm/datasets/`)

**Optional:**
- HF `transformers` library (for checkpoint conversion)
- Multimodal evaluation datasets

## Estimated Timeline

- **Phase 1-2**: Setup and configuration (~3-4 hours)
- **Phase 3**: Vision components integration (~1 hour)
- **Phase 4**: Core model implementation (~8-12 hours)
- **Phase 5**: State dict adapter (~4-6 hours)
- **Phase 6-7**: Configuration and datasets (~3-4 hours)
- **Phase 8**: Testing and validation (~4-6 hours)
- **Phase 9-10**: Optimization and documentation (~3-4 hours)

**Total Estimated**: 25-40 hours

## Next Steps

1. ✅ Complete analysis and task breakdown
2. Create project structure (Phase 1)
3. Implement args.py with configuration classes (Phase 2)
4. Begin core model implementation (Phase 4)
5. Implement state dict adapter (Phase 5)
6. Test with HF checkpoint and validate outputs (Phase 8)
