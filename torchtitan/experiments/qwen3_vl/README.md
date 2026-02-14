# Qwen3-VL: Vision-Language Model Experiment

Training [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) in TorchTitan.

## Architecture

- **Vision encoder**: HF `Qwen3VLVisionModel` (SigLIP-2 based) with thin wrapper
- **Text decoder**: Qwen3 with MOE, extended with DeepStack visual feature injection
- **Position encoding**: 3D RoPE (MRoPE) for temporal, height, width dimensions
- **Parameters**: 31B total (2B dense + 29B sparse MoE, 128 experts, top-8)

## Project Structure

```
torchtitan/experiments/qwen3_vl/
├── model/              # Model implementation
│   ├── args.py         # Qwen3VLModelArgs, Qwen3VLVisionArgs
│   ├── model.py        # Qwen3VLModel, Qwen3VLTextModel
│   ├── vision.py       # Qwen3VLVisionEncoder wrapper
│   └── state_dict_adapter.py  # HF ↔ TorchTitan checkpoint conversion
├── datasets/           # VL dataset infrastructure
│   ├── vl_datasets.py  # HuggingFaceVLDataset, VL_DATASETS registry
│   ├── packing.py      # VL sample packing
│   └── utils.py        # Preprocessing & collation utilities
├── infra/              # Distributed training
│   └── parallelize.py  # Hybrid parallelization (Vision FSDP + Language TP/EP/FSDP)
├── train_configs/      # TOML training configurations
├── train_spec.py       # TrainSpec integration
└── tests/              # Unit tests
```

## Quick Start

```bash
# Train with VQAv2 (vision-language)
CONFIG_FILE="./torchtitan/experiments/qwen3_vl/train_configs/qwen3_vl_30b_moe.toml" ./run_train.sh

# Train with C4 (text-only, auto-detected)
CONFIG_FILE="./torchtitan/experiments/qwen3_vl/train_configs/qwen3_vl_30b_moe_c4.toml" ./run_train.sh
```

## Datasets

The dataloader auto-detects dataset type from the config:

| Dataset | Type | Config |
|---------|------|--------|
| `vqav2` | Vision-Language | `dataset = "vqav2"` |
| `c4` | Text-only | `dataset = "c4"` |

VL datasets use sample packing for improved GPU utilization. New datasets can be added to the `VL_DATASETS` registry in `vl_datasets.py`.

## Parallelization

- **Vision encoder**: FSDP2 only (small, ~540M params)
- **Language model**: Full TP + EP + CP + FSDP via `parallelize_qwen3`

## Dependencies

- PyTorch with CUDA
- `transformers` (for `Qwen3VLProcessor` and vision encoder)
- Existing TorchTitan infrastructure (`torchtitan/models/qwen3/`, `torchtitan/models/moe/`)