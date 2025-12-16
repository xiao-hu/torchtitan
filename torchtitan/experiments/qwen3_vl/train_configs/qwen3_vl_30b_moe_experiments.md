# Qwen3-VL 30B-A3B Training Experiments

**Model**: Qwen3-VL-30B-A3B-Instruct  
**Hardware**: 8x H200 (141GB)  
**Dataset**: VQAv2 validation

## Setup

```bash
pip install 'transformers>=4.37.0'
CONFIG_FILE='./torchtitan/experiments/qwen3_vl/train_configs/qwen3_vl_30b_moe.toml' ./run_train.sh
```

---

## Experiments

### Exp 1: Baseline (No Packing)

```toml
local_batch_size = 1, seq_len = 8192, compile = false
```

**Results**: `memory: 63.54GiB (45%), tps: 25, mfu: 0.08%`

**Analysis**: Low GPU utilization due to short sequences (~400-800 tokens)

---

### Exp 2: Sample Packing

```toml
local_batch_size = 1, seq_len = 8192, compile = false
# Packing enabled with buffer_size=204
```

**Results**: `memory: 120.53GiB (86%), tps: 35, mfu: 0.14%`

**Analysis**:
- ✅ +40% throughput (25→35 tps)
- ✅ +75% MFU (0.08%→0.14%)
- ⚠️ Still very low MFU (should be 40-60%)
- ⚠️ CPU bottleneck (97-98% CPU utilization)

---

### Exp 3: Torch Compile

```toml
local_batch_size = 1, seq_len = 8192, compile = true
```

**Results**: `memory: 103.97GiB (74%), tps: 34, mfu: 0.14%`

**Analysis**:
- ✅ Reduced memory usage
- ❌ No MFU improvement (step 2 is still compiling)
- ❌ CPU remains bottleneck (100% utilization)

**Note**: Compile warmup takes 20-40 steps due to variable packed shapes

---

## Bottleneck Analysis

**GPU Profile**: 30-38% utilization (idle 60-70%)  
**CPU Profile**: 97-98% utilization (MAXED OUT!)

**Time per Step**:
```
PIL decode + load:  60-100ms  ← CPU bottleneck
Sample packing:     50-80ms   ← CPU bottleneck  
Tokenization:       10-20ms   ← CPU bottleneck
GPU compute:        100-150ms
──────────────────────────────
Total:              220-350ms
GPU idle:           70-200ms  ← 60-70% wasted!
```
