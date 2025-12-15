# Qwen3-30B-A3B MOE Training Experiments

Iterative optimization experiments to improve MFU from 2.34% to 29.90% on 8×H200 GPUs.

## Environment Setup

```shell
# Base image: nvcr.io/nvidia/pytorch:25.11-py3
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
cd <torchtitan_dir>
pip install -r requirements.txt
pip install 'transformers>=4.37.0'  # For Qwen3-VL support
```

### Verify Installation
```python
import torchtitan
print(torchtitan.__version__)
from torchtitan.models.qwen3 import Qwen3Model, Qwen3StateDictAdapter, qwen3_args

model_flavor = "30B-A3B"
model_args = qwen3_args[model_flavor]
model = Qwen3Model(model_args)
print(f"Created Qwen3Model with {sum(p.numel() for p in model.parameters()):,} parameters")
# Output: Created Qwen3Model with 30,532,122,624 parameters
```

## Training Command

```shell
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_30b_moe.toml" ./run_train.sh
```

**Hardware**: 8×H200 GPUs (140GB VRAM each)

---

## MFU Optimization Experiments

### Experiment 1: Baseline (Naive Config)

**Configuration**:
```toml
[training]
local_batch_size = 2
seq_len = 4096

[activation_checkpoint]
mode = "full"  # ["none", "selective", "full"]

[compile]
enable = false

[parallelism]
tensor_parallel_degree = 4 
context_parallel_degree = 1
expert_parallel_degree = 2
expert_tensor_parallel_degree = 4
```

**Result**:
```
[2025-12-13 06:45:34] step: 530  loss: 5.9866  memory: 63.36GiB (45.35%)  
tps: 829  tflops: 23.15  mfu: 2.34%
```

**Observation**: Low MFU due to conservative settings and disabled compile.

---

### Experiment 2: Enable Compile + Increase Throughput ⭐

**Configuration**:
```toml
[training]
local_batch_size = 12
seq_len = 8192

[activation_checkpoint]
mode = "selective"  # Reduced checkpointing overhead

[compile]
enable = true  # Critical optimization
```

**Result**:
```
[2025-12-13 06:57:02] step: 10  loss: 10.0544  memory: 116.82GiB (83.61%)
tps: 2,228  tflops: 83.74  mfu: 8.47%
```

**Improvement**: **3.6×** MFU increase (2.34% → 8.47%)  
**Key changes**: Torch compile + larger batch size + longer sequences

---

### Experiment 3: Push Memory Limits

**Configuration**: Increased `local_batch_size = 16` (from Exp 2)

**Result**:
```
[2025-12-13 07:10:23] step: 20  loss: 9.3015  memory: 135.04GiB (96.65%)
tps: 1,588  tflops: 59.66  mfu: 6.03%
```

**Observation**: **MFU decreased** (8.47% → 6.03%) due to memory thrashing at 96.65%.  
**Lesson**: Operating near 100% memory causes severe performance degradation.

---

### Experiment 4: Optimize Expert Parallelism

**Configuration**: Increased EP, decreased TP
```toml
[training]
local_batch_size = 8
seq_len = 8192

[activation_checkpoint]
mode = "selective"

[compile]
enable = true

[parallelism]
tensor_parallel_degree = 1     # Reduced from 4
context_parallel_degree = 1
expert_parallel_degree = 8      # Increased from 2
expert_tensor_parallel_degree = 1  # Reduced from 4
```

**Result**:
```
[2025-12-14 04:46:02] WARNING: 25 CUDA memory allocation retries
[2025-12-14 04:41:25] step: 10  loss: 9.9698  memory: 134.43GiB (96.21%)
tps: 4,959  tflops: 186.35  mfu: 18.84%
```

**Improvement**: **3.1×** MFU increase (6.03% → 18.84%)  
**Observation**: High memory pressure (96.21%) still causes allocation retries.

---

### Experiment 5: Balance Memory Usage

**Configuration**: Reduced `local_batch_size = 6` (from Exp 4)

**Result**:
```
[2025-12-14 05:11:59] step: 10  loss: 9.9277  memory: 115.41GiB (82.60%)
tps: 5,748  tflops: 215.99  mfu: 21.84%
```

**Improvement**: **1.2×** MFU increase (18.84% → 21.84%)  
**Key insight**: Healthy memory headroom (82.60%) eliminates allocation retries and improves throughput.

---

### Experiment 6: Reduce Communication Overhead

**Configuration**: Reduced `expert_parallel_degree = 4` (from 8)
```toml
[training]
local_batch_size = 6

[parallelism]
tensor_parallel_degree = 1
context_parallel_degree = 1
expert_parallel_degree = 4      # Reduced from 8
expert_tensor_parallel_degree = 1
```

**Result**:
```
[2025-12-14 05:43:30] step: 10  loss: 10.0371  memory: 116.48GiB (83.37%)
tps: 6,512  tflops: 244.72  mfu: 24.74%
```

**Improvement**: **1.1×** MFU increase (21.84% → 24.74%)  
**Key insight**: Lower EP reduces All-to-All communication latency while maintaining similar memory usage.

---

### Experiment 7: Optimize Compute-to-Communication Ratio ⭐

**Configuration**: Increased seq_len by 1.5×, adjusted batch size
```toml
[training]
local_batch_size = 4    # Reduced to maintain memory
seq_len = 12288          # Increased from 8192

[parallelism]
tensor_parallel_degree = 1
context_parallel_degree = 1
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1
```

**Result**:
```
[2025-12-14 05:53:48] Trainer initialized: local_batch_size=4, global_batch_size=32, 
                      gradient_accumulation_steps=1, seq_len=12288
[2025-12-14 05:55:38] step: 10  loss: 9.9834  memory: 116.01GiB (83.03%)
tps: 6,260  tflops: 295.72  mfu: 29.90%
```

**Improvement**: **1.2×** MFU increase (24.74% → 29.90%)  
**Final Result**: **12.8× improvement** from baseline (2.34% → 29.90%)

---

## Key Takeaways

### 1. **Tensor Parallelism (TP) is Inefficient for MOE**
- TP introduces communication overhead across **every layer** (including non-MOE layers)
- **Keeping TP=1** is essential to avoid unnecessary all-reduce operations
- MOE models benefit more from expert-level parallelism than tensor-level parallelism

### 2. **Expert Tensor Parallelism (ETP) Should Be Minimal**
- With TP=1, keeping **ETP=1** prevents unnecessary sharding within experts
- Maximizes computation-to-communication ratio on expert devices
- Reduces synchronization points during expert forward/backward passes

### 3. **Expert Parallelism (EP) is the Dominant Knob**
- EP effectively shards heavy MOE weights across devices
- However, **lower EP (4) outperforms higher EP (8)** due to:
  - Reduced All-to-All communication complexity
  - Lower communication buffer overhead
  - Better balance between parallelism and communication cost

### 4. **Batch Size for Memory Saturation**
- After optimizing parallelism, use batch size to target **80-90% memory usage**
- Once communication becomes bottleneck, **increasing batch size decreases MFU**
- Sweet spot: `local_batch_size = 4-6` with `seq_len = 8192-12288`

### 5. **Memory Headroom is Critical**
- Operating at >95% memory causes **thrashing and allocation retries**
- Aim for **80-85% memory utilization** for optimal performance
- Memory pressure destroys MFU regardless of parallelism strategy

### 6. **Sequence Length Boosts Compute-to-Communication Ratio**
- Increasing `seq_len` significantly improves C/A ratio (compute/all-to-all)
- Longer sequences amortize communication overhead over more computation
- **seq_len = 12288** provides best MFU with `batch_size = 4`

### 7. **Torch Compile is Essential**
- Provides **10-30% speedup** with minimal code changes
- Critical for achieving competitive MFU on modern GPUs
- Should be enabled in all production training runs

---

## Recommended Configuration

For **Qwen3-30B-A3B** on **8×H200 GPUs**:

```toml
[training]
local_batch_size = 4
seq_len = 12288

[activation_checkpoint]
mode = "selective"

[compile]
enable = true

[parallelism]
tensor_parallel_degree = 1
context_parallel_degree = 1
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1
```

**Expected Performance**: MFU ~29-30%, Memory ~83%, TPS ~6,200

---

## Optimization Checklist

When optimizing MOE training:
1. ✅ Enable torch compile first (quick win)
2. ✅ Set TP=1 and ETP=1 (reduce communication)
3. ✅ Tune EP (start with 4, then try 2 or 8)
4. ✅ Increase seq_len to maximize C/A ratio
5. ✅ Adjust batch size to target 80-85% memory
6. ✅ Use selective activation checkpointing
7. ✅ Monitor for memory allocation retries
8. ✅ Profile communication patterns if MFU plateaus

---

## References

- Model: [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- TorchTitan: https://github.com/pytorch/torchtitan
- Training config: `torchtitan/models/qwen3/train_configs/qwen3_30b_moe.toml`
