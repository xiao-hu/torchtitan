## setup env
```shell
# image: nvcr.io/nvidia/pytorch:25.11-py3
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
cd <torchtitan_dir>
pip install -r requirements.txt
pip install 'transformers>=4.37.0' # for qwen3-vl
```

### Test installation:
```python
import torchtitan
print(torchtitan.__version__)
from torchtitan.models.qwen3 import Qwen3Model, Qwen3StateDictAdapter, qwen3_args
model_flavor: str = "30B-A3B"
model_args = qwen3_args[model_flavor]
model = Qwen3Model(model_args)
print(f"Created Qwen3Model with {sum(p.numel() for p in model.parameters()):,} parameters")
# Created Qwen3Model with 30,532,122,624 parameters
```

## Train Qwen3-30B-A3B Model:
```shell
CONFIG_FILE="./custom_task/qwen3_30b.toml" ./run_train.sh
```
Hardware: 8xH200 GPUs

### MFU improvement:

#### Exp 1

Config:
```toml
[training]
local_batch_size = 2
seq_len = 4096
[activation_checkpoint]
mode = "full"  # ["none", "selective", "full"]
[compile]
enable=false
[parallelism]
tensor_parallel_degree = 4 
context_parallel_degree = 1
expert_parallel_degree = 2
expert_tensor_parallel_degree = 4
```

Performance:
```
[rank0]:[titan] 2025-12-13 06:45:34,345 - root - INFO - step: 530  loss:  5.9866  grad_norm:  0.5402  memory: 63.36GiB(45.35%)  tps: 829  tflops: 23.15  mfu: 2.34%
```

#### Exp 2 (*)

Config:
```toml
[training]
local_batch_size = 12
seq_len = 8192
[activation_checkpoint]
mode = "selective"  # ["none", "selective", "full"]
[compile]
enable=true
```

Performance:
```
[rank0]:[titan] 2025-12-13 06:57:02,246 - root - INFO - step: 10  loss: 10.0544  grad_norm:  6.6503  memory: 116.82GiB(83.61%)  tps: 2,228  tflops: 83.74  mfu: 8.47%
```
Which is very reasonable for MOE models.


#### Exp 3

Increase batch size based on Exp 2 config:
```toml
[training]
local_batch_size = 16
```

Performance:
```
[rank0]:[titan] 2025-12-13 07:10:23,426 - root - INFO - step: 20  loss:  9.3015  grad_norm:  2.1376  memory: 135.04GiB(96.65%)  tps: 1,588  tflops: 59.66  mfu: 6.03%
```
Memory pressure is too high. MFU drops significantly. local_batch_size 12 is better.

#### Exp 4

Increase EP and decrease TP:
```toml
[training]
local_batch_size = 8
seq_len = 8192
[activation_checkpoint]
mode = "selective"  # ["none", "selective", "full"]
[compile]
enable=true
[parallelism]
tensor_parallel_degree = 1
context_parallel_degree = 1
expert_parallel_degree = 8
expert_tensor_parallel_degree = 1
```

Performance:
```
[rank0]:[titan] 2025-12-14 04:46:02,423 - root - WARNING - 25 CUDA memory allocation retries.
[rank0]:[titan] 2025-12-14 04:41:25,709 - root - INFO - step: 10  loss:  9.9698  grad_norm:  5.7460  memory: 134.43GiB(96.21%)  tps: 4,959  tflops: 186.35  mfu: 18.84%
```

#### Exp 5
Reduce batch size to 6.

```toml
[training]
local_batch_size = 6
```
Performance:
```
[rank0]:[titan] 2025-12-14 05:11:59,164 - root - INFO - step: 10  loss:  9.9277  grad_norm:  4.8983  memory: 115.41GiB(82.60%)  tps: 5,748  tflops: 215.99  mfu: 21.84%
```

#### Exp 6
Reduce EP to 4.

```toml
[training]
local_batch_size = 6
[parallelism]
tensor_parallel_degree = 1
context_parallel_degree = 1
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1
```
Performance:
```
[rank0]:[titan] 2025-12-14 05:43:30,311 - root - INFO - step: 10  loss: 10.0371  grad_norm:  6.2420  memory: 116.48GiB(83.37%)  tps: 6,512  tflops: 244.72  mfu: 24.74%
```

Observation: decreasing EP does not increase memory pressure likly due to the reduced communicaiton buffer.

### Exp 7
Increase sequence length by 1.5x.

```toml
[training]
local_batch_size = 4
seq_len = 12288
[parallelism]
tensor_parallel_degree = 1
context_parallel_degree = 1
expert_parallel_degree = 4
expert_tensor_parallel_degree = 1
```

```
[rank0]:[titan] 2025-12-14 05:53:48,651 - root - INFO - Trainer is initialized with local batch size 4, global batch size 32, gradient accumulation steps 1, sequence length 12288, total steps 3000 (warmup 600)
[rank0]:[titan] 2025-12-14 05:55:38,070 - root - INFO - step: 10  loss:  9.9834  grad_norm:  4.4551  memory: 116.01GiB(83.03%)  tps: 6,260  tflops: 295.72  mfu: 29.90%
```

Key takeaways:
* TP is Inefficient for MoE. TP introduces communication overhead across every layer. Keeping $\mathbf{TP=1}$ is essential to avoid communication-intensive operations in non-MoE layers.
* ETP: With TP=1, keeping ETP=1 prevents unnecessary sharding within the experts, maximizing the computation-to-communication ratio on the expert devices.
* EP is Dominant for MoE. EP effectively shards the heavy MoE weights. Lowering EP from 8 to 4 reduced the complexity/latency of the All-to-All communication, leading to a direct MFU boost.
* Batch Size is for Saturation. After optimizing EP/seq_len, LBS is used to push memory usage to $80\% \rightarrow 90\%$. Once the communication bottleneck is hit, increasing LBS decreases MFU.
* Memory Headroom is Critical. Operating too close to $100\%$ causes memory thrashing, which forces low-level stalls and destroys MFU, regardless of parallelism strategy.
* Sequence Length: 
* Increasing $\mathbf{seq\_len}$ significantly boosted the Computation-to-Communication ($\frac{C}{A}$) Ratio.
