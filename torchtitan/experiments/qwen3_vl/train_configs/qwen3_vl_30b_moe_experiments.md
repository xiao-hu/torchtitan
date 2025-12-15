## Train Qwen3-30B-A3B Model:
Setup:
```
pip install 'transformers>=4.37.0'  # For Qwen3-VL support
```
Training command:
```shell
CONFIG_FILE='./torchtitan/experiments/qwen3_vl/train_configs/qwen3_vl_30b_moe.toml' ./run_train.sh
```
Hardware: 8xH200 GPUs

#### Exp 1
No packing
Config:
```toml
[training]
local_batch_size = 1
seq_len = 8192
[compile]
enable=false
```

Performance:
```
[rank0]:[titan] 2025-12-15 23:05:00,244 - root - INFO - step:  2  loss: 14.3789  grad_norm: 1230.7789  memory: 63.54GiB(45.48%)  tps: 25  tflops: 0.76  mfu: 0.08%
```

#### Exp 2
Enable Packing
Config:
```toml
[training]
local_batch_size = 1
seq_len = 8192
[compile]
enable=false
```

Performance:
```
[rank0]:[titan] 2025-12-15 23:18:41,747 - root - INFO - step:  2  loss: 20.4725  grad_norm: 702.1525  memory: 120.53GiB(86.27%)  tps: 35  tflops: 1.42  mfu: 0.14%
```