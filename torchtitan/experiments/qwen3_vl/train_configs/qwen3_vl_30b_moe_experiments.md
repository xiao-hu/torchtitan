# Qwen3-VL 30B-A3B Training Experiments

**Model**: Qwen3-VL-30B-A3B-Instruct  
**Hardware**: 8x H200 (141GB)  
**Dataset**: VQAv2 validation

## Setup

```bash
pip install 'transformers>=4.37.0 tensordict'
CONFIG_FILE='./torchtitan/experiments/qwen3_vl/train_configs/qwen3_vl_30b_moe.toml' ./run_train.sh
```

---

## Experiments

### Exp 1: Baseline (No Packing)

```toml
local_batch_size = 1
seq_len = 8192
compile = false
```

**Note**: `local_batch_size > 1` is not supported due to variable image resolutions across samples

**Results**: `memory: 63.54GiB (45%), tps: 25, mfu: 0.08%`

**Analysis**: Low GPU utilization due to short sequences (~400-800 tokens)

---

### Exp 2: Sample Packing

```toml
local_batch_size = 1
seq_len = 8192
compile = false
# Packing enabled with buffer_size=204
```

**Results**: `memory: 120.53GiB (86%), tps: 35, mfu: 0.14%`

**Analysis**:
- ✅ +40% throughput (25→35 tps)
- ✅ +75% MFU (0.08%→0.14%)
- ⚠️ Still very low MFU

---

### Exp 3: Torch Compile

```toml
local_batch_size = 1
seq_len = 8192
compile = true
```

**Results**: `memory: 103.97GiB (74%), tps: 34, mfu: 0.14%`

**Analysis**:
- ✅ Reduced memory usage
- MFU is mostly the same, suggesting other bottlenecks
**Bottleneck Analysis**

* Is visual model or language model the bottleneck?
* Visual data loading efficiency?
  * Loading and preprocessing visual data is more CPU intensive
  * Multi-process data workers are not supported in TorchTitan yet: https://github.com/pytorch/torchtitan/issues/2073
  * PIL causes **segmentation faults** with CUDA + multiprocessing

### Exp 4: Text-Only Training Baseline

**Goal**: Verify that Qwen3-VL maintains comparable performance when trained on text-only datasets (no vision overhead).

**Configuration**:
```toml
[training]
local_batch_size = 4  # Increased for text-only (no vision memory overhead)
seq_len = 12288
dataset = "c4"  # Text-only dataset from DATASETS registry
```

**Results** (steady-state performance after warmup):
```
[rank0]:[titan] 2025-12-17 01:18:26,708 - root - INFO - step:  1  loss: 12.4153  grad_norm:  8.2782  memory: 87.87GiB(62.89%)  tps: 945  tflops: 47.71  mfu: 4.82%
[rank0]:[titan] 2025-12-17 01:18:26,708 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[rank0]:[titan] 2025-12-17 01:18:34,942 - root - INFO - step:  2  loss: 12.3984  grad_norm:  8.0045  memory: 116.16GiB(83.14%)  tps: 5,970  tflops: 301.33  mfu: 30.47%
[rank0]:[titan] 2025-12-17 01:18:42,407 - root - INFO - step:  3  loss: 12.3513  grad_norm:  7.9040  memory: 116.16GiB(83.14%)  tps: 6,585  tflops: 332.35  mfu: 33.60%
[rank0]:[titan] 2025-12-17 01:18:49,878 - root - INFO - step:  4  loss: 12.3057  grad_norm:  8.1505  memory: 116.41GiB(83.32%)  tps: 6,580  tflops: 332.11  mfu: 33.58%
[rank0]:[titan] 2025-12-17 01:18:57,341 - root - INFO - step:  5  loss: 12.2364  grad_norm:  8.0841  memory: 116.41GiB(83.32%)  tps: 6,586  tflops: 332.44  mfu: 33.61%
[rank0]:[titan] 2025-12-17 01:19:04,815 - root - INFO - step:  6  loss: 12.1519  grad_norm:  8.1318  memory: 116.41GiB(83.32%)  tps: 6,577  tflops: 331.97  mfu: 33.57%
[rank0]:[titan] 2025-12-17 01:19:12,274 - root - INFO - step:  7  loss: 12.0584  grad_norm:  8.1898  memory: 116.41GiB(83.32%)  tps: 6,590  tflops: 332.64  mfu: 33.63%
[rank0]:[titan] 2025-12-17 01:19:19,772 - root - INFO - step:  8  loss: 11.8934  grad_norm:  9.1551  memory: 116.41GiB(83.32%)  tps: 6,556  tflops: 330.91  mfu: 33.46%
[rank0]:[titan] 2025-12-17 01:19:27,304 - root - INFO - step:  9  loss: 11.7361  grad_norm:  9.4663  memory: 116.51GiB(83.39%)  tps: 6,527  tflops: 329.44  mfu: 33.31%
[rank0]:[titan] 2025-12-17 01:19:34,836 - root - INFO - step: 10  loss: 11.5838  grad_norm:  9.6145  memory: 116.51GiB(83.39%)  tps: 6,526  tflops: 329.40  mfu: 33.31%
[rank0]:[titan] 2025-12-17 01:19:42,393 - root - INFO - step: 11  loss: 11.3803  grad_norm:  9.8774  memory: 116.51GiB(83.39%)  tps: 6,505  tflops: 328.34  mfu: 33.20%
[rank0]:[titan] 2025-12-17 01:19:49,999 - root - INFO - step: 12  loss: 11.2006  grad_norm:  9.5630  memory: 116.51GiB(83.39%)  tps: 6,462  tflops: 326.18  mfu: 32.98%
[rank0]:[titan] 2025-12-17 01:19:57,656 - root - INFO - step: 13  loss: 11.0067  grad_norm:  8.2690  memory: 116.51GiB(83.39%)  tps: 6,420  tflops: 324.06  mfu: 32.77%
[rank0]:[titan] 2025-12-17 01:20:05,335 - root - INFO - step: 14  loss: 10.8099  grad_norm:  8.3617  memory: 116.51GiB(83.39%)  tps: 6,401  tflops: 323.08  mfu: 32.67%
[rank0]:[titan] 2025-12-17 01:20:13,099 - root - INFO - step: 15  loss: 10.6878  grad_norm: 12.7063  memory: 116.51GiB(83.39%)  tps: 6,332  tflops: 319.60  mfu: 32.32%
[rank0]:[titan] 2025-12-17 01:20:20,849 - root - INFO - step: 16  loss: 10.5777  grad_norm: 12.1309  memory: 116.51GiB(83.39%)  tps: 6,343  tflops: 320.15  mfu: 32.37%
[rank0]:[titan] 2025-12-17 01:20:28,696 - root - INFO - step: 17  loss: 10.4500  grad_norm: 10.7103  memory: 116.51GiB(83.39%)  tps: 6,264  tflops: 316.19  mfu: 31.97%
[rank0]:[titan] 2025-12-17 01:20:36,556 - root - INFO - step: 18  loss: 10.3665  grad_norm:  8.8775  memory: 116.51GiB(83.39%)  tps: 6,255  tflops: 315.69  mfu: 31.92%
```

**Analysis**:
- ✅ **Excellent performance**: ~6,500 tps, 330 tflops, **33% MFU**
- ✅ **~185x throughput improvement** vs. VQA multimodal (35 tps → 6,500 tps)
- ✅ **~240x MFU improvement** vs. VQA multimodal (0.14% → 33%)
- ✅ Stable training with decreasing loss trajectory
- 💡 **Key insight**: CPU bottleneck is eliminated with text-only data (no PIL decoding, simpler tokenization)

---

### Exp 5: 
The bottleneck is from Visual part.
Tried cached dataset (cache preprocess visual data to tensors and load tensors directly during training), but MFU is not improved.

Revisit vision model parallelization.
FSDP was only applied to core blocks.
Extend FSDP to all visual models:

```
[rank0]:[titan] 2025-12-17 18:16:22,736 - root - INFO - step:  1  loss: 11.6715  grad_norm: 413993.1250  memory: 75.25GiB(53.86%)  tps: 150  tflops: 6.11  mfu: 0.62%                                             
[rank0]:[titan] 2025-12-17 18:16:50,801 - root - INFO - step:  2  loss:  9.9699  grad_norm: 30038.7969  memory: 102.62GiB(73.44%)  tps: 285  tflops: 11.64  mfu: 1.18%
[rank0]:[titan] 2025-12-17 18:17:00,445 - root - INFO - step:  3  loss: 11.5645  grad_norm: 20174.4258  memory: 102.83GiB(73.60%)  tps: 833  tflops: 33.98  mfu: 3.44%
[rank0]:[titan] 2025-12-17 18:17:10,540 - root - INFO - step:  4  loss: 16.1480  grad_norm: 4221.2612  memory: 102.83GiB(73.60%)  tps: 792  tflops: 32.33  mfu: 3.27%
[rank0]:[titan] 2025-12-17 18:17:21,030 - root - INFO - step:  5  loss:  7.3720  grad_norm: 496.6781  memory: 103.01GiB(73.72%)  tps: 771  tflops: 31.48  mfu: 3.18%
[rank0]:[titan] 2025-12-17 18:17:32,345 - root - INFO - step:  6  loss: 13.0652  grad_norm: 313.7617  memory: 103.18GiB(73.85%)  tps: 721  tflops: 29.43  mfu: 2.98%
[rank0]:[titan] 2025-12-17 18:17:47,824 - root - INFO - step:  7  loss:  7.5180  grad_norm: 70.6903  memory: 103.18GiB(73.85%)  tps: 521  tflops: 21.25  mfu: 2.15%
[rank0]:[titan] 2025-12-17 18:18:02,113 - root - INFO - step:  8  loss: 10.3346  grad_norm: 128.6145  memory: 103.25GiB(73.90%)  tps: 355  tflops: 14.48  mfu: 1.46%
[rank0]:[titan] 2025-12-17 18:18:20,413 - root - INFO - step:  9  loss:  8.3097  grad_norm: 55.0017  memory: 103.50GiB(74.08%)  tps: 443  tflops: 18.07  mfu: 1.83%
[rank0]:[titan] 2025-12-17 18:18:24,978 - root - INFO - step: 10  loss:  6.8966  grad_norm: 53.4530  memory: 103.50GiB(74.08%)  tps: 1,764  tflops: 71.98  mfu: 7.28%
```

Observations:
* MFU improved significantly, but not stable
* Disabling compile for the vision model decreases the model's performance significantly: MFU drops back to ~0.13%
* Still lags behind text-only model. 

Potential issues:
* variable text length, image numbers and resolutions in the training data lead to frequent recompiling

Identified issue with MFU calculation, visual model not counted (see [analysis](../mfu_analysis.md)). MFU underestimated by ~10%. Yet to fix.

### Exp 6: Text Sequence Padding

Pad sequence length to a multiple of 256 to reduce text model recompile: 
```
...
[rank0]:[titan] 2025-12-17 19:29:41,263 - root - INFO - step:  8  loss:  7.2810  grad_norm: 54.2424  memory: 103.27GiB(73.91%)  tps: 269  tflops: 10.98  mfu: 1.11%
[rank0]:[titan] 2025-12-17 19:29:54,609 - root - INFO - step:  9  loss: 11.3383  grad_norm: 140.1408  memory: 103.60GiB(74.15%)  tps: 614  tflops: 25.05  mfu: 2.53%
[rank0]:[titan] 2025-12-17 19:29:59,137 - root - INFO - step: 10  loss:  9.5201  grad_norm: 43.1138  memory: 103.60GiB(74.15%)  tps: 1,810  tflops: 73.87  mfu: 7.47%
```

* It seems that padding sequence length in collator doesn't help much.
* However, enable profiling and found that recompile is indeed the bottleneck and it takes over 70% GPU cycles.
* Likely bottleneck: Visual model recompiles frequently due to variable image numbers and resolution in the training data

## Summary:
* We enabled Qwen3-VL MoE. 
* The language model supports full N-D parallelism.
* The vision encoder is small and is parallelized via FSDP2
* When training Qwen3-VL-30B-A3B-Instruct on C4 (text only), we got over 30% MFU, matching what we observed for Qwen3 text-only model.
* When training Qwen3-VL-30B-A3B-Instruct on VQAV2 (text + visual), we got unstable MFU 2-7%.
  * Due to variable image resolutions in VQAV2, we cannot batch multiple samples (unless we pad pixels or resize images).
  * We enabling sample packing to improve training efficiency keeping local_batch_size=1.
  
## Future Work:
* Optimize the compilation of the visual model to reduce recompile (70% of the GPU time)
  * Do we need to support variable resolutions?
  * Do we need to support variable number of images per sample/sequence?
  * Optimize torch.compile visual model
    * General optimizations: e.g., fullgraph, compile mode, etc.
    * Targeted optimization: mark dynamic dimensions for pixel values/grid_thw, etc.
* Optimize sample packing: loss scaling, block mask
* Enable multiple data workers to improve VL data loading efficiency:
  * Need to solve the PIL segmentation fault error
  * May not be the major bottleneck for the VQAV2 dataset but needed to enable video data training
* Fix MFU calculation in TorchTitan (see [analysis](../mfu_analysis.md))
