"""Analyze profiler trace for performance bottleneck breakdown."""
import json
import sys
from collections import defaultdict

trace_file = sys.argv[1] if len(sys.argv) > 1 else "/checkpoints/xxie-sandbox/titan/output/qwen3-vl-30b/profile_trace/iteration_20/rank0_trace.json"

print(f"Loading {trace_file}...")
with open(trace_file) as f:
    data = json.load(f)
events = data.get("traceEvents", [])
print(f"Total events: {len(events)}")

# 1. GPU kernel breakdown
kernel_cats = defaultdict(lambda: [0, 0])
for e in events:
    cat = str(e.get("cat", ""))
    name = str(e.get("name", ""))
    dur = e.get("dur", 0)
    if not isinstance(dur, (int, float)) or cat != "kernel":
        continue
    nl = name.lower()
    if "nccl" in nl:
        kernel_cats["NCCL"][0] += 1; kernel_cats["NCCL"][1] += dur
    elif "flash" in nl or "sdpa" in nl or "cudnn" in nl:
        kernel_cats["Attention"][0] += 1; kernel_cats["Attention"][1] += dur
    elif "cutlass" in nl or "gemm" in nl or "xmma" in nl or "nvjet" in nl:
        kernel_cats["GEMM"][0] += 1; kernel_cats["GEMM"][1] += dur
    elif "index" in nl or "gather" in nl or "scatter" in nl:
        kernel_cats["Index/Gather"][0] += 1; kernel_cats["Index/Gather"][1] += dur
    elif "elementwise" in nl or "vectorized" in nl or "copy" in nl:
        kernel_cats["Elementwise"][0] += 1; kernel_cats["Elementwise"][1] += dur
    elif "nonzero" in nl:
        kernel_cats["NonZero"][0] += 1; kernel_cats["NonZero"][1] += dur
    else:
        kernel_cats["Other"][0] += 1; kernel_cats["Other"][1] += dur

total_kernel = sum(v[1] for v in kernel_cats.values())
print(f"\nGPU kernel time: {total_kernel/1e6:.2f}s (2 profiled steps)")
for cat, (cnt, dur) in sorted(kernel_cats.items(), key=lambda x: -x[1][1]):
    pct = 100 * dur / total_kernel if total_kernel > 0 else 0
    print(f"  {cat:20s}: {dur/1e6:.3f}s ({pct:5.1f}%)  x{cnt}")

# 2. Step time + GPU utilization
steps = []
for e in events:
    name = str(e.get("name", ""))
    dur = e.get("dur", 0)
    if not isinstance(dur, (int, float)):
        continue
    if name.startswith("ProfilerStep#"):
        steps.append((name, dur))
step_total = sum(d for _, d in steps)
print(f"\nProfiler steps: {len(steps)}, total: {step_total/1e6:.2f}s")
for name, dur in steps:
    print(f"  {name}: {dur/1e6:.3f}s")
if step_total > 0:
    print(f"GPU utilization: {100*total_kernel/step_total:.0f}%")
    print(f"GPU idle: {100*(1 - total_kernel/step_total):.0f}%")

# 3. NCCL detail
nccl_ops = defaultdict(lambda: [0, 0])
for e in events:
    name = str(e.get("name", ""))
    dur = e.get("dur", 0)
    cat = str(e.get("cat", ""))
    if not isinstance(dur, (int, float)):
        continue
    if cat == "kernel" and "nccl" in name.lower():
        simple = name.split("(")[0]
        nccl_ops[simple][0] += 1
        nccl_ops[simple][1] += dur
print("\nNCCL detail:")
for name, (cnt, dur) in sorted(nccl_ops.items(), key=lambda x: -x[1][1]):
    print(f"  {name[:55]:55s}: {dur/1e6:.3f}s  x{cnt}")

# 4. Sync points
item_count = item_dur = nz_count = nz_dur = 0
for e in events:
    name = str(e.get("name", ""))
    dur = e.get("dur", 0)
    cat = str(e.get("cat", ""))
    if not isinstance(dur, (int, float)):
        continue
    if name == "aten::_local_scalar_dense":
        item_count += 1; item_dur += dur
    if cat == "cpu_op" and "nonzero" in name and name != "aten::is_nonzero":
        nz_count += 1; nz_dur += dur
print(f"\nSync points:")
print(f"  item() calls: {item_count}, total: {item_dur/1e6:.3f}s")
print(f"  nonzero calls: {nz_count}, total: {nz_dur/1e6:.3f}s")

# 5. Top CPU ops
cpu_ops = defaultdict(lambda: [0, 0])
for e in events:
    cat = str(e.get("cat", ""))
    name = str(e.get("name", ""))
    dur = e.get("dur", 0)
    if not isinstance(dur, (int, float)) or cat != "cpu_op":
        continue
    cpu_ops[name][0] += 1
    cpu_ops[name][1] += dur
print("\nTop 15 CPU ops:")
for name, (cnt, dur) in sorted(cpu_ops.items(), key=lambda x: -x[1][1])[:15]:
    print(f"  {name:50s}: {dur/1e6:.3f}s  x{cnt}")

# 6. Compilation time
compile_dur = 0
for e in events:
    name = str(e.get("name", ""))
    dur = e.get("dur", 0)
    if not isinstance(dur, (int, float)):
        continue
    if "Compilation" in name:
        compile_dur += dur
print(f"\nCompilation time: {compile_dur/1e6:.3f}s")
