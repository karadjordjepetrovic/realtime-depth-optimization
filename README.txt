# Real-Time Depth Estimation: Fitting a Vision Transformer into a Game Frame Budget

**Goal:** run monocular depth estimation ([Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)) on a consumer GPU fast enough for real-time use, measured the way a game engine measures a frame: median and p95 milliseconds against a 16.6 ms (60 FPS) budget.

**Result:** **19.1 ms → 6.86 ms (2.8×, 52 → 146 FPS)** on an RTX 3070. That is about 41% of a 60 FPS frame, with output identical to normal execution and 0.07% mean error vs FP32.

![Game frame and its predicted depth map](assets/frame_vs_depth.png)
<!-- TODO: put one game frame and its depth map side by side in assets/frame_vs_depth.png -->

---

## Results

Batch size 1, 518×518 input, model inference only (preprocessing and CPU-to-GPU transfer excluded). Each row is 200 timed runs after 50 warm-up runs, measured with CUDA events.

| Configuration | Median | p95 | FPS | Speedup |
|---|---|---|---|---|
| FP32 (baseline, eager) | 19.13 ms | 19.58 ms | 52 | 1.00× |
| FP16 | 10.92 ms | 11.62 ms | 92 | 1.75× |
| BF16 | 11.05 ms | 12.02 ms | 90 | 1.73× |
| FP16 + `torch.compile` | 14.63 ms | 15.10 ms | 68 | 1.31× ⚠️ slower than plain FP16 |
| FP16 + `channels_last` | 10.40 ms | 11.00 ms | 96 | 1.84× |
| FP16 + CUDA Graph | 7.07 ms | 7.12 ms | 141 | 2.71× |
| **FP16 + `channels_last` + CUDA Graph** | **6.86 ms** | **7.31 ms** | **146** | **2.79×** |

Raw data: [`results/results.csv`](results/results.csv)

---

## What the profiler showed, step by step

### 1. FP32 baseline: the tensor cores were idle

PyTorch Profiler, about 18.1 ms of GPU work per frame:

| Part | Kernel | Per frame | Share |
|---|---|---|---|
| Attention (12 layers) | `fmha_cutlassF_f32` | ~7.1 ms | ~40% |
| Linear layers (72 matrix multiplies) | `ampere_sgemm_*` | ~6.1 ms | ~34% |
| Convolutions (depth head) | cuDNN Winograd | ~2.8 ms | ~15% |
| Element-wise and other | `add_`, LayerNorm, GELU, … | ~2.0 ms | ~11% |

- `sgemm` means FP32 matrix multiplies on regular CUDA cores. The RTX 3070's tensor cores were not used.
- Attention ran on PyTorch's FP32 fallback kernel, because FlashAttention supports only 16-bit types.
- Attention dominates because its cost grows with the square of the number of tokens: 1,369 image patches at 518×518.

### 2. FP16: tensor cores and FlashAttention, 1.75× faster

| Part | FP32 | FP16 | Why |
|---|---|---|---|
| Linear layers | 6.1 ms | 2.6 ms | Tensor-core GEMMs (`s16816gemm`, `xmma_gemm`) |
| Attention | 7.1 ms | 1.4 ms (5×) | `flash_fwd_kernel` (FlashAttention) |
| Convolutions | 2.8 ms | ~1.6 ms | `cutlass_tensorop_f16` kernels |

**Accuracy vs FP32**, measured on a real game frame:

| Precision | Mean relative error |
|---|---|
| FP16 | **0.071%** |
| BF16 | 0.680% |

BF16 has fewer mantissa bits (7 vs 10), so it is about 10× less precise here at the same speed. **FP16 is the right choice for this model.**

### 3. After FP16, the bottleneck moved from the GPU to the CPU

The FP16 profile showed only **~7.2 ms of GPU work** in a **10.9 ms** frame. For about 3.7 ms per frame the GPU was idle, waiting for Python to launch the next of hundreds of small kernels.

**Hypothesis:** removing launch overhead should bring the frame time close to the ~7.2 ms of GPU work.

### 4. CUDA Graphs confirmed it: 10.9 → 7.07 ms

A CUDA Graph records every kernel launch of one forward pass once, then replays it with a single call, like re-submitting a pre-recorded command list instead of re-recording it every frame.

- **Measured 7.07 ms**, matching the prediction.
- **Frame-time jitter almost disappeared:** p95 − median dropped from ~0.6 ms to under 0.1 ms.
- **Correctness:** the CUDA Graph output is bit-identical to normal execution (max abs diff = 0.0).

### 5. `channels_last`: removing hidden layout conversions

The FP16 profile showed `nchwToNhwcKernel` running **45 times per frame (~0.34 ms)**. cuDNN was converting tensors from NCHW to the NHWC layout its tensor-core convolution kernels prefer. Storing the model and input in `channels_last` (NHWC) format removed the conversions: −0.66 ms without CUDA Graphs, −0.22 ms with them.

### 6. `torch.compile` made it slower

FP16 + `torch.compile` (default mode) measured **14.6 ms vs 10.9 ms** without it. This is reported as measured; the cause is not diagnosed yet. Suspects:

- **Graph breaks** in the Hugging Face model code, which split the model into separately compiled pieces.
- **Kernel replacement:** the compiler swapping cuDNN or FlashAttention kernels for generated Triton kernels that are slower here.

Investigating this is on the roadmap.

---

## How to reproduce

**Environment:**
- GPU: RTX 3070 8 GB, driver 576.52
- OS: Windows 11 + WSL2 (Ubuntu 24.04)
- Software: Python 3.12, PyTorch `<VERSION>`+cu128 <!-- TODO: copy the version from results.csv -->

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers pillow matplotlib

python demo.py                                              # one frame -> depth map (sanity check)
python accuracy.py                                          # FP16 / BF16 error vs FP32
python bench.py --dtype fp32                                # baseline
python bench.py --dtype fp16 --channels-last --cudagraph    # fastest configuration
python bench.py --dtype fp16 --profile                      # per-kernel breakdown + trace (open in ui.perfetto.dev)
```

`bench.py` flags: `--dtype {fp32,fp16,bf16}`, `--channels-last`, `--compile`, `--cudagraph`, `--size N` (a multiple of 14), `--profile`.

**Measurement method:**
- CUDA events placed in the GPU stream, so the timing is GPU-side, not Python-side.
- 50 warm-up runs (clock ramp-up, allocator, cuDNN algorithm selection), then 200 timed runs.
- `synchronize()` after each run, so each sample is single-frame latency.
- Median and p95 reported, plus GPU name and PyTorch version, for reproducibility.

## Repository

| Path | What it is |
|---|---|
| `demo.py` | Run the model on one frame and save the depth map |
| `accuracy.py` | FP16/BF16 vs FP32 error on a real frame |
| `bench.py` | Benchmark and profiler with optimization flags |
| `results/` | `results.csv` and profiler tables/traces |

## Roadmap

- [ ] **Custom fused CUDA kernel:** residual add + LayerNorm (memory-bound element-wise chain), compared with a Triton version.
- [ ] **TensorRT** engine for comparison with the PyTorch path.
- [ ] **Diagnose the `torch.compile` regression:** graph-break logs, then `mode="reduce-overhead"`.
- [ ] **Roofline analysis** of the remaining GEMM, attention and convolution kernels with Nsight Compute.
- [ ] **End-to-end latency,** including preprocessing and transfer, on a live video stream.
- [ ] **Target:** under 5 ms per frame, a realistic slice of a game's GPU budget.

---

*Model: [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf). See its model card for the license. Gameplay frames were captured by the author.*
