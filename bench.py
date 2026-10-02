import argparse, csv, os, statistics, torch
from transformers import AutoModelForDepthEstimation

# ---------- 1. Command-line flags ----------
p = argparse.ArgumentParser()
p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
p.add_argument("--channels-last", action="store_true")
p.add_argument("--compile", action="store_true")
p.add_argument("--cudagraph", action="store_true")
p.add_argument("--size", type=int, default=518)   # multiple of 14
p.add_argument("--profile", action="store_true")
args = p.parse_args()

DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

# ---------- 2. Load the model onto the GPU ----------
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf").to("cuda", dtype=DT).eval()

# ---------- 3. Create the input ----------
x = torch.randn(1, 3, args.size, args.size, device="cuda", dtype=DT)

# ---------- 4. Optional: channels_last memory layout ----------
if args.channels_last:
    model = model.to(memory_format=torch.channels_last)
    x = x.contiguous(memory_format=torch.channels_last)

# ---------- 5. Optional: torch.compile ----------
fn = torch.compile(model) if args.compile else model

def forward():
    return fn(pixel_values=x).predicted_depth

# ---------- 6. Optional: CUDA Graph ----------
if args.cudagraph:
    with torch.inference_mode():
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):              # warm up on a side stream (required before capture)
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):           # record all kernel launches of one forward pass
            static_out = forward()
    def run():
        graph.replay()                          # one call re-launches the whole recorded graph
        return static_out
else:
    run = forward

# ---------- 7. Timing ----------
def bench(iters=200, warmup=50):
    with torch.inference_mode():
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); run(); end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    times.sort()
    return statistics.median(times), times[int(0.95 * len(times))]

# Build a label from all enabled options, e.g. "fp16+cl+cudagraph"
label = args.dtype
if args.channels_last: label += "+cl"
if args.compile:       label += "+compile"
if args.cudagraph:     label += "+cudagraph"

os.makedirs("results", exist_ok=True)

# ---------- 8a. Profile mode ----------
if args.profile:
    from torch.profiler import profile, ProfilerActivity
    bench(iters=10)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            for _ in range(10): run()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace(f"results/trace_{label}.json")

# ---------- 8b. Benchmark mode ----------
else:
    med, p95 = bench()
    print(f"{label:22s} median {med:6.2f} ms   p95 {p95:6.2f} ms   ~{1000/med:5.1f} FPS")
    new = not os.path.exists("results/results.csv")
    with open("results/results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["config", "median_ms", "p95_ms", "gpu", "torch", "size"])
        w.writerow([label, f"{med:.2f}", f"{p95:.2f}",
                    torch.cuda.get_device_name(0), torch.__version__, args.size])

# ---------- 9. Correctness check: CUDA Graph output vs normal execution ----------
if args.cudagraph:
    with torch.inference_mode():
        ref = forward().float().clone()    # normal execution = reference
        out = run().float().clone()        # CUDA Graph replay on the same input
        torch.cuda.synchronize()
    max_abs = (out - ref).abs().max().item()
    mean_rel = ((out - ref).abs().mean() / ref.abs().mean()).item() * 100
    print(f"CUDA Graph vs normal: max abs diff = {max_abs:.6f}, mean rel diff = {mean_rel:.4f}%")