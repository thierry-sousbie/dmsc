import ctypes
import json
import multiprocessing
import os
import statistics
import sys
import time
from contextlib import contextmanager

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dmsc import compute_dmsc, generate_noisy_landscape

try:
    libc = ctypes.CDLL(None)
except Exception:
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"))


@contextmanager
def suppress_c_stdout():
    """
    Temporarily redirects OS-level file descriptor 1 (stdout) to /dev/null,
    while aggressively flushing C-level buffers to prevent delayed printing.
    """
    # yield
    # return
    sys.stdout.flush()  # Flush Python's buffers
    libc.fflush(None)
    devnull = os.open(os.devnull, os.O_WRONLY)  # Flush C streams
    old_stdout = os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        # Force C++ to flush its printf buffer into devnull before restoring the file descriptor to the terminal.
        sys.stdout.flush()
        libc.fflush(None)
        os.dup2(old_stdout, 1)
        os.close(old_stdout)
        os.close(devnull)


def benchmark_extraction(name, func, img_tensor, threshold, num_runs=6, run_profiler=False, is_batched=False, **kwargs):
    """Run a warmup followed by timed extraction and optional profiling."""

    # Warmup
    warmup_start = time.perf_counter()
    with suppress_c_stdout():
        _ = func(img_tensor, threshold, **kwargs)
        if img_tensor.device.type == "cuda":
            torch.cuda.synchronize()
        elif img_tensor.device.type == "mps":
            torch.mps.synchronize()
    warmup_end = time.perf_counter()
    warmup_ms = (warmup_end - warmup_start) * 1000

    # Always expand to physically separate blocks to prevent L2 cache cheating
    test_tensor = img_tensor.unsqueeze(0).expand(num_runs, -1, -1).contiguous()

    times_ms = []
    total_crit_pts = 0

    if is_batched:
        # Run the entire batch exactly ONCE
        start = time.perf_counter()
        with suppress_c_stdout():
            res_list = func(test_tensor, threshold, **kwargs)
            if img_tensor.device.type == "cuda":
                torch.cuda.synchronize()
            elif img_tensor.device.type == "mps":
                torch.mps.synchronize()
        end = time.perf_counter()

        # Calculate Average time PER IMAGE
        batch_time_ms = (end - start) * 1000
        times_ms = [batch_time_ms / num_runs]

        for res in res_list:
            total_crit_pts += len(res.max_pts) + len(res.min_pts) + len(res.sad_pts)
    else:
        # Standard sequential runs - explicitly prevent memory recycling to ensure fair comparison
        res_list = []
        for i in range(num_runs):
            start = time.perf_counter()
            with suppress_c_stdout():
                res = func(test_tensor[i], threshold, **kwargs)
                if img_tensor.device.type == "cuda":
                    torch.cuda.synchronize()
                elif img_tensor.device.type == "mps":
                    torch.mps.synchronize()
            end = time.perf_counter()

            times_ms.append((end - start) * 1000)
            res_list.append(res)  # Keep it alive to prevent caching allocator recycling
            total_crit_pts += len(res.max_pts) + len(res.min_pts) + len(res.sad_pts)

    # 3. Stats Output
    avg_time = statistics.mean(times_ms)
    median_time = statistics.median(times_ms)
    std_time = statistics.stdev(times_ms) if len(times_ms) > 1 else None
    avg_crit = total_crit_pts // num_runs

    spread = f"{std_time:5.2f}" if std_time is not None else "  n/a"
    print(
        f"{name:<30} | {warmup_ms:>8.2f} ms | {avg_time:>7.2f} ± {spread} ms"
        f" | {median_time:>7.2f} ms | {avg_crit:>12} pts"
    )

    # 4. Profiler Pass
    if run_profiler:
        activities = [ProfilerActivity.CPU]
        if img_tensor.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        safe_name = name.replace(" ", "_").replace("-", "").lower()
        trace_filename = f"trace_{safe_name}_t{threshold}.json"

        with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
            with record_function(f"{name}_extraction"):
                with suppress_c_stdout():
                    _ = func(img_tensor, threshold, **kwargs)
                    if img_tensor.device.type == "cuda":
                        torch.cuda.synchronize()
                    elif img_tensor.device.type == "mps":
                        torch.mps.synchronize()

        prof.export_chrome_trace(trace_filename)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))

    return {
        "configuration": name,
        "threshold": threshold,
        "warmup_ms": warmup_ms,
        "times_ms": times_ms,
        "mean_ms": avg_time,
        "median_ms": median_time,
        "std_ms": std_time,
        "critical_points": avg_crit,
        "batch_size": num_runs if is_batched else 1,
    }


def run_all_benchmarks(
    enable_profiler=False,
    test_cpu=True,
    test_gpu=True,
    trace_valleys=True,
    trace_ridges=True,
    trace_peaks=True,
    trace_basins=True,
    test_batches=True,
    num_threads=None,
    resolution=None,
    seed=1234,
    output_json=None,
):
    if resolution is None:
        resolution = 2048

    H, W = resolution, resolution
    # H, W = 1024, 1024
    # H, W = 512, 512

    if num_threads is None or num_threads < 0:
        num_threads = multiprocessing.cpu_count()

    torch.manual_seed(seed)
    print(f"Generating {H}x{W} noisy landscape (seed={seed})...")
    img_cpu = generate_noisy_landscape(H, W, z_scale=1.0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    img_gpu = img_cpu.to(device)
    thresholds = [0.0, 0.25]
    results = []

    def record(*args, **kwargs):
        result = benchmark_extraction(*args, **kwargs)
        results.append(result)

    for t in thresholds:
        print(f"\n{'=' * 85}")
        opt_str = f"{trace_valleys=}, {trace_ridges=}, {trace_peaks=}, {trace_basins=}"
        print(f" BENCHMARK({opt_str}): Persistence Threshold = {t}")
        print(f"{'=' * 85}")
        print(
            f"{'Configuration':<30} | {'Warmup':>11} | {'Hot (Mean ± Std) / image':>25}"
            f" | {'Median':>10} | {'Avg Crit Pts':>16}"
        )
        print("-" * 100)

        if test_cpu:
            # 1. CPU Single Thread
            torch.set_num_threads(1)
            record(
                "CPU Seq - 1 Thread",
                compute_dmsc,
                img_cpu,
                t,
                num_runs=2,
                run_profiler=enable_profiler,
                trace_valleys=trace_valleys,
                trace_ridges=trace_ridges,
                trace_peaks=trace_peaks,
                trace_basins=trace_basins,
            )

            # 2. CPU Max Threads (Sequential)
            torch.set_num_threads(num_threads)
            record(
                f"CPU Seq - {num_threads} Threads",
                compute_dmsc,
                img_cpu,
                t,
                num_runs=10,
                run_profiler=enable_profiler,
                trace_valleys=trace_valleys,
                trace_ridges=trace_ridges,
                trace_peaks=trace_peaks,
                trace_basins=trace_basins,
            )

            # 3. CPU Max Threads (Batched)
            if test_batches:
                record(
                    f"CPU Batch - {num_threads} Threads",
                    compute_dmsc,
                    img_cpu,
                    t,
                    num_runs=10,
                    run_profiler=enable_profiler,
                    is_batched=True,
                    trace_valleys=trace_valleys,
                    trace_ridges=trace_ridges,
                    trace_peaks=trace_peaks,
                    trace_basins=trace_basins,
                )

        # Hardware Accelerated (GPU)
        if test_gpu and device.type != "cpu":
            torch.set_num_threads(num_threads)

            # 4. GPU (Sequential)
            record(
                f"GPU Seq - {device.type.upper()} Driver",
                compute_dmsc,
                img_gpu,
                t,
                num_runs=10,
                run_profiler=enable_profiler,
                trace_valleys=trace_valleys,
                trace_ridges=trace_ridges,
                trace_peaks=trace_peaks,
                trace_basins=trace_basins,
            )

            # 5. GPU (Batched)
            if test_batches:
                record(
                    f"GPU Batch - {device.type.upper()} Driver",
                    compute_dmsc,
                    img_gpu,
                    t,
                    num_runs=10,
                    run_profiler=enable_profiler,
                    is_batched=True,
                    trace_valleys=trace_valleys,
                    trace_ridges=trace_ridges,
                    trace_peaks=trace_peaks,
                    trace_basins=trace_basins,
                )

    if output_json:
        report = {
            "seed": seed,
            "resolution": [H, W],
            "num_threads": num_threads,
            "torch_version": str(torch.__version__),
            "device": str(device),
            "results": results,
        }
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote benchmark results to {output_json}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Topology Extraction Benchmark")
    parser.add_argument("--profile", action="store_true", help="Run an additional pass with torch.profiler")
    parser.add_argument(
        "--no-valleys", action="store_false", dest="trace_valleys", default=True, help="Disable tracing valleys"
    )
    parser.add_argument(
        "--no-ridges", action="store_false", dest="trace_ridges", default=True, help="Disable tracing ridges"
    )
    parser.add_argument(
        "--no-peaks", action="store_false", dest="trace_peaks", default=True, help="Disable tracing peaks"
    )
    parser.add_argument(
        "--no-basins", action="store_false", dest="trace_basins", default=True, help="Disable tracing basins"
    )
    parser.add_argument("--resolution", type=int, default=2048, help="Image resolution")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads to use (-1 -> max avail.)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used to generate the landscape")
    parser.add_argument("--output-json", help="Write raw timings and metadata to this JSON file")
    args = parser.parse_args()

    run_all_benchmarks(
        enable_profiler=args.profile,
        trace_valleys=args.trace_valleys,
        trace_ridges=args.trace_ridges,
        trace_peaks=args.trace_peaks,
        trace_basins=args.trace_basins,
        num_threads=args.num_threads,
        resolution=args.resolution,
        seed=args.seed,
        output_json=args.output_json,
    )
    if args.profile:
        print("Visit http://ui.perfetto.dev to check your profiling results.")
