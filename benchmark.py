import ctypes
import multiprocessing
import os
import statistics
import sys
import time
from contextlib import contextmanager

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from csrc.dmsc import compute_dmsc

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


def generate_noisy_landscape(H=4096, W=4096):
    y, x = torch.meshgrid(torch.linspace(-2, 2, H), torch.linspace(-2, 2, W), indexing="ij")
    peak1 = torch.exp(-((x - 1) ** 2 + (y) ** 2) / 0.5)
    peak2 = torch.exp(-((x + 1) ** 2 + (y) ** 2) / 0.5)
    valley1 = -0.5 * torch.exp(-((x) ** 2 + (y - 1) ** 2) / 0.5)
    valley2 = -1.0 * torch.exp(-((x) ** 2 + (y + 1) ** 2) / 0.5)

    z = peak1 + peak2 + valley1 + valley2
    noise = (torch.rand(H, W) - 0.5) * 0.1
    z += noise

    return z.to(torch.float32)


def benchmark_extraction(name, func, img_tensor, threshold, num_runs=6, run_profiler=False, is_batched=False, **kwargs):
    """Runs a warmup, measures it, averages hot runs, and optionally profiles."""

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

    # expand the tensor if testing batch parallelism
    if is_batched:
        test_tensor = img_tensor.unsqueeze(0).expand(num_runs, -1, -1).contiguous()
    else:
        test_tensor = img_tensor

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
        times_ms = [batch_time_ms / num_runs] * num_runs  # Hack to make the avg_time math identical

        for res in res_list:
            total_crit_pts += len(res.max_pts) + len(res.min_pts) + len(res.sad_pts)
    else:
        # Standard sequential runs
        for _ in range(num_runs):
            start = time.perf_counter()
            with suppress_c_stdout():
                res = func(test_tensor, threshold, **kwargs)
                if img_tensor.device.type == "cuda":
                    torch.cuda.synchronize()
                elif img_tensor.device.type == "mps":
                    torch.mps.synchronize()
            end = time.perf_counter()

            times_ms.append((end - start) * 1000)
            total_crit_pts += len(res.max_pts) + len(res.min_pts) + len(res.sad_pts)

    # 3. Stats Output
    avg_time = sum(times_ms) / num_runs
    std_time = statistics.stdev(times_ms) if num_runs > 1 and not is_batched else 0.0
    avg_crit = total_crit_pts // num_runs

    print(f"{name:<30} | {warmup_ms:>8.2f} ms | {avg_time:>7.2f} ± {std_time:>5.2f} ms | {avg_crit:>12} pts")

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
                    _ = func(test_tensor, threshold, **kwargs)
                    if img_tensor.device.type == "cuda":
                        torch.cuda.synchronize()
                    elif img_tensor.device.type == "mps":
                        torch.mps.synchronize()

        prof.export_chrome_trace(trace_filename)


def run_all_benchmarks(
    enable_profiler=False,
    test_cpu=True,
    test_gpu=True,
    trace_valleys=True,
    trace_ridges=True,
    trace_peaks=True,
    trace_basins=True,
    test_batches=True,
):

    H, W = 2048, 2048
    # H, W = 1024, 1024
    # H, W = 512, 512
    num_threads = multiprocessing.cpu_count()

    print(f"Generating {H}x{W} noisy landscape...")
    img_cpu = generate_noisy_landscape(H, W)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    img_gpu = img_cpu.to(device)
    thresholds = [0.0, 0.25]
    # thresholds = [0.25]

    for t in thresholds:
        print(f"\n{'=' * 85}")
        print(f" BENCHMARK: Persistence Threshold = {t}")
        print(f"{'=' * 85}")
        print(f"{'Configuration':<30} | {'Warmup':>11} | {'Hot (Avg ± Std) / image':>24} | {'Avg Crit Pts':>16}")
        print("-" * 85)

        if test_cpu:
            # 1. CPU Single Thread
            torch.set_num_threads(1)
            benchmark_extraction(
                "CPU Seq - 1 Thread",
                compute_dmsc,
                img_cpu,
                t,
                num_runs=2,
                run_profiler=enable_profiler,
                block_size=128,
                trace_valleys=trace_valleys,
                trace_ridges=trace_ridges,
                trace_peaks=trace_peaks,
                trace_basins=trace_basins,
            )

            # 2. CPU Max Threads (Sequential)
            torch.set_num_threads(num_threads)
            benchmark_extraction(
                f"CPU Seq - {num_threads} Threads",
                compute_dmsc,
                img_cpu,
                t,
                num_runs=10,
                run_profiler=enable_profiler,
                block_size=128,
                trace_valleys=trace_valleys,
                trace_ridges=trace_ridges,
                trace_peaks=trace_peaks,
                trace_basins=trace_basins,
            )

            # 3. CPU Max Threads (Batched)
            if test_batches:
                benchmark_extraction(
                    f"CPU Batch - {num_threads} Threads",
                    compute_dmsc,
                    img_cpu,
                    t,
                    num_runs=10,
                    run_profiler=enable_profiler,
                    block_size=128,
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
            benchmark_extraction(
                f"GPU Seq - {device.type.upper()} Driver",
                compute_dmsc,
                img_gpu,
                t,
                num_runs=10,
                run_profiler=enable_profiler,
                block_size=128,
                trace_valleys=trace_valleys,
                trace_ridges=trace_ridges,
                trace_peaks=trace_peaks,
                trace_basins=trace_basins,
            )

            # 5. GPU (Batched)
            if test_batches:
                benchmark_extraction(
                    f"GPU Batch - {device.type.upper()} Driver",
                    compute_dmsc,
                    img_gpu,
                    t,
                    num_runs=10,
                    run_profiler=enable_profiler,
                    block_size=128,
                    is_batched=True,
                    trace_valleys=trace_valleys,
                    trace_ridges=trace_ridges,
                    trace_peaks=trace_peaks,
                    trace_basins=trace_basins,
                )


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
    args = parser.parse_args()

    run_all_benchmarks(
        enable_profiler=args.profile,
        trace_valleys=args.trace_valleys,
        trace_ridges=args.trace_ridges,
        trace_peaks=args.trace_peaks,
        trace_basins=args.trace_basins,
    )
    if args.profile:
        print("Visit http://ui.perfetto.dev to check your profiling results.")
