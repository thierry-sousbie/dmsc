import argparse
import multiprocessing
import os
import time

import numpy as np
import torch

from dmsc import compute_dmsc, generate_noisy_landscape


def run_evaluation(img, extraction_fn, suffix, p_threshold, no_plots=False, seed=None, **kwargs):
    """Runs the entire pipeline for a given extraction function and saves the plots."""
    print(f"\n--- RUNNING EVALUATION: {suffix.upper()} ---")

    title = f"Random Seed: {seed}" if seed is not None else None

    print("Testing Primal Execution Mode (is_dual=False)...")
    ms_raw = extraction_fn(img, -1.0, return_gradient=True, **kwargs)
    ms_flt = extraction_fn(img, p_threshold, **kwargs)

    if not no_plots:
        ms_raw.plot(
            img.cpu(),
            ms_other=ms_flt,
            name="Raw",
            name_other="Filtered",
            title=title,
            filename=f"visualizations/dmsc_dashboard_primal_{suffix}.png",
        )

    print("Testing Dual Execution Mode (is_dual=True)...")
    ms_raw_min = extraction_fn(img, -1.0, return_gradient=True, is_dual=True, **kwargs)
    ms_flt_min = extraction_fn(img, p_threshold, return_gradient=False, is_dual=True, **kwargs)

    if not no_plots:
        ms_raw_min.plot(
            img.cpu(),
            ms_other=ms_flt_min,
            name="Raw",
            name_other="Filtered",
            title=title,
            filename=f"visualizations/dmsc_dashboard_dual_{suffix}.png",
        )


def test_dmsc(
    no_plots=False,
    with_loop=True,
    trace_valleys=True,
    trace_ridges=True,
    trace_peaks=True,
    trace_basins=True,
    seed=None,
    resolution=32,
    noise_scale=0.1,
):
    print(f"Default Threads: {torch.get_num_threads()}")

    print("Generating noisy landscape...")
    p_threshold = noise_scale
    img = generate_noisy_landscape(resolution, resolution, with_loop=with_loop, z_scale=1.0, noise_scale=noise_scale)

    # Run Multi-Threaded Evaluation (1 core)
    # print("\nRunning Multi-Threaded (1 core)...")
    torch.set_num_threads(1)
    run_evaluation(
        img,
        compute_dmsc,
        "st",
        no_plots=no_plots,
        trace_valleys=trace_valleys,
        trace_ridges=trace_ridges,
        trace_peaks=trace_peaks,
        trace_basins=trace_basins,
        seed=seed,
        p_threshold=p_threshold,
    )

    max_threads = multiprocessing.cpu_count()
    # print(f"\nRunning Multi-Threaded ({max_threads} cores)...")
    torch.set_num_threads(max_threads)
    run_evaluation(
        img,
        compute_dmsc,
        "mt",
        no_plots=no_plots,
        trace_valleys=trace_valleys,
        trace_ridges=trace_ridges,
        trace_peaks=trace_peaks,
        trace_basins=trace_basins,
        seed=seed,
        p_threshold=p_threshold,
    )

    max_threads = multiprocessing.cpu_count()
    # print(f"\nRunning GPU Multi-Threaded({max_threads} cores)...")
    torch.set_num_threads(max_threads)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # print(f"Using device: {device}")

    img = img.to(device)

    run_evaluation(
        img,
        compute_dmsc,
        "gpu",
        no_plots=no_plots,
        trace_valleys=trace_valleys,
        trace_ridges=trace_ridges,
        trace_peaks=trace_peaks,
        trace_basins=trace_basins,
        seed=seed,
        p_threshold=p_threshold,
    )


if __name__ == "__main__":
    # Setup argparse for command line seed input
    parser = argparse.ArgumentParser(description="Test DMSC")
    parser.add_argument("--seed", type=int, default=1780627675, help="Random seed for generating the noisy landscape.")
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
    parser.add_argument("--resolution", type=int, default=32, help="Image resolution")
    parser.add_argument("--noise-scale", type=float, default=0.1, help="Amount of noise")
    args = parser.parse_args()

    # Determine seed: use provided or generate a unique one based on time
    seed = args.seed if args.seed is not None else int(time.time())
    print(f"Using random seed: {seed}")

    # Enforce reproducibility across torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs("visualizations", exist_ok=True)
    test_dmsc(
        seed=seed,
        trace_valleys=args.trace_valleys,
        trace_ridges=args.trace_ridges,
        trace_peaks=args.trace_peaks,
        trace_basins=args.trace_basins,
        resolution=args.resolution,
        noise_scale=args.noise_scale,
    )
