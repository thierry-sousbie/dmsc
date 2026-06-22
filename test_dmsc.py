import argparse
import multiprocessing
import os
import time

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # Use Agg for non-interactive plotting
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt

from csrc.dmsc import compute_dmsc


def generate_noisy_landscape(H=20, W=20, with_loop=False):
    """Generates a landscape with noise to test persistence simplification."""
    y, x = torch.meshgrid(torch.linspace(-2, 2, H), torch.linspace(-2, 2, W), indexing="ij")

    peak1 = torch.exp(-((x - 1) ** 2 + (y) ** 2) / 0.5)
    peak2 = torch.exp(-((x + 1) ** 2 + (y) ** 2) / 0.5)
    valley1 = -0.5 * torch.exp(-((x) ** 2 + (y - 1) ** 2) / 0.5)
    valley2 = -1.0 * torch.exp(-((x) ** 2 + (y + 1) ** 2) / 0.5)

    z = peak1 + peak2 + valley1 + valley2
    if with_loop:
        peak3 = 1.0 * torch.exp(-(((x) ** 2 + (y + 1) ** 2) ** 3) / 0.5)
        slope = -y / 4
        z += peak3 + slope

    noise = (torch.rand(H, W) - 0.5) * 0.1
    z = z * 2 + noise
    return z.to(torch.float32)


def plot_discrete_gradient(ax, img, ms_complex, H, W, plot_bg=True, title="Raw Discrete Gradient Vector Field"):
    """Plots the raw dmsc gradient vector field (Vectorized for high performance)."""
    if plot_bg:
        ax.imshow(img.cpu().numpy(), cmap="viridis", origin="lower", alpha=0.6, zorder=0)

    grad_numpy = ms_complex.grad_indices.cpu().numpy()

    # Vectorized extraction to avoid massive Python for-loop overhead
    valid_mask = grad_numpy != -1
    src_ids = np.arange(len(grad_numpy))[valid_mask]
    dst_ids = grad_numpy[valid_mask]

    def get_dim(c_types):
        return np.where(c_types == 0, 0, np.where((c_types == 1) | (c_types == 2), 1, 2))

    src_dims = get_dim(src_ids % 4)
    dst_dims = get_dim(dst_ids % 4)

    # Only draw arrows from higher dimension to lower dimension
    arrow_mask = src_dims > dst_dims
    src_ids = src_ids[arrow_mask]
    dst_ids = dst_ids[arrow_mask]

    # Decode coordinates correctly based on cell type using the unified method
    src_coords = ms_complex.to_coordinates_yx(torch.from_numpy(src_ids), staggered=True).cpu().numpy()
    dst_coords = ms_complex.to_coordinates_yx(torch.from_numpy(dst_ids), staggered=True).cpu().numpy()

    y_i, x_i = src_coords[:, 0], src_coords[:, 1]
    y_t, x_t = dst_coords[:, 0], dst_coords[:, 1]

    bounds_mask = (
        (y_i >= -1) & (y_i <= H) & (x_i >= -1) & (x_i <= W) & (y_t >= -1) & (y_t <= H) & (x_t >= -1) & (x_t <= W)
    )

    if np.any(bounds_mask):
        ax.quiver(
            x_i[bounds_mask],
            y_i[bounds_mask],
            x_t[bounds_mask] - x_i[bounds_mask],
            y_t[bounds_mask] - y_i[bounds_mask],
            scale_units="xy",
            angles="xy",
            scale=1,
            color="white",
            width=0.003,
            headwidth=4,
            headlength=5,
            alpha=0.85,
            zorder=10,
        )

    max_pts = ms_complex.offset_max_pts_yx.cpu().numpy()
    min_pts = ms_complex.offset_min_pts_yx.cpu().numpy()
    sad_pts = ms_complex.offset_sad_pts_yx.cpu().numpy()

    # 4. Plot Critical Nodes
    if len(max_pts) > 0:
        ax.scatter(
            max_pts[:, 1], max_pts[:, 0], c="red", marker="^", edgecolors="black", s=60, label="Maxima", zorder=5
        )

    if len(min_pts) > 0:
        ax.scatter(
            min_pts[:, 1], min_pts[:, 0], c="blue", marker="v", edgecolors="black", s=60, label="Minima", zorder=5
        )

    if len(sad_pts) > 0:
        ax.scatter(
            sad_pts[:, 1], sad_pts[:, 0], c="cyan", marker="s", edgecolors="black", s=40, label="Saddles", zorder=5
        )

    ax.set_title(title)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)


def plot_complex_layer(
    img, ms_complex, ax, title, region_type=None, plot_boundaries=True, plot_edges=None, plot_pairs=False
):
    """Plots the simplified structural topological complex maps from an MSComplex object.
    region_type can be None, "peaks", or "basins".
    """
    plot_regions = region_type is not None
    if plot_edges is None:
        plot_edges = not plot_regions

    regions = None
    if region_type == "peaks":
        regions = (
            ms_complex.peaks.cpu().numpy() if ms_complex.peaks is not None and ms_complex.peaks.numel() > 0 else None
        )
    elif region_type == "basins":
        regions = (
            ms_complex.basins.cpu().numpy() if ms_complex.basins is not None and ms_complex.basins.numel() > 0 else None
        )

    # Use to_coordinates_yx to decode the raw 1D arrays into displayable 2D points
    max_pts = ms_complex.to_coordinates_yx(ms_complex.max_pts, staggered=True).cpu().numpy()
    min_pts = ms_complex.to_coordinates_yx(ms_complex.min_pts, staggered=True).cpu().numpy()
    sad_pts = ms_complex.to_coordinates_yx(ms_complex.sad_pts, staggered=True).cpu().numpy()

    H, W = img.shape
    if regions is not None:
        rH, rW = regions.shape
        if rH == H and rW == W:
            my_extent = [-0.5, W - 0.5, -0.5, H - 0.5]
            off_x, off_y = 0.5, 0.5
        else:
            my_extent = [-1.0, W, -1.0, H]
            off_x, off_y = 0.0, 0.0
    else:
        my_extent = [-0.5, W - 0.5, -0.5, H - 0.5]
        off_x, off_y = 0.5, 0.5

    # Plot Background (Regions or Raw Image)
    if plot_regions:
        if regions is not None:
            unique_vals, ids_map = np.unique(regions, return_inverse=True)
            shuffled_vals = np.random.permutation(unique_vals)
            colored_regions = shuffled_vals[ids_map].reshape(regions.shape)
            masked_regions = np.ma.masked_where(regions == -1, colored_regions)
            # num_missing = np.sum(regions == -1)
            # if num_missing > 0:
            #     print(f"Number of unassigned (-1) pixels in dual regions: {num_missing}")
            ax.imshow(masked_regions, cmap="tab20", origin="lower", extent=my_extent, interpolation="nearest", zorder=1)
        else:
            ax.set_facecolor("white")
    else:
        ax.imshow(img.cpu().numpy(), cmap="viridis", origin="lower", extent=[-0.5, W - 0.5, -0.5, H - 0.5], zorder=0)

    # Plot Region Boundaries (Watershed lines)
    if plot_boundaries and regions is not None:
        segs = []
        diff_x = regions[:, :-1] != regions[:, 1:]
        y_idx_x, x_idx_x = np.where(diff_x)
        if len(y_idx_x) > 0:
            segs_x = np.empty((len(y_idx_x), 2, 2))
            segs_x[:, 0, 0] = x_idx_x + off_x
            segs_x[:, 0, 1] = y_idx_x + off_y - 1.0
            segs_x[:, 1, 0] = x_idx_x + off_x
            segs_x[:, 1, 1] = y_idx_x + off_y
            segs.append(segs_x)

        diff_y = regions[:-1, :] != regions[1:, :]
        y_idx_y, x_idx_y = np.where(diff_y)
        if len(y_idx_y) > 0:
            segs_y = np.empty((len(y_idx_y), 2, 2))
            segs_y[:, 0, 0] = x_idx_y + off_x - 1.0
            segs_y[:, 0, 1] = y_idx_y + off_y
            segs_y[:, 1, 0] = x_idx_y + off_x
            segs_y[:, 1, 1] = y_idx_y + off_y
            segs.append(segs_y)

        if segs:
            all_segs = np.vstack(segs)
            boundary_color = "black" if region_type == "peaks" else "white"
            boundary_style = "solid" if region_type == "peaks" else "dashed"
            lc = mcoll.LineCollection(
                all_segs, colors=boundary_color, linestyles=boundary_style, linewidths=1.2, alpha=0.9, zorder=2
            )
            ax.add_collection(lc)

    # Plot Topological Edges (Manifolds or Straight Lines)
    if plot_edges:

        def add_edges(saddles, targets, edges_indices, color, style):
            if edges_indices is None or len(edges_indices) == 0:
                return
            segs = []
            for s_idx, t_idx in edges_indices.cpu().numpy():
                sy, sx = saddles[s_idx]
                ty, tx = targets[t_idx]
                segs.append([[sx, sy], [tx, ty]])
            lc = mcoll.LineCollection(segs, colors=color, linestyles=style, linewidths=1.5, alpha=0.7, zorder=3)
            ax.add_collection(lc)

        if len(ms_complex.ridges) > 0:
            for i in range(len(sad_pts)):
                arc1, arc2 = ms_complex.get_ridge(i, split_arcs=True)
                for arc in (arc1, arc2):
                    if len(arc) > 0:
                        coords = ms_complex.to_coordinates_yx(arc, staggered=True).cpu().numpy()
                        ax.plot(
                            coords[:, 1],
                            coords[:, 0],
                            color="red",
                            linestyle="solid",
                            linewidth=1.5,
                            alpha=0.7,
                            zorder=3,
                        )
        else:
            add_edges(sad_pts, max_pts, ms_complex.e_max, color="red", style="solid")

        if len(ms_complex.valleys) > 0:
            for i in range(len(sad_pts)):
                arc1, arc2 = ms_complex.get_valley(i, split_arcs=True)
                for arc in (arc1, arc2):
                    if len(arc) > 0:
                        coords = ms_complex.to_coordinates_yx(arc, staggered=True).cpu().numpy()
                        ax.plot(
                            coords[:, 1],
                            coords[:, 0],
                            color="blue",
                            linestyle="dashed",
                            linewidth=1.5,
                            alpha=0.7,
                            zorder=3,
                        )
        else:
            add_edges(sad_pts, min_pts, ms_complex.e_min, color="blue", style="dashed")

    # Plot Persistence Pair Connections
    if plot_pairs:

        def add_pair_lines(saddle_pts, extrema_pts, paired_extrema_indices, color_fg, label):
            if paired_extrema_indices is None or len(paired_extrema_indices) == 0:
                return

            valid_mask = paired_extrema_indices != -1
            if not torch.any(valid_mask):
                return

            valid_saddle_ids = saddle_pts[valid_mask]
            valid_extrema_indices = paired_extrema_indices[valid_mask].long()
            valid_extrema_ids = extrema_pts[valid_extrema_indices]

            ex_coords = ms_complex.to_coordinates_yx(valid_extrema_ids, staggered=True).cpu().numpy()
            sad_coords = ms_complex.to_coordinates_yx(valid_saddle_ids, staggered=True).cpu().numpy()

            segs = []
            for i in range(len(ex_coords)):
                ey, ex = ex_coords[i]
                sy, sx = sad_coords[i]
                segs.append([[ex, ey], [sx, sy]])

            lc_bg = mcoll.LineCollection(segs, colors="white", linestyles="solid", linewidths=1.5, alpha=0.9, zorder=6)
            lc_fg = mcoll.LineCollection(segs, colors=color_fg, linestyles="solid", linewidths=1.0, alpha=1.0, zorder=7)
            ax.add_collection(lc_bg)
            ax.add_collection(lc_fg)
            ax.plot([], [], color=color_fg, linewidth=2, label=label)

        add_pair_lines(
            ms_complex.sad_pts,
            ms_complex.max_pts,
            getattr(ms_complex, "ppairs_max", None),
            "magenta",
            "Max-Saddle Pair",
        )
        add_pair_lines(
            ms_complex.sad_pts, ms_complex.min_pts, getattr(ms_complex, "ppairs_min", None), "lime", "Min-Saddle Pair"
        )

    # Plot Critical Nodes
    if len(max_pts) > 0:
        ax.scatter(
            max_pts[:, 1], max_pts[:, 0], c="red", marker="^", edgecolors="black", s=60, label="Maxima", zorder=8
        )
    if len(min_pts) > 0:
        ax.scatter(
            min_pts[:, 1], min_pts[:, 0], c="blue", marker="v", edgecolors="black", s=60, label="Minima", zorder=8
        )
    if len(sad_pts) > 0:
        ax.scatter(
            sad_pts[:, 1], sad_pts[:, 0], c="cyan", marker="s", edgecolors="black", s=40, label="Saddles", zorder=8
        )

    ax.set_title(title)
    if region_type is not None or plot_pairs:
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)


def plot_barcode(ax, ms_raw, ms_flt=None, title="Persistence Barcode"):
    """Plots the persistence barcode diagram showing feature lifetimes before and after simplification."""

    def extract_and_sort(ms_complex):
        if ms_complex is None:
            return np.array([]), np.array([])
        p_max = ms_complex.p_max.cpu().numpy()
        p_min = ms_complex.p_min.cpu().numpy()
        # Filter out near-zero or invalid persistences
        p_max = p_max[p_max > 1e-6]
        p_min = p_min[p_min > 1e-6]
        # Sort descending for a clean waterfall visualization
        return np.sort(p_max)[::-1], np.sort(p_min)[::-1]

    p_max_raw, p_min_raw = extract_and_sort(ms_raw)
    p_max_flt, p_min_flt = extract_and_sort(ms_flt)

    # Plot Maxima (Red)
    if len(p_max_raw) > 0:
        x_max = np.arange(len(p_max_raw))

        # Overlay preserved features as thick, dark lines
        if len(p_max_flt) > 0:
            x_max_flt = np.arange(len(p_max_flt))
            ax.vlines(x_max_flt, 0, p_max_flt, color="darkred", linewidth=3.0, alpha=1.0, label="Preserved Maxima")
        # Draw all raw features as thin, light lines
        ax.vlines(x_max, 0, p_max_raw, color="lightcoral", linewidth=1.5, alpha=0.7, label="Raw Maxima")

    else:
        x_max = []

    # Plot Minima (Blue)
    if len(p_min_raw) > 0:
        # Add a visual gap on the X-axis between the two classes
        offset = len(p_max_raw) + max(1, len(p_max_raw) // 10)
        x_min = np.arange(len(p_min_raw)) + offset
        # Overlay preserved features as thick, dark lines
        if len(p_min_flt) > 0:
            x_min_flt = np.arange(len(p_min_flt)) + offset
            ax.vlines(x_min_flt, 0, p_min_flt, color="darkblue", linewidth=3.0, alpha=1.0, label="Preserved Minima")
        # Draw all raw features as thin, light lines
        ax.vlines(x_min, 0, p_min_raw, color="lightskyblue", linewidth=1.5, alpha=0.7, label="Raw Minima")

    ax.set_title(title)
    ax.set_ylabel("Persistence (Lifetime)")
    ax.set_xlabel("Feature Index (Sorted)")

    if len(p_max_raw) > 0 or len(p_min_raw) > 0:
        ax.legend(loc="upper right", fontsize=8)


def create_dashboard(filename, img, H, W, ms_raw, ms_flt, seed=None):
    """Generates a strictly 3x3 plot dashboard displaying the entire extraction and simplification pipeline."""
    fig, axes = plt.subplots(3, 3, figsize=(24, 24))

    # Add the seed to the main title above the pictures if provided
    if seed is not None:
        fig.suptitle(f"Random Seed: {seed}", fontsize=26, fontweight="bold")

    # --- ROW 0: Density Views ---
    # 0,0: Discrete gradient overlayed on density (Raw)
    plot_discrete_gradient(axes[0, 0], img, ms_raw, H, W, plot_bg=True, title="Discrete Gradient on Density (Raw)")

    # 0,1: MS complex overlayed on density (Raw) -> NOW WITH PAIRS
    plot_complex_layer(
        img,
        ms_raw,
        axes[0, 1],
        "MS Complex on Density (Raw)",
        region_type=None,
        plot_edges=True,
        plot_boundaries=False,
        plot_pairs=True,
    )

    # 0,2: MS complex overlayed on density (Filtered) -> NOW WITH PAIRS
    plot_complex_layer(
        img,
        ms_flt,
        axes[0, 2],
        "MS Complex on Density (Filtered)",
        region_type=None,
        plot_edges=True,
        plot_boundaries=False,
        plot_pairs=True,
    )

    # --- ROW 1: Peak Views ---
    # 1,0: Keep this panel clean/empty to focus attention on the barcode below
    axes[1, 0].axis("off")

    # 1,1: MSComplex overlayed on peak regions (Raw)
    plot_complex_layer(
        img,
        ms_raw,
        axes[1, 1],
        "MS Complex overlayed on Peak Regions",
        region_type="peaks",
        plot_edges=True,
        plot_boundaries=False,
    )
    # 1,2: MSComplex overlayed on peak regions (Filtered)
    plot_complex_layer(
        img,
        ms_flt,
        axes[1, 2],
        "MS Complex overlayed on Peak Regions",
        region_type="peaks",
        plot_edges=True,
        plot_boundaries=False,
    )

    # --- ROW 2: Basin Views ---
    # 2,0: Barcode Diagram (Overlay Raw + Filtered)
    plot_barcode(axes[2, 0], ms_raw, ms_flt, title="Persistence Barcode")

    # 2,1: MSComplex overlayed on basins regions (Raw)
    plot_complex_layer(
        img,
        ms_raw,
        axes[2, 1],
        "MS Complex overlayed on Basin Regions",
        region_type="basins",
        plot_boundaries=False,
        plot_edges=True,
    )

    # 2,2: MSComplex overlayed on basins regions (Filtered)
    plot_complex_layer(
        img,
        ms_flt,
        axes[2, 2],
        "MS Complex overlayed on Basin Regions",
        region_type="basins",
        plot_boundaries=False,
        plot_edges=True,
    )

    plt.tight_layout()

    # Adjust layout to make room for the suptitle so it doesn't overlap
    if seed is not None:
        fig.subplots_adjust(top=0.95)

    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {filename}")


def run_evaluation(img, H, W, extraction_fn, suffix, no_plots=False, seed=None, **kwargs):
    """Runs the entire pipeline for a given extraction function and saves the plots."""
    print(f"\n--- RUNNING EVALUATION: {suffix.upper()} ---")

    print("Testing Primal Execution Mode (is_dual=False)...")
    ms_raw = extraction_fn(img, -1.0, return_gradient=True, **kwargs)
    # print(ms_raw)
    ms_flt = extraction_fn(img, 0.15, **kwargs)
    # print(ms_flt)

    if not no_plots:
        create_dashboard(
            f"visualizations/dmsc_dashboard_primal_{suffix}.png", img.cpu(), H, W, ms_raw, ms_flt, seed=seed
        )

    print("Testing Dual Execution Mode (is_dual=True)...")
    ms_raw_min = extraction_fn(img, -1.0, return_gradient=True, is_dual=True, **kwargs)
    ms_flt_min = extraction_fn(img, 0.15, return_gradient=False, is_dual=True, **kwargs)

    if not no_plots:
        create_dashboard(
            f"visualizations/dmsc_dashboard_dual_{suffix}.png", img.cpu(), H, W, ms_raw_min, ms_flt_min, seed=seed
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
):
    print(f"Default Threads: {torch.get_num_threads()}")

    H, W = (resolution, resolution)

    print("Generating noisy landscape...")
    img = generate_noisy_landscape(H, W, with_loop=with_loop)

    # Run Multi-Threaded Evaluation (1 core)
    # print("\nRunning Multi-Threaded (1 core)...")
    torch.set_num_threads(1)
    run_evaluation(
        img,
        H,
        W,
        compute_dmsc,
        "st",
        no_plots=no_plots,
        trace_valleys=trace_valleys,
        trace_ridges=trace_ridges,
        trace_peaks=trace_peaks,
        trace_basins=trace_basins,
        seed=seed,
    )

    max_threads = multiprocessing.cpu_count()
    # print(f"\nRunning Multi-Threaded ({max_threads} cores)...")
    torch.set_num_threads(max_threads)
    run_evaluation(
        img,
        H,
        W,
        compute_dmsc,
        "mt",
        no_plots=no_plots,
        trace_valleys=trace_valleys,
        trace_ridges=trace_ridges,
        trace_peaks=trace_peaks,
        trace_basins=trace_basins,
        seed=seed,
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
        H,
        W,
        compute_dmsc,
        "gpu",
        no_plots=no_plots,
        trace_valleys=trace_valleys,
        trace_ridges=trace_ridges,
        trace_peaks=trace_peaks,
        trace_basins=trace_basins,
        seed=seed,
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
    )
