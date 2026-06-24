import time
from dataclasses import astuple, dataclass, fields

import torch

from .csrc import dmsc_cpu, dmsc_gpu


@dataclass
class MSComplex:
    """
    The extracted Morse-Smale Complex containing critical points and manifolds.

    Grid Mapping Convention based on `is_dual`:
    - Primal (False): Maxima = Faces (H+1, W+1), Minima = Vertices (H, W)
    - Dual (True): Maxima = Vertices (H, W), Minima = Faces (H+1, W+1)

    All geometric attributes (points, ridges, valleys) are returned as flat 1D tensors
    containing raw cell_ids. Use `to_coordinates_yx()` to decode them.

    Attributes:
        shape (torch.Tensor): Tensor [H, W] containing the original image dimensions.

        max_pts (torch.Tensor): 1D Tensor of raw cell_ids for all surviving Maxima.
        min_pts (torch.Tensor): 1D Tensor of raw cell_ids for all surviving Minima.
        sad_pts (torch.Tensor): 1D Tensor of raw cell_ids for all surviving Saddle points.

        e_max (torch.Tensor): Tensor (E, 2) of shape [saddle_idx, max_idx] representing the max/saddle connectivity.
        e_min (torch.Tensor): Tensor (E, 2) of shape [saddle_idx, min_idx] representing the min/saddle connectivity.
        p_max (torch.Tensor): Tensor (N,) containing the persistence value of each maximum.
        p_min (torch.Tensor): Tensor (N,) containing the persistence value of each minimum.
        grad_indices (torch.Tensor): Tensor (H*W*4) containing the raw combinatorial gradient pairings.

        peaks (torch.Tensor): Tensor mapping each element to its local ascending extremum (Maxima basin).
            -> Primal: Maps to the (H+1)x(W+1) corner grid.
            -> Dual: Maps to the HxW pixel grid.
        ridges (torch.Tensor): Flat 1D tensor of raw cell_ids forming ascending manifolds.
        ridge_offsets (torch.Tensor): CSR offsets to slice `ridges` for each saddle.
        ridge_arc_offsets (torch.Tensor): The absolute index separating arc 1 and arc 2 within `ridges`.

        basins (torch.Tensor): Tensor mapping each element to its local descending extremum (Minima basin).
            -> Primal: Maps to the HxW pixel grid.
            -> Dual: Maps to the (H+1)x(W+1) corner grid.
        valleys (torch.Tensor): Flat 1D tensor of raw cell_ids forming descending manifolds (trenches).
        valley_offsets (torch.Tensor): CSR offsets to slice `valleys` for each saddle.
        valley_arc_offsets (torch.Tensor): The absolute index separating arc 1 and arc 2 within `valleys`.
    """

    shape: torch.Tensor

    max_pts: torch.Tensor
    min_pts: torch.Tensor
    sad_pts: torch.Tensor
    e_max: torch.Tensor
    e_min: torch.Tensor
    p_max: torch.Tensor
    p_min: torch.Tensor
    ppairs_max: torch.Tensor
    ppairs_min: torch.Tensor
    grad_indices: torch.Tensor

    peaks: torch.Tensor
    ridges: torch.Tensor
    ridge_offsets: torch.Tensor
    ridge_arc_offsets: torch.Tensor

    basins: torch.Tensor
    valleys: torch.Tensor
    valley_offsets: torch.Tensor
    valley_arc_offsets: torch.Tensor

    def __str__(self):
        output = "MSComplex("
        for field in fields(self):
            name = field.name
            value = getattr(self, name)

            if hasattr(value, "shape"):
                output += f" {name}:{tuple(value.shape)},"
        output += ")"
        return output

    @property
    def Nx(self) -> int:
        return int(self.shape[1].item()) + 1

    def to_coordinates_yx(self, cell_tensor: torch.Tensor, staggered: bool = True) -> torch.Tensor:
        """
        Decodes a 1D tensor of cell_ids into a 2D tensor of [y, x] coordinates.

        Args:
            cell_tensor: 1D tensor of raw cell_ids.
            staggered: If False, returns the base integer grid coordinates.
                       If True, applies sub-pixel (-0.5) offsets corresponding to
                       the exact physical centers of edges and corners for plotting.

        Returns:
            Tensor of shape (N, 2). dtype is int32 if staggered=False, float32 if staggered=True.
        """
        if cell_tensor.numel() == 0:
            dtype = torch.float32 if staggered else torch.int32
            return torch.empty((0, 2), dtype=dtype, device=cell_tensor.device)

        c_type = cell_tensor % 4
        idx = cell_tensor // 4

        y = idx // self.Nx
        x = idx % self.Nx

        if not staggered:
            return torch.stack([y, x], dim=-1).to(torch.int32)

        y = y.float()
        x = x.float()
        y -= torch.where((c_type == 2) | (c_type == 3), 0.5, 0.0)
        x -= torch.where((c_type == 1) | (c_type == 3), 0.5, 0.0)

        return torch.stack([y, x], dim=-1)

    @property
    def offset_max_pts_yx(self):
        return self.to_coordinates_yx(self.max_pts)

    @property
    def offset_min_pts_yx(self):
        return self.to_coordinates_yx(self.min_pts)

    @property
    def offset_sad_pts_yx(self):
        return self.to_coordinates_yx(self.sad_pts)

    def get_ridge(self, saddle_idx: int, split_arcs=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        start = self.ridge_offsets[saddle_idx]
        end = self.ridge_offsets[saddle_idx + 1] if saddle_idx + 1 < len(self.ridge_offsets) else len(self.ridges)
        pts = self.ridges[start:end]

        if split_arcs:
            split = self.ridge_arc_offsets[saddle_idx]
            arc1 = pts[: split - start]
            arc2 = pts[split - start : end - start]
            return arc1, arc2
        else:
            return pts

    def get_valley(self, saddle_idx: int, split_arcs=False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        start = self.valley_offsets[saddle_idx]
        end = self.valley_offsets[saddle_idx + 1] if saddle_idx + 1 < len(self.valley_offsets) else len(self.valleys)
        pts = self.valleys[start:end]

        if split_arcs:
            split = self.valley_arc_offsets[saddle_idx]
            arc1 = pts[: split - start]
            arc2 = pts[split - start : end - start]
            return arc1, arc2
        else:
            return pts

    def __iter__(self):
        return iter(astuple(self))

    def plot(self, img, ms_other=None, name="", name_other="Other", title=None, filename=None):
        """Generates a complete dashboard of the MS complex. If ms_other is provided, it plots a side-by-side comparison."""
        from .plots import create_dashboard

        return create_dashboard(
            img, self, ms_other=ms_other, name=name, name_other=name_other, title=title, filename=filename
        )

    def plot_gradient(self, ax, img, plot_bg=True, title="Raw Discrete Gradient Vector Field"):
        from .plots import plot_discrete_gradient

        return plot_discrete_gradient(self, ax, img, plot_bg=plot_bg, title=title)

    def plot_complex(
        self, ax, img, title="MS Complex", region_type=None, plot_boundaries=True, plot_edges=None, plot_pairs=False
    ):
        from .plots import plot_complex_layer

        return plot_complex_layer(
            self,
            ax,
            img,
            title=title,
            region_type=region_type,
            plot_boundaries=plot_boundaries,
            plot_edges=plot_edges,
            plot_pairs=plot_pairs,
        )

    def plot_barcode(self, ax, ms_other=None, name="", name_other="Other", title="Persistence Barcode"):
        from .plots import plot_barcode

        plot_barcode(self, ax, ms_other=ms_other, name=name, name_other=name_other, title=title)


def extract_full_complex(
    img,
    persistence_threshold,
    return_gradient,
    is_dual,
    trace_max_arcs,
    trace_min_arcs,
    trace_max_groups,
    trace_min_groups,
):
    if img.device.type == "cuda" or img.device.type == "mps":
        return dmsc_gpu.extract_dmsc(
            img,
            persistence_threshold,
            return_gradient,
            is_dual,
            trace_max_arcs,
            trace_min_arcs,
            trace_max_groups,
            trace_min_groups,
        )
    else:
        return dmsc_cpu.extract_dmsc(
            img,
            persistence_threshold,
            return_gradient,
            is_dual,
            trace_max_arcs,
            trace_min_arcs,
            trace_max_groups,
            trace_min_groups,
        )


def compute_dmsc(
    img,
    persistence_threshold,
    return_gradient=True,
    is_dual=False,
    trace_valleys=True,
    trace_ridges=True,
    trace_peaks=True,
    trace_basins=True,
    verbose=False,
) -> MSComplex | list[MSComplex]:
    """
    Extracts the discrete Morse-Smale complex using CPU/GPU multithreading.
    Returns a single MSComplex if img is 2D, or a list of MSComplex objects if img is 3D (batched).
    """
    if img.dim() not in (2, 3):
        raise ValueError(f"Input tensor must be 2D [H, W] or 3D [B, H, W]. Got {img.dim()}D.")

    start_time = time.perf_counter()
    # The C++ unified API automatically handles both single and batched layouts
    output = extract_full_complex(
        img,
        persistence_threshold,
        return_gradient,
        is_dual,
        trace_max_arcs=trace_ridges,
        trace_min_arcs=trace_valleys,
        trace_max_groups=trace_peaks,
        trace_min_groups=trace_basins,
    )

    end_time = time.perf_counter()
    if verbose:
        print(f"Time for MS-complex computation: {(end_time - start_time) * 1000:.3f} ms")

    if img.dim() == 2:
        return MSComplex(*output)
    else:
        return [MSComplex(*res) for res in output]
