import os

import numpy as np
import pytest
import torch

from csrc.dmsc import compute_dmsc

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="module")
def landscape():
    path = os.path.join(DATA_DIR, "landscape_32x32.pt")
    if not os.path.exists(path):
        pytest.fail(f"Ground truth data not found at {path}. Run generate_ground_truth.py first.")
    return torch.load(path)


def assert_mscomplex_equivalence(
    gt_dict, computed_ms, trace_valleys=True, trace_ridges=True, trace_peaks=True, trace_basins=True
):
    """
    Asserts that the computed MSComplex is topologically and geometrically equivalent
    to the ground truth dictionary, ignoring arbitrary ordering of arrays.
    """
    # 1. Check basic sizes
    assert len(gt_dict["max_pts"]) == len(computed_ms.max_pts), "Number of maxima differ"
    assert len(gt_dict["min_pts"]) == len(computed_ms.min_pts), "Number of minima differ"
    assert len(gt_dict["sad_pts"]) == len(computed_ms.sad_pts), "Number of saddles differ"

    # 2. Check sets of critical points (unordered equivalence)
    assert set(gt_dict["max_pts"].cpu().numpy()) == set(computed_ms.max_pts.cpu().numpy()), "Set of maxima differ"
    assert set(gt_dict["min_pts"].cpu().numpy()) == set(computed_ms.min_pts.cpu().numpy()), "Set of minima differ"
    assert set(gt_dict["sad_pts"].cpu().numpy()) == set(computed_ms.sad_pts.cpu().numpy()), "Set of saddles differ"

    # 3. Check persistence values (sorted equivalence)
    gt_p_max = np.sort(gt_dict["p_max"].cpu().numpy())
    comp_p_max = np.sort(computed_ms.p_max.cpu().numpy())
    assert np.allclose(gt_p_max, comp_p_max), "Persistence values for max-sad pairs differ"

    gt_p_min = np.sort(gt_dict["p_min"].cpu().numpy())
    comp_p_min = np.sort(computed_ms.p_min.cpu().numpy())
    assert np.allclose(gt_p_min, comp_p_min), "Persistence values for min-sad pairs differ"

    # 4. Check array lengths for geometries
    assert len(gt_dict["e_max"]) == len(computed_ms.e_max), "Number of max edges differ"
    assert len(gt_dict["e_min"]) == len(computed_ms.e_min), "Number of min edges differ"

    # we need to strip duplicate because our ground truth had a small bug
    # TODO: -> regenerate the references (but be absolutely sure it s correct beforehand)
    def strip_duplicates(arr):
        if len(arr) == 0:
            return arr
        mask = np.ones(len(arr), dtype=bool)
        mask[1:] = arr[1:] != arr[:-1]
        return arr[mask]

    if trace_ridges:
        gt_r = strip_duplicates(gt_dict["ridges"].cpu().numpy())
        cp_r = strip_duplicates(computed_ms.ridges.cpu().numpy())
        assert len(gt_r) == len(cp_r), "Total ridge length differ"
        assert len(gt_dict["ridge_offsets"]) == len(computed_ms.ridge_offsets), "Number of ridge offsets differ"
    else:
        assert len(computed_ms.ridges) == 0, "Expected ridges to be empty"
        assert len(computed_ms.ridge_offsets) == 0, "Expected ridge_offsets to be empty"

    if trace_valleys:
        gt_v = strip_duplicates(gt_dict["valleys"].cpu().numpy())
        cp_v = strip_duplicates(computed_ms.valleys.cpu().numpy())
        assert len(gt_v) == len(cp_v), "Total valley length differ"
        assert len(gt_dict["valley_offsets"]) == len(computed_ms.valley_offsets), "Number of valley offsets differ"
    else:
        assert len(computed_ms.valleys) == 0, "Expected valleys to be empty"
        assert len(computed_ms.valley_offsets) == 0, "Expected valley_offsets to be empty"

    # 5. Check region equivalence (Peaks and Basins)
    def check_regions(gt_regions, comp_regions, name="Regions"):
        if gt_regions is None or gt_regions.numel() == 0:
            assert comp_regions is None or comp_regions.numel() == 0
            return

        gt_arr = gt_regions.cpu().numpy().flatten()
        comp_arr = comp_regions.cpu().numpy().flatten()
        assert len(gt_arr) == len(comp_arr), f"{name} shape mismatch"

        # Check mask of unassigned (-1) regions matches
        gt_valid = gt_arr != -1
        comp_valid = comp_arr != -1
        assert np.array_equal(gt_valid, comp_valid), f"{name} valid mask mismatch"

        if not np.any(gt_valid):
            return

        # Check 1-to-1 mapping of region labels
        gt_labels = np.unique(gt_arr[gt_valid])
        for lbl in gt_labels:
            mask = gt_arr == lbl
            comp_lbls = np.unique(comp_arr[mask])
            assert len(comp_lbls) == 1, f"GT {name} label {lbl} is split into Computed labels {comp_lbls}"

        comp_labels = np.unique(comp_arr[comp_valid])
        for lbl in comp_labels:
            mask = comp_arr == lbl
            gt_lbls = np.unique(gt_arr[mask])
            assert len(gt_lbls) == 1, f"Computed {name} label {lbl} is split into GT labels {gt_lbls}"

    if trace_peaks:
        check_regions(gt_dict["peaks"], computed_ms.peaks, "Peaks")
    else:
        assert computed_ms.peaks is None or len(computed_ms.peaks) == 0, "Expected peaks to be empty"

    if trace_basins:
        check_regions(gt_dict["basins"], computed_ms.basins, "Basins")
    else:
        assert computed_ms.basins is None or len(computed_ms.basins) == 0, "Expected basins to be empty"


@pytest.mark.parametrize(
    "mode, threads, device_str",
    [("st", 1, "cpu"), ("mt", 8, "cpu"), ("gpu", 8, "mps" if torch.backends.mps.is_available() else "cuda")],
)
@pytest.mark.parametrize("is_dual", [False, True])
@pytest.mark.parametrize("is_filtered, threshold", [(False, -1.0), (True, 0.15)])
@pytest.mark.parametrize("trace_valleys", [True, False])
@pytest.mark.parametrize("trace_ridges", [True, False])
@pytest.mark.parametrize("trace_peaks", [True, False])
@pytest.mark.parametrize("trace_basins", [True, False])
def test_dmsc_regression(
    landscape,
    mode,
    threads,
    device_str,
    is_dual,
    is_filtered,
    threshold,
    trace_valleys,
    trace_ridges,
    trace_peaks,
    trace_basins,
):
    device = torch.device(device_str)
    # The extension will throw an error if the GPU device is missing on this platform
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    img = landscape.to(device)
    torch.set_num_threads(threads)

    dual_str = "dual" if is_dual else "primal"
    flt_str = "flt" if is_filtered else "raw"
    gt_filename = f"ms_{mode}_{dual_str}_{flt_str}.pt"
    gt_path = os.path.join(DATA_DIR, gt_filename)

    if not os.path.exists(gt_path):
        pytest.fail(f"Ground truth {gt_path} not found.")

    gt_dict = torch.load(gt_path, weights_only=False)

    # Compute MS complex
    ms = compute_dmsc(
        img,
        threshold,
        return_gradient=not is_filtered,
        is_dual=is_dual,
        block_size=32,
        trace_valleys=trace_valleys,
        trace_ridges=trace_ridges,
        trace_peaks=trace_peaks,
        trace_basins=trace_basins,
    )

    # Validate Equivalence
    assert_mscomplex_equivalence(gt_dict, ms, trace_valleys, trace_ridges, trace_peaks, trace_basins)
