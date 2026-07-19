import pytest
import torch

from dmsc import compute_dmsc


def assert_complex_equal(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.device == expected_tensor.device
        actual_cpu = actual_tensor.cpu()
        expected_cpu = expected_tensor.cpu()
        if actual_cpu.is_floating_point():
            torch.testing.assert_close(actual_cpu, expected_cpu, rtol=0, atol=0, equal_nan=True)
        else:
            assert torch.equal(actual_cpu, expected_cpu)


def assert_partition_equal(actual, expected):
    actual = actual.cpu().flatten()
    expected = expected.cpu().flatten()
    assert torch.equal(actual == -1, expected == -1)

    valid = expected != -1
    for label in torch.unique(expected[valid]):
        assert torch.unique(actual[expected == label]).numel() == 1
    for label in torch.unique(actual[valid]):
        assert torch.unique(expected[actual == label]).numel() == 1


def canonical_edges(complex_data, edge_name, extremum_name):
    edges = getattr(complex_data, edge_name).cpu()
    saddles = complex_data.sad_pts.cpu()
    extrema = getattr(complex_data, extremum_name).cpu()
    return {(int(saddles[saddle]), int(extrema[extremum])) for saddle, extremum in edges}


def assert_complex_equivalent(actual, expected):
    for name in ("max_pts", "min_pts", "sad_pts"):
        assert set(getattr(actual, name).cpu().tolist()) == set(getattr(expected, name).cpu().tolist())

    assert canonical_edges(actual, "e_max", "max_pts") == canonical_edges(expected, "e_max", "max_pts")
    assert canonical_edges(actual, "e_min", "min_pts") == canonical_edges(expected, "e_min", "min_pts")
    torch.testing.assert_close(actual.p_max.cpu().sort().values, expected.p_max.cpu().sort().values)
    torch.testing.assert_close(actual.p_min.cpu().sort().values, expected.p_min.cpu().sort().values)
    assert_partition_equal(actual.peaks, expected.peaks)
    assert_partition_equal(actual.basins, expected.basins)


@pytest.mark.parametrize("shape", [(0, 4), (4, 0), (0, 4, 4)])
def test_rejects_empty_dimensions(shape):
    with pytest.raises(ValueError, match="non-empty"):
        compute_dmsc(torch.empty(shape), 0.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32, torch.int64])
def test_rejects_unsupported_dtype(dtype):
    with pytest.raises(TypeError, match="float32"):
        compute_dmsc(torch.zeros((4, 4), dtype=dtype), 0.0)


def test_accepts_noncontiguous_input():
    field = torch.randn((7, 11), generator=torch.Generator().manual_seed(4)).transpose(0, 1)
    assert not field.is_contiguous()

    actual = compute_dmsc(field, 0.1)
    expected = compute_dmsc(field.contiguous(), 0.1)

    assert_complex_equal(actual, expected)


def test_batch_matches_sequential_calls():
    generator = torch.Generator().manual_seed(8)
    fields = torch.randn((3, 9, 7), generator=generator)

    batched = compute_dmsc(fields, 0.15)
    sequential = [compute_dmsc(field, 0.15) for field in fields]

    assert len(batched) == len(sequential)
    for actual, expected in zip(batched, sequential, strict=True):
        assert_complex_equal(actual, expected)


@pytest.mark.parametrize("value", [0.0, float("inf"), float("-inf"), float("nan")])
def test_tied_and_nonfinite_fields_are_deterministic(value):
    field = torch.full((6, 5), value)

    first = compute_dmsc(field, 0.0)
    second = compute_dmsc(field, 0.0)

    assert_complex_equal(first, second)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_batch_matches_sequential_calls():
    generator = torch.Generator().manual_seed(12)
    fields = torch.randn((3, 9, 7), generator=generator).to("mps")

    batched = compute_dmsc(fields, 0.15)
    sequential = [compute_dmsc(field, 0.15) for field in fields]

    assert len(batched) == len(sequential)
    for actual, expected in zip(batched, sequential, strict=True):
        assert_complex_equivalent(actual, expected)
