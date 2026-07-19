#include <ATen/Parallel.h>
#include <pybind11/stl.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <array>
#include <future>
#include <memory>
#include <vector>

#include "./cpu/dmsc_impl.hxx"
#include "./dmsc_struct.hxx"
#include "./gpu/dmsc_impl.hxx"
#include "./input_validation.hxx"

using namespace gpu;

template <bool IS_DUAL, typename Workspace>
void prepare_single_dmsc_gpu(torch::Tensor scalar_field, bool trace_max_arcs, bool trace_min_arcs, Workspace& ws) {
  gpu::compute_gradient<IS_DUAL>(ws, scalar_field);
  gpu::trace_from_saddles<IS_DUAL>(ws);
  if (trace_max_arcs || trace_min_arcs) {
    gpu::trace_raw_arcs_geometry(ws, trace_max_arcs, trace_min_arcs);
  }
}

template <bool IS_DUAL, typename Workspace>
DMSComplex finalize_single_dmsc_gpu(bool return_gradient, bool trace_max_arcs, bool trace_min_arcs,
                                    bool trace_max_groups, bool trace_min_groups, Workspace& ws) {
  if (trace_max_arcs || trace_min_arcs) {
    gpu::simplify_arcs_geometry(ws, trace_max_arcs, trace_min_arcs);
  }
  if (trace_max_groups || trace_min_groups) {
    gpu::compute_cell_groups<IS_DUAL>(ws, trace_max_groups, trace_min_groups);
  }
  return ws.template complex<IS_DUAL>(return_gradient, trace_max_arcs, trace_min_arcs, trace_max_groups,
                                      trace_min_groups);
}

/*
Physical Geometry   cell_type Topology (IS_DUAL=false)  Topology (IS_DUAL=true)
Vertex (Pixel)      0         Minimum (Lowest scalar)   Maximum (Highest scalar)
Horizontal Edge     1         Saddle                    Saddle
Vertical Edge       2         Saddle                    Saddle
Face (Square Gap)   3         Maximum (Highest scalar)  Minimum (Lowest scalar)

cell values are computed as the maximum (resp. minimum) of its vertex values when IS_DUAL=false (resp. true)

When IS_DUAL is true, the value of a cell is given by that of its lowest vertex and we sort in opposite
direction. At the end, we swap minima and maxima, so we compute the dual complex with the role of
vertices/edges/faces reversed, i.e. they represent maxima/saddles/minima respectively.
*/
template <bool IS_DUAL = false>
DMSComplex extract_single_dmsc_gpu_t(torch::Tensor scalar_field, float persistence_threshold, bool return_gradient,
                                     bool trace_max_arcs, bool trace_min_arcs, bool trace_max_groups,
                                     bool trace_min_groups, gpu::Workspace& ws) {
  if (scalar_field.device().is_cuda()) {
#ifdef __APPLE__
    throw std::runtime_error("Extension was not compiled with CUDA support.");
#endif

  } else if (scalar_field.device().is_mps()) {
#ifndef __APPLE__
    throw std::runtime_error("Extension was not compiled with Metal support.");
#endif
  } else {
    throw std::runtime_error("Input tensor must be on a CUDA or MPS device.");
  }

  tbb::global_control control(tbb::global_control::max_allowed_parallelism, at::get_num_threads());
  if (IS_DUAL) {
    std::swap(trace_max_arcs, trace_min_arcs);
    std::swap(trace_max_groups, trace_min_groups);
  }

  prepare_single_dmsc_gpu<IS_DUAL>(scalar_field, trace_max_arcs, trace_min_arcs, ws);

  cpu::compute_ppairs_and_simplify<IS_DUAL>(ws, persistence_threshold, trace_max_arcs, trace_min_arcs);

  return finalize_single_dmsc_gpu<IS_DUAL>(return_gradient, trace_max_arcs, trace_min_arcs, trace_max_groups,
                                           trace_min_groups, ws);
}

template <bool IS_DUAL>
void extract_batched_dmsc_gpu(torch::Tensor scalar_field, float persistence_threshold, bool return_gradient,
                              bool trace_max_arcs, bool trace_min_arcs, bool trace_max_groups, bool trace_min_groups,
                              std::vector<DMSComplex>& results) {
  if (IS_DUAL) {
    std::swap(trace_max_arcs, trace_min_arcs);
    std::swap(trace_max_groups, trace_min_groups);
  }

  struct PipelineSlot {
    std::unique_ptr<gpu::Workspace> workspace;
    std::future<void> persistence;
    int image_index = -1;
  };

  int H = scalar_field.size(1);
  int W = scalar_field.size(2);
  std::array<PipelineSlot, 2> slots;
  for (auto& slot : slots) slot.workspace = std::make_unique<gpu::Workspace>(H, W);

  auto finalize_slot = [&](PipelineSlot& slot) {
    if (slot.image_index < 0) return;
    slot.persistence.get();
    results[slot.image_index] =
        finalize_single_dmsc_gpu<IS_DUAL>(return_gradient, trace_max_arcs, trace_min_arcs, trace_max_groups,
                                          trace_min_groups, *slot.workspace);
    slot.image_index = -1;
  };

  for (int b = 0; b < scalar_field.size(0); ++b) {
    PipelineSlot& slot = slots[b % slots.size()];
    finalize_slot(slot);

    prepare_single_dmsc_gpu<IS_DUAL>(scalar_field[b], trace_max_arcs, trace_min_arcs, *slot.workspace);
    slot.image_index = b;
    gpu::Workspace* workspace = slot.workspace.get();
    slot.persistence = std::async(std::launch::async, [=]() {
      cpu::compute_ppairs_and_simplify<IS_DUAL>(*workspace, persistence_threshold, trace_max_arcs, trace_min_arcs);
    });
  }

  for (auto& slot : slots) finalize_slot(slot);
}

pybind11::object extract_dmsc_gpu(torch::Tensor scalar_field, float persistence_threshold, bool return_gradient = false,
                                  bool is_dual = false, bool trace_max_arcs = false, bool trace_min_arcs = false,
                                  bool trace_max_groups = false, bool trace_min_groups = false) {
  validate_dmsc_input(scalar_field);
  TORCH_CHECK(scalar_field.device().is_cuda() || scalar_field.device().is_mps(),
              "GPU backend requires a CUDA or MPS tensor, got ", scalar_field.device());

#ifndef __APPLE__
  c10::cuda::CUDAGuard device_guard(scalar_field.device());
#endif

  scalar_field = scalar_field.contiguous();

  if (scalar_field.dim() == 2) {
    int H = scalar_field.size(0);
    int W = scalar_field.size(1);

    DMSComplex result;
    {
      pybind11::gil_scoped_release release;
      gpu::Workspace ws(H, W);
      if (is_dual) {
        result = extract_single_dmsc_gpu_t<true>(scalar_field, persistence_threshold, return_gradient, trace_max_arcs,
                                                 trace_min_arcs, trace_max_groups, trace_min_groups, ws);
      } else {
        result = extract_single_dmsc_gpu_t<false>(scalar_field, persistence_threshold, return_gradient, trace_max_arcs,
                                                  trace_min_arcs, trace_max_groups, trace_min_groups, ws);
      }
    }
    // Cast the single struct to a generic Python object
    return pybind11::cast(result);
  } else {
    // 3D Batched case
    int B = scalar_field.size(0);

    std::vector<DMSComplex> results(B);
    {
      pybind11::gil_scoped_release release;
      tbb::global_control control(tbb::global_control::max_allowed_parallelism, at::get_num_threads());
      if (is_dual) {
        extract_batched_dmsc_gpu<true>(scalar_field, persistence_threshold, return_gradient, trace_max_arcs,
                                       trace_min_arcs, trace_max_groups, trace_min_groups, results);
      } else {
        extract_batched_dmsc_gpu<false>(scalar_field, persistence_threshold, return_gradient, trace_max_arcs,
                                        trace_min_arcs, trace_max_groups, trace_min_groups, results);
      }
    }

    // Cast the C++ vector to a Python list
    return pybind11::cast(results);
  }
}

PYBIND11_MODULE(dmsc_gpu, m) {
  pybind11::class_<DMSComplex>(m, "DMSComplex", pybind11::module_local(), "...")
      .def_readonly("shape", &DMSComplex::shape)
      .def_readonly("max_pts", &DMSComplex::max_pts)
      .def_readonly("min_pts", &DMSComplex::min_pts)
      .def_readonly("sad_pts", &DMSComplex::sad_pts)
      .def_readonly("e_max", &DMSComplex::e_max)
      .def_readonly("e_min", &DMSComplex::e_min)
      .def_readonly("p_max", &DMSComplex::p_max)
      .def_readonly("p_min", &DMSComplex::p_min)
      .def_readonly("ppairs_max", &DMSComplex::ppairs_max)
      .def_readonly("ppairs_min", &DMSComplex::ppairs_min)
      .def_readonly("grad_indices", &DMSComplex::grad_indices)
      .def_readonly("peaks", &DMSComplex::peaks)
      .def_readonly("ridges", &DMSComplex::ridges)
      .def_readonly("ridge_offsets", &DMSComplex::ridge_offsets)
      .def_readonly("ridge_arc_offsets", &DMSComplex::ridge_arc_offsets)
      .def_readonly("basins", &DMSComplex::basins)
      .def_readonly("valleys", &DMSComplex::valleys)
      .def_readonly("valley_offsets", &DMSComplex::valley_offsets)
      .def_readonly("valley_arc_offsets", &DMSComplex::valley_arc_offsets)
      .def("__getitem__",
           [](const DMSComplex& r, int idx) {
             switch (idx) {
               case 0:
                 return r.shape;
               case 1:
                 return r.max_pts;
               case 2:
                 return r.min_pts;
               case 3:
                 return r.sad_pts;
               case 4:
                 return r.e_max;
               case 5:
                 return r.e_min;
               case 6:
                 return r.p_max;
               case 7:
                 return r.p_min;
               case 8:
                 return r.ppairs_max;
               case 9:
                 return r.ppairs_min;
               case 10:
                 return r.grad_indices;
               case 11:
                 return r.peaks;
               case 12:
                 return r.ridges;
               case 13:
                 return r.ridge_offsets;
               case 14:
                 return r.ridge_arc_offsets;
               case 15:
                 return r.basins;
               case 16:
                 return r.valleys;
               case 17:
                 return r.valley_offsets;
               case 18:
                 return r.valley_arc_offsets;
               default:
                 throw pybind11::index_error();
             }
           })
      .def("__len__", [](const DMSComplex& r) { return 19; });

  m.def("extract_dmsc", &extract_dmsc_gpu, "GPU based Discrete Morse-Smale complex computation",
        pybind11::arg("scalar_field"), pybind11::arg("persistence_threshold"), pybind11::arg("return_gradient") = false,
        pybind11::arg("is_dual") = false, pybind11::arg("trace_max_arcs") = false,
        pybind11::arg("trace_min_arcs") = false, pybind11::arg("trace_max_groups") = false,
        pybind11::arg("trace_min_groups") = false);
}
