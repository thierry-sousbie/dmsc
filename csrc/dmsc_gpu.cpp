#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <pybind11/stl.h>  // REQUIRED to return std::vector as a Python List
#include <tbb/global_control.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

#include "./cpu/arcs_simplification.hxx"
#include "./cpu/cell_compare.hxx"
#include "./cpu/persistence.hxx"
#include "./dmsc_struct.hxx"
#include "./gpu/workspace.hxx"

#ifdef __APPLE__
#include "./gpu/dmsc_mps.hxx"
#else
#include "./gpu/dmsc_cuda.hxx"
#endif

using namespace gpu;

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
DMSComplex extract_single_dmsc_gpu_t(torch::Tensor scalar_field, float persistence_threshold, int block_size,
                                     bool return_gradient, bool trace_arcs, bool trace_manifolds, gpu::Workspace& ws) {
  int H = scalar_field.size(0);
  int W = scalar_field.size(1);
  int Nx = W + 1;
  int num_cells = 4 * (H + 1) * (W + 1);

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

  // --- DISCRETE GRADIENT ---
  torch::Tensor active_field = scalar_field;
  gpu::compute_gradient<IS_DUAL>(ws, active_field);
  auto& gradient_data = ws.gradient_data;
  // --- PATH TRACING ---
  // We could move because using a reference seems to slow the compiler optimization a bit later on
  // but this is meant to be removed after clean up anyways ...
  std::vector<int>& paired_with = gradient_data.paired_with;
  std::vector<int>& crit_saddles = gradient_data.cp.saddles;
  std::vector<int>& crit_mins = gradient_data.cp.mins;
  std::vector<int>& crit_maxes = gradient_data.cp.maxes;

  // --- PHASE 2: PARALLEL PATH TRACING ---
  gpu::trace_from_saddles<IS_DUAL>(ws);
  auto& arcs_topology = ws.arcs_topology;
  auto& max_saddles = arcs_topology.sorted_max_saddles;
  auto& min_saddles = arcs_topology.sorted_min_saddles;

  // Ensure the image data is strictly on the CPU for the Phase 2/3 value lookups
  active_field = active_field.cpu().contiguous();
  const float* data = active_field.data_ptr<float>();

  // Pre-calculate mappings for faster access
  std::vector<float>& crit_min_vals = ws.hlp.crit_min_vals;
  std::vector<float>& crit_max_vals = ws.hlp.crit_max_vals;
  std::vector<int>& fast_crit_map = ws.hlp.fast_crit_map;
  crit_min_vals.resize(crit_mins.size());
  crit_max_vals.resize(crit_maxes.size());
  fast_crit_map.assign(num_cells, -1);

  // --- MS-complex, persistence pairs and simplification ---
  {
    RECORD_FUNCTION("persistence_preproc_cpu1", {});
    tbb::parallel_invoke(
        [&]() {
          for (size_t i = 0; i < crit_maxes.size(); ++i) {
            fast_crit_map[crit_maxes[i]] = i;
            crit_max_vals[i] =
                cpu::cell_value<IS_DUAL>(3, get_y(crit_maxes[i], Nx), get_x(crit_maxes[i], Nx), H, W, data);
          }
        },
        [&]() {
          for (size_t i = 0; i < crit_mins.size(); ++i) {
            fast_crit_map[crit_mins[i]] = i;
            crit_min_vals[i] =
                cpu::cell_value<IS_DUAL>(0, get_y(crit_mins[i], Nx), get_x(crit_mins[i], Nx), H, W, data);
          }
        },
        [&]() {
          for (size_t i = 0; i < crit_saddles.size(); ++i) {
            fast_crit_map[crit_saddles[i]] = i;
          }
        });
  }
  {
    RECORD_FUNCTION("persistence_preproc_cpu2", {});
    // localize the mapping to the structure for cache, making the persistence thresholding loop faster
    tbb::parallel_for(tbb::blocked_range<size_t>(0, min_saddles.size(), 1024),
                      [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                          auto& ev = min_saddles[i];
                          ev.c1_mid = (ev.c1_id != -1) ? fast_crit_map[ev.c1_id] : -1;
                          ev.c2_mid = (ev.c2_id != -1) ? fast_crit_map[ev.c2_id] : -1;
                        }
                      });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, max_saddles.size(), 1024),
                      [&](const tbb::blocked_range<size_t>& r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                          auto& ev = max_saddles[i];
                          ev.c1_mid = (ev.c1_id != -1) ? fast_crit_map[ev.c1_id] : -1;
                          ev.c2_mid = (ev.c2_id != -1) ? fast_crit_map[ev.c2_id] : -1;
                        }
                      });
  }

  // arcs geometry computation (ridges and valleys)
  // SaddleNodes saddle_nodes;  // arcs incident to saddle points
  auto& saddle_nodes = ws.saddle_nodes;
  if (trace_arcs) {
    gpu::trace_raw_arcs_geometry(ws, gradient_data, fast_crit_map.data(), arcs_topology.max_arcs_len,
                                 arcs_topology.min_arcs_len);
  }

  // UnionFind uf_max(crit_maxes.size());
  // UnionFind uf_min(crit_mins.size());
  // std::vector<bool> max_alive(crit_maxes.size(), true);
  // std::vector<bool> min_alive(crit_mins.size(), true);
  // std::vector<CancelEvent> min_cancellations;
  // std::vector<CancelEvent> max_cancellations;

  // compute_ppairs_and_simplify<IS_DUAL>(persistence_threshold, trace_arcs, fast_crit_map, min_saddles, max_saddles,
  //                                      crit_mins, crit_maxes, crit_min_vals, crit_maxes_vals, min_alive, max_alive,
  //                                      uf_min, uf_max, min_cancellations, max_cancellations);
  cpu::compute_ppairs_and_simplify<IS_DUAL>(ws, persistence_threshold, trace_arcs);
  auto& min_alive = ws.p_data.min_alive;
  auto& max_alive = ws.p_data.max_alive;
  auto& uf_min = ws.p_data.uf_min;
  auto& uf_max = ws.p_data.uf_max;
  // auto& min_cancellations = ws.p_data.min_cancellations;
  // auto& max_cancellations = ws.p_data.max_cancellations;

  // --- 1-manifolds graph ---
  if (trace_arcs) {
    // simplify_arcs_geometry(ws, saddle_nodes, crit_maxes.size(), crit_mins.size(), min_cancellations,
    // max_cancellations);
    cpu::simplify_arcs_geometry(ws);
  }

  // --- Compute basins of attraction ---
  if (trace_manifolds) {
    gpu::compute_cell_groups<IS_DUAL>(ws);
  }

  return ws.complex<IS_DUAL>(return_gradient, trace_arcs);
}

pybind11::object extract_dmsc_gpu(torch::Tensor scalar_field, float persistence_threshold, int block_size = 128,
                                  bool return_gradient = false, bool is_dual = false, bool trace_arcs = false,
                                  bool trace_manifolds = false) {
  TORCH_CHECK(scalar_field.dim() == 2 || scalar_field.dim() == 3, "Input tensor must be 2D [H, W] or 3D [B, H, W]");

  scalar_field = scalar_field.contiguous();

  if (scalar_field.dim() == 2) {
    int H = scalar_field.size(0);
    int W = scalar_field.size(1);

    DMSComplex result;
    gpu::Workspace ws(H, W);
    if (is_dual) {
      result = extract_single_dmsc_gpu_t<true>(scalar_field, persistence_threshold, block_size, return_gradient,
                                               trace_arcs, trace_manifolds, ws);
    } else {
      result = extract_single_dmsc_gpu_t<false>(scalar_field, persistence_threshold, block_size, return_gradient,
                                                trace_arcs, trace_manifolds, ws);
    }
    // Cast the single struct to a generic Python object
    return pybind11::cast(result);
  } else {
    // 3D Batched case
    int B = scalar_field.size(0);
    int H = scalar_field.size(1);
    int W = scalar_field.size(2);

    std::vector<DMSComplex> results;
    results.reserve(B);
    gpu::Workspace ws(H, W);

    // Sequentially iterate the batch dimension. (GPU kernels will max out hardware for each spatial grid)
    for (int b = 0; b < B; ++b) {
      torch::Tensor img = scalar_field[b];  // Shallow slice wrapper

      if (is_dual) {
        results.push_back(extract_single_dmsc_gpu_t<true>(img, persistence_threshold, block_size, return_gradient,
                                                          trace_arcs, trace_manifolds, ws));
      } else {
        results.push_back(extract_single_dmsc_gpu_t<false>(img, persistence_threshold, block_size, return_gradient,
                                                           trace_arcs, trace_manifolds, ws));
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

  // Only one function exported now!
  m.def("extract_dmsc", &extract_dmsc_gpu, "GPU based Discrete Morse-Smale complex computation",
        pybind11::arg("scalar_field"), pybind11::arg("persistence_threshold"), pybind11::arg("block_size") = 128,
        pybind11::arg("return_gradient") = false, pybind11::arg("is_dual") = false, pybind11::arg("trace_arcs") = false,
        pybind11::arg("trace_manifolds") = false);
}