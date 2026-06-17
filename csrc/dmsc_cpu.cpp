#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <pybind11/stl.h>  // REQUIRED to return std::vector as a Python List
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "./cpu/arcs_geometry.hxx"
#include "./cpu/arcs_simplification.hxx"
#include "./cpu/arcs_simplification_test.hxx"
#include "./cpu/arcs_topology.hxx"
#include "./cpu/cell_groups.hxx"
#include "./cpu/gradient.hxx"
#include "./cpu/persistence.hxx"
#include "./cpu/workspace.hxx"
#include "./dmsc_struct.hxx"

/*
Cell_type Name    Geometry           Count       Topology (PRIMAL)  Topology (DUAL)
3         Face    pixel corner       (W+1)x(H+1) Maximum            Minimum
2         V-edge  edge btwn pixels   (W  )x(H+1) Saddle             Saddle
1         H-edge  edge btwn pixels   (W+1)x(H  ) Saddle             Saddle
0         Vertex  pixel              (W  )x(H  ) Minimum            Maximum

note that the cell complex is defined as the dual of the image tesselation into a pixel grid. The
value of a cell is that of it highest adjacent vertex (pixel) by convention.

When IS_DUAL is true, cell types are reversed, and the value of a cell is given by that of its
lowest vertex and we sort in opposite direction. At the end, we swap minima and maxima, so we
compute the dual complex with the role of vertices/edges/faces reversed, i.e. they
represent maxima/saddles/minima respectively.
*/
template <bool IS_DUAL = false>
DMSComplex extract_single_dmsc_cpu_t(torch::Tensor scalar_field, float persistence_threshold, int block_size,
                                     bool return_gradient, bool trace_arcs, bool trace_manifolds, cpu::Workspace& ws) {
  int H = scalar_field.size(0);
  int W = scalar_field.size(1);
  int Nx = W + 1;
  int num_cells = 4 * (H + 1) * (W + 1);
  const float* data = scalar_field.data_ptr<float>();

  tbb::global_control control(tbb::global_control::max_allowed_parallelism, at::get_num_threads());

  int num_blocks_y = (H + block_size - 1) / block_size;
  int num_blocks_x = (W + block_size - 1) / block_size;
  int total_blocks = num_blocks_y * num_blocks_x;

  auto gradient_data = compute_gradient_and_crit_points<IS_DUAL>(scalar_field, total_blocks, num_blocks_x, block_size);
  // We could move because using a reference seems to slow the compiler optimization a bit later on
  // but this is meant to be removed after clean up anyways ...
  std::vector<int>& paired_with = gradient_data.paired_with;
  std::vector<int>& crit_saddles = gradient_data.cp.saddles;
  std::vector<int>& crit_mins = gradient_data.cp.mins;
  std::vector<int>& crit_maxes = gradient_data.cp.maxes;

  // --- PATH TRACING ---
  auto arcs_topology = trace_from_saddles<IS_DUAL>(data, gradient_data, H, W, Nx);
  std::vector<SadEvent> max_saddles = std::move(arcs_topology.sorted_max_saddles);
  std::vector<SadEvent> min_saddles = std::move(arcs_topology.sorted_min_saddles);

  // Pre-calculate mappings for faster access
  std::vector<float> crit_min_vals(crit_mins.size());
  std::vector<float> crit_maxes_vals(crit_maxes.size());
  std::vector<int> fast_crit_map(num_cells, -1);
  {
    RECORD_FUNCTION("persistence_preproc_cpu1", {});
    tbb::parallel_invoke(
        [&]() {
          for (size_t i = 0; i < crit_maxes.size(); ++i) {
            fast_crit_map[crit_maxes[i]] = i;
            crit_maxes_vals[i] = cell_value<IS_DUAL>(3, get_y(crit_maxes[i], Nx), get_x(crit_maxes[i], Nx), H, W, data);
          }
        },
        [&]() {
          for (size_t i = 0; i < crit_mins.size(); ++i) {
            fast_crit_map[crit_mins[i]] = i;
            crit_min_vals[i] = cell_value<IS_DUAL>(0, get_y(crit_mins[i], Nx), get_x(crit_mins[i], Nx), H, W, data);
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
  SaddleNodes saddle_nodes;  // arcs incident to saddle points
  if (trace_arcs) {
    saddle_nodes = trace_raw_arcs_geometry(gradient_data, fast_crit_map, arcs_topology.max_arcs_len,
                                           arcs_topology.min_arcs_len, H, W, Nx);
  }

  UnionFind uf_max(crit_maxes.size());
  UnionFind uf_min(crit_mins.size());

  std::vector<bool> max_alive(crit_maxes.size(), true);
  std::vector<bool> min_alive(crit_mins.size(), true);

  std::vector<CancelEvent> min_cancellations;
  std::vector<CancelEvent> max_cancellations;

  compute_ppairs_and_simplify<IS_DUAL>(persistence_threshold, trace_arcs, fast_crit_map, min_saddles, max_saddles,
                                       crit_mins, crit_maxes, crit_min_vals, crit_maxes_vals, min_alive, max_alive,
                                       uf_min, uf_max, min_cancellations, max_cancellations);

  // --- 1-manifolds graph ---
  if (trace_arcs) {
    simplify_arcs_geometry(saddle_nodes, crit_maxes.size(), crit_mins.size(), min_cancellations, max_cancellations);
  }
  // --- Compute basins of attraction ---
  CellGroupsData cell_groups;
  if (trace_manifolds)
    cell_groups = compute_cell_groups(paired_with, fast_crit_map, uf_max, crit_maxes, uf_min, crit_mins, max_alive,
                                      min_alive, H, W);

  std::vector<int> ridge_faces, ridge_vertices;
  std::vector<int> ridge_faces_offsets = {0}, ridge_vertices_offsets = {0};
  std::vector<int> arc_faces_offsets, arc_vertices_offsets;

  if (trace_arcs) {
    RECORD_FUNCTION("populate_ridges_and_valleys_cpu", {});

    // FIX: Extract raw pointers from the Tensors to use as C++ iterators
    const int* flat_max_ptr = saddle_nodes.flat_max_geom.data_ptr<int>();
    const int* flat_min_ptr = saddle_nodes.flat_min_geom.data_ptr<int>();

    for (const auto& ev : min_saddles) {
      int s_idx = fast_crit_map[ev.saddle_id];
      const auto& node = saddle_nodes.saddle_nodes[s_idx];

      if (!node.alive) continue;  // Safety check

      const int* max_arc0_start = flat_max_ptr + node.max_arcs[0].offset;
      ridge_faces.insert(ridge_faces.end(), max_arc0_start, max_arc0_start + node.max_arcs[0].length);
      arc_faces_offsets.push_back(ridge_faces.size());

      const int* max_arc1_start = flat_max_ptr + node.max_arcs[1].offset;
      ridge_faces.insert(ridge_faces.end(), max_arc1_start, max_arc1_start + node.max_arcs[1].length);
      ridge_faces_offsets.push_back(ridge_faces.size());

      const int* min_arc0_start = flat_min_ptr + node.min_arcs[0].offset;
      ridge_vertices.insert(ridge_vertices.end(), min_arc0_start, min_arc0_start + node.min_arcs[0].length);
      arc_vertices_offsets.push_back(ridge_vertices.size());

      const int* min_arc1_start = flat_min_ptr + node.min_arcs[1].offset;
      ridge_vertices.insert(ridge_vertices.end(), min_arc1_start, min_arc1_start + node.min_arcs[1].length);
      ridge_vertices_offsets.push_back(ridge_vertices.size());
    }
  }

  // --- Generate formatted ouptut data ---
  std::vector<int> final_maxes, final_mins, final_sads;
  std::vector<int> max_to_out(crit_maxes.size(), -1);
  std::vector<int> min_to_out(crit_mins.size(), -1);

  // Maps the dense input index of a saddle to its final output index
  std::vector<int> sad_to_out(crit_saddles.size(), -1);

  std::vector<std::pair<int, int>> edges_sad_max, edges_sad_min;
  std::vector<float> p_max_list, p_min_list;
  std::vector<int> ppairs_max_list, ppairs_min_list;

  {
    RECORD_FUNCTION("deduplicate_preproc_cpu", {});
    // Because of the synchronized erase, the size is exact and identical for both vectors
    int num_surviving_sads = min_saddles.size();
    final_sads.reserve(num_surviving_sads);
    p_max_list.resize(num_surviving_sads, 0.0f);
    p_min_list.resize(num_surviving_sads, 0.0f);
    ppairs_max_list.resize(num_surviving_sads, -1);
    ppairs_min_list.resize(num_surviving_sads, -1);
    edges_sad_max.reserve(num_surviving_sads * 2);
    edges_sad_min.reserve(num_surviving_sads * 2);

    tbb::parallel_invoke(
        [&]() {
          final_maxes.reserve(crit_maxes.size());
          for (size_t i = 0; i < crit_maxes.size(); ++i) {
            if (max_alive[i]) {
              max_to_out[i] = final_maxes.size();
              int m = crit_maxes[i];
              final_maxes.push_back(m);
            }
          }
        },
        [&]() {
          final_mins.reserve(crit_mins.size());
          for (size_t i = 0; i < crit_mins.size(); ++i) {
            if (min_alive[i]) {
              min_to_out[i] = final_mins.size();
              int m = crit_mins[i];
              final_mins.push_back(m);
            }
          }
        },
        [&]() {
          for (auto& ev : min_saddles) {
            int dense_idx = fast_crit_map[ev.saddle_id];
            int s_idx = final_sads.size();
            final_sads.push_back(ev.saddle_id);
            sad_to_out[dense_idx] = s_idx;
            ev.saddle_id = s_idx;
            ev.pair_id = (ev.pair_id != -1) ? fast_crit_map[ev.pair_id] : -1;
          }
        },
        [&]() {
          for (auto& ev : max_saddles) {
            int dense_idx = fast_crit_map[ev.saddle_id];
            ev.saddle_id = dense_idx;
            ev.pair_id = (ev.pair_id != -1) ? fast_crit_map[ev.pair_id] : -1;
          }
        });
  }

  {
    RECORD_FUNCTION("deduplicate_cpu", {});
    tbb::parallel_invoke(
        [&]() {
          for (const auto& ev : min_saddles) {
            int s_idx = ev.saddle_id;

            p_min_list[s_idx] = ev.persistence;
            ppairs_min_list[s_idx] = (ev.pair_id != -1) ? min_to_out[uf_min.find(ev.pair_id)] : -1;

            int root_min1 = (ev.c1_id != -1) ? uf_min.find(ev.c1_mid) : -1;
            int root_min2 = (ev.c2_id != -1) ? uf_min.find(ev.c2_mid) : -1;

            if (root_min1 != -1 && min_to_out[root_min1] != -1) {
              edges_sad_min.push_back({s_idx, min_to_out[root_min1]});
            }
            if (root_min1 != root_min2 && root_min2 != -1 && min_to_out[root_min2] != -1) {
              edges_sad_min.push_back({s_idx, min_to_out[root_min2]});
            }
          }
        },
        [&]() {
          for (const auto& ev : max_saddles) {
            int s_idx = sad_to_out[ev.saddle_id];

            p_max_list[s_idx] = ev.persistence;
            ppairs_max_list[s_idx] = (ev.pair_id != -1) ? max_to_out[uf_max.find(ev.pair_id)] : -1;

            int root_max1 = (ev.c1_id != -1) ? uf_max.find(ev.c1_mid) : -1;
            int root_max2 = (ev.c2_id != -1) ? uf_max.find(ev.c2_mid) : -1;

            if (root_max1 != -1 && max_to_out[root_max1] != -1) {
              edges_sad_max.push_back({s_idx, max_to_out[root_max1]});
            }
            if (root_max1 != root_max2 && root_max2 != -1 && max_to_out[root_max2] != -1) {
              edges_sad_max.push_back({s_idx, max_to_out[root_max2]});
            }
          }
        });
  }

  auto opts_i = torch::TensorOptions().dtype(torch::kInt32);
  auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);

  auto t_shape = torch::empty({2}, opts_i);
  t_shape.accessor<int32_t, 1>()[0] = H;
  t_shape.accessor<int32_t, 1>()[1] = W;

  torch::Tensor t_max, t_min, t_sad, t_e_max, t_e_min, t_ppairs_max, t_ppairs_min;
  torch::Tensor t_p_max, t_p_min, t_grad;
  torch::Tensor t_ridge_faces, t_ridge_faces_off, t_arc_faces_off;
  torch::Tensor t_ridge_vertices, t_ridge_vertices_off, t_arc_vertices_off;

  {
    RECORD_FUNCTION("final_memcpy", {});

    // FIX: Explicitly use int64_t to satisfy PyTorch's IntArrayRef requirement
    auto to_tensor_i = [&](const auto& vec, std::vector<int64_t> shape) {
      return vec.empty() ? torch::empty(shape, opts_i) : torch::from_blob((void*)vec.data(), shape, opts_i).clone();
    };
    auto to_tensor_f = [&](const auto& vec, std::vector<int64_t> shape) {
      return vec.empty() ? torch::empty(shape, opts_f) : torch::from_blob((void*)vec.data(), shape, opts_f).clone();
    };

    t_max = to_tensor_i(final_maxes, {(int64_t)final_maxes.size()});
    t_min = to_tensor_i(final_mins, {(int64_t)final_mins.size()});
    t_sad = to_tensor_i(final_sads, {(int64_t)final_sads.size()});

    t_e_max = to_tensor_i(edges_sad_max, {(int64_t)edges_sad_max.size(), 2});
    t_e_min = to_tensor_i(edges_sad_min, {(int64_t)edges_sad_min.size(), 2});

    t_ppairs_max = to_tensor_i(ppairs_max_list, {(int64_t)ppairs_max_list.size()});
    t_ppairs_min = to_tensor_i(ppairs_min_list, {(int64_t)ppairs_min_list.size()});

    t_p_max = to_tensor_f(p_max_list, {(int64_t)p_max_list.size()});
    t_p_min = to_tensor_f(p_min_list, {(int64_t)p_min_list.size()});

    t_grad = return_gradient ? torch::from_blob((void*)paired_with.data(), {num_cells}, opts_i).clone()
                             : torch::empty({0}, opts_i);

    t_ridge_faces = to_tensor_i(ridge_faces, {(int64_t)ridge_faces.size()});
    t_ridge_faces_off = to_tensor_i(ridge_faces_offsets, {(int64_t)ridge_faces_offsets.size()});
    t_arc_faces_off = to_tensor_i(arc_faces_offsets, {(int64_t)arc_faces_offsets.size()});

    t_ridge_vertices = to_tensor_i(ridge_vertices, {(int64_t)ridge_vertices.size()});
    t_ridge_vertices_off = to_tensor_i(ridge_vertices_offsets, {(int64_t)ridge_vertices_offsets.size()});
    t_arc_vertices_off = to_tensor_i(arc_vertices_offsets, {(int64_t)arc_vertices_offsets.size()});
  }

  DMSComplex result;
  {
    RECORD_FUNCTION("output", {});
    result.shape = t_shape;
    result.sad_pts = t_sad;
    result.grad_indices = t_grad;

    if (IS_DUAL) {
      result.max_pts = t_min;
      result.min_pts = t_max;
      result.e_max = t_e_min;
      result.e_min = t_e_max;
      result.p_max = t_p_min;
      result.p_min = t_p_max;
      result.ppairs_max = t_ppairs_min;
      result.ppairs_min = t_ppairs_max;

      // In Dual, Maxima are Vertices, so Ascending Ridges are Vertex paths
      result.peaks = cell_groups.vertex_groups;
      result.ridges = t_ridge_vertices;
      result.ridge_offsets = t_ridge_vertices_off;
      result.ridge_arc_offsets = t_arc_vertices_off;

      // In Dual, Minima are Faces, so Descending Trenches are Face paths
      result.basins = cell_groups.face_groups;
      result.valleys = t_ridge_faces;
      result.valley_offsets = t_ridge_faces_off;
      result.valley_arc_offsets = t_arc_faces_off;

    } else {
      // In Primal: Maxima are Faces (3), Minima are Vertices (0)
      result.max_pts = t_max;
      result.min_pts = t_min;
      result.e_max = t_e_max;
      result.e_min = t_e_min;
      result.p_max = t_p_max;
      result.p_min = t_p_min;
      result.ppairs_max = t_ppairs_max;
      result.ppairs_min = t_ppairs_min;

      // In Primal, Maxima are Faces, so Ascending Ridges are Face paths
      result.peaks = cell_groups.face_groups;
      result.ridges = t_ridge_faces;
      result.ridge_offsets = t_ridge_faces_off;
      result.ridge_arc_offsets = t_arc_faces_off;

      // In Primal, Minima are Vertices, so Descending Trenches are Vertex paths
      result.basins = cell_groups.vertex_groups;
      result.valleys = t_ridge_vertices;
      result.valley_offsets = t_ridge_vertices_off;
      result.valley_arc_offsets = t_arc_vertices_off;
    }
  }

  return result;
}

pybind11::object extract_dmsc_cpu(torch::Tensor scalar_field, float persistence_threshold, int block_size = 128,
                                  bool return_gradient = false, bool is_dual = false, bool trace_arcs = false,
                                  bool trace_manifolds = false) {
  TORCH_CHECK(scalar_field.dim() == 2 || scalar_field.dim() == 3, "Input tensor must be 2D [H, W] or 3D [B, H, W]");

  scalar_field = scalar_field.contiguous();

  if (scalar_field.dim() == 2) {
    // int H = scalar_field.size(0);
    // int W = scalar_field.size(1);

    DMSComplex result;
    cpu::Workspace ws;
    if (is_dual) {
      result = extract_single_dmsc_cpu_t<true>(scalar_field, persistence_threshold, block_size, return_gradient,
                                               trace_arcs, trace_manifolds, ws);
    } else {
      result = extract_single_dmsc_cpu_t<false>(scalar_field, persistence_threshold, block_size, return_gradient,
                                                trace_arcs, trace_manifolds, ws);
    }

    return pybind11::cast(result);

  } else {
    // 3D Batched case
    int B = scalar_field.size(0);
    int H = scalar_field.size(1);
    int W = scalar_field.size(2);
    int num_cells = 4 * (H + 1) * (W + 1);

    // Workspace Pool: Allocated exactly once for the entire batch
    std::vector<int> fast_crit_map(num_cells);
    std::vector<int> fast_region_id(num_cells);

    std::vector<DMSComplex> results;
    results.reserve(B);
    cpu::Workspace ws;

    for (int b = 0; b < B; ++b) {
      torch::Tensor img = scalar_field[b];  // Shallow slice wrapper

      if (is_dual) {
        results.push_back(extract_single_dmsc_cpu_t<true>(img, persistence_threshold, block_size, return_gradient,
                                                          trace_arcs, trace_manifolds, ws));
      } else {
        results.push_back(extract_single_dmsc_cpu_t<false>(img, persistence_threshold, block_size, return_gradient,
                                                           trace_arcs, trace_manifolds, ws));
      }
    }

    return pybind11::cast(results);
  }
}

PYBIND11_MODULE(dmsc_cpu, m) {
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
  m.def("extract_dmsc", &extract_dmsc_cpu, "Multithreaded Discrete Morse-Smale complex computation",
        pybind11::arg("scalar_field"), pybind11::arg("persistence_threshold"), pybind11::arg("block_size") = 128,
        pybind11::arg("return_gradient") = false, pybind11::arg("is_dual") = false, pybind11::arg("trace_arcs") = false,
        pybind11::arg("trace_manifolds") = false);
}