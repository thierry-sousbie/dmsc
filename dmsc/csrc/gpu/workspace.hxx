#pragma once

#include <tbb/parallel_invoke.h>

#include <cstring>
#include <vector>

#include "../cpu/cell_groups_struct.hxx"
#include "../cpu/persistence_struct.hxx"
#include "../dmsc_struct.hxx"
#include "../managed_tensor.hxx"
#include "./arcs_geometry_struct.hxx"
#include "./arcs_topology_struct.hxx"
#include "./gradient_struct.hxx"

namespace gpu {
struct WSHelpers {
  std::vector<float> crit_min_vals;
  std::vector<float> crit_max_vals;
  // look up table of size num_cells mapping critical cell_id to index in gradient_data.cp
  std::vector<int> fast_crit_map;

  ManagedTensor fast_region_id;

  ManagedTensor temp_flat_max;
  ManagedTensor temp_flat_min;

  std::vector<int> temp_max_to_out;
  std::vector<int> temp_min_to_out;
  std::vector<int> temp_sad_to_out;

  ManagedTensor out_max_pts;
  ManagedTensor out_min_pts;
  ManagedTensor out_sad_pts;
  ManagedTensor out_e_max;
  ManagedTensor out_e_min;
  ManagedTensor out_p_max;
  ManagedTensor out_p_min;
  ManagedTensor out_ppairs_max;
  ManagedTensor out_ppairs_min;
  ManagedTensor out_grad;

  ManagedTensor out_ridge_faces;
  ManagedTensor out_ridge_faces_off;
  ManagedTensor out_arc_faces_off;
  ManagedTensor out_ridge_vertices;
  ManagedTensor out_ridge_vertices_off;
  ManagedTensor out_arc_vertices_off;

  WSHelpers()
      : fast_region_id("fast_region_id", false),
        temp_flat_max("temp_flat_max", false),
        temp_flat_min("temp_flat_min", false),
        out_max_pts("out_max_pts", false),
        out_min_pts("out_min_pts", false),
        out_sad_pts("out_sad_pts", false),
        out_e_max("out_e_max", false),
        out_e_min("out_e_min", false),
        out_p_max("out_p_max", false),
        out_p_min("out_p_min", false),
        out_ppairs_max("out_ppairs_max", false),
        out_ppairs_min("out_ppairs_min", false),
        out_grad("out_grad", false),
        out_ridge_faces("out_ridge_faces", false),
        out_ridge_faces_off("out_ridge_faces_off", false),
        out_arc_faces_off("out_arc_faces_off", false),
        out_ridge_vertices("out_ridge_vertices", false),
        out_ridge_vertices_off("out_ridge_vertices_off", false),
        out_arc_vertices_off("out_arc_vertices_off", false) {}

};

struct Workspace {
  int H;
  int W;
  int Nx;
  int num_cells;

  WSHelpers hlp;

  torch::Tensor d_data;  // hold a copy of the input scalar field
  torch::Tensor d_fast_crit_map;
  gpu::GradientData gradient_data;
  gpu::ArcsTopology arcs_topology;
  gpu::SaddleNodes saddle_nodes;
  cpu::CellGroupsData cell_groups;
  cpu::PersistenceData p_data;

  Workspace(int H, int W) : H(H), W(W), Nx(W + 1), num_cells(4 * (H + 1) * (W + 1)) {}

  template <bool IS_DUAL = false>
  DMSComplex complex(bool return_gradient, bool trace_max_arcs, bool trace_min_arcs, bool trace_face_groups,
                     bool trace_vertex_groups) {
    RECORD_FUNCTION("Workspace::complex", {});

    auto dev = d_data.device();
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto dev_opts_i = torch::TensorOptions().dtype(torch::kInt32).device(dev);

    // Retrieve necessary structures from the workspace
    auto& crit_saddles = gradient_data.cp.saddles;
    auto& crit_mins = gradient_data.cp.mins;
    auto& crit_maxes = gradient_data.cp.maxes;
    auto& max_saddles = arcs_topology.sorted_max_saddles;
    auto& min_saddles = arcs_topology.sorted_min_saddles;
    auto& fast_crit_map = hlp.fast_crit_map;

    auto& min_alive = p_data.min_alive;
    auto& max_alive = p_data.max_alive;
    auto& uf_min = p_data.uf_min;
    auto& uf_max = p_data.uf_max;

    int num_surviving_sads = min_saddles.size();

    // Prepare dense lookup tables linking original index to output array index
    hlp.temp_max_to_out.assign(crit_maxes.size(), -1);
    hlp.temp_min_to_out.assign(crit_mins.size(), -1);
    hlp.temp_sad_to_out.assign(crit_saddles.size(), -1);

    int* max_to_out = hlp.temp_max_to_out.data();
    int* min_to_out = hlp.temp_min_to_out.data();
    int* sad_to_out = hlp.temp_sad_to_out.data();

    int* out_sad = hlp.out_sad_pts.request({num_surviving_sads}, opts_i).template data_ptr<int>();
    float* out_p_max = hlp.out_p_max.request({num_surviving_sads}, opts_f).template data_ptr<float>();
    float* out_p_min = hlp.out_p_min.request({num_surviving_sads}, opts_f).template data_ptr<float>();
    int* out_ppairs_max = hlp.out_ppairs_max.request({num_surviving_sads}, opts_i).template data_ptr<int>();
    int* out_ppairs_min = hlp.out_ppairs_min.request({num_surviving_sads}, opts_i).template data_ptr<int>();
    int* out_e_max = hlp.out_e_max.request({num_surviving_sads * 2, 2}, opts_i).template data_ptr<int>();
    int* out_e_min = hlp.out_e_min.request({num_surviving_sads * 2, 2}, opts_i).template data_ptr<int>();

    // 1. Filter out dead critical points on CPU to populate map arrays
    int max_count = 0, min_count = 0;

    tbb::parallel_invoke(
        [&] {
          for (size_t i = 0; i < crit_maxes.size(); ++i) {
            if (max_alive[i]) max_to_out[i] = max_count++;
          }
        },
        [&] {
          for (size_t i = 0; i < crit_mins.size(); ++i) {
            if (min_alive[i]) min_to_out[i] = min_count++;
          }
        },
        [&] {
          for (int i = 0; i < num_surviving_sads; ++i) {
            auto& ev = min_saddles[i];
            int dense_idx = fast_crit_map[ev.saddle_id];
            sad_to_out[dense_idx] = i;
            out_sad[i] = ev.saddle_id;
            ev.saddle_id = i;
            ev.pair_id = (ev.pair_id != -1) ? fast_crit_map[ev.pair_id] : -1;
          }
        },
        [&] {
          for (int i = 0; i < num_surviving_sads; ++i) {
            auto& ev = max_saddles[i];
            int dense_idx = fast_crit_map[ev.saddle_id];
            ev.saddle_id = dense_idx;
            ev.pair_id = (ev.pair_id != -1) ? fast_crit_map[ev.pair_id] : -1;
          }
        });

    int e_max_count = 0, e_min_count = 0;

    tbb::parallel_invoke(
        [&] {
          // 2. Resolve final edges and persistence pairings based on simplified topology
          for (int i = 0; i < num_surviving_sads; ++i) {
            const auto& ev = min_saddles[i];
            int s_idx = ev.saddle_id;

            out_p_min[s_idx] = ev.persistence;
            out_ppairs_min[s_idx] = (ev.pair_id != -1) ? min_to_out[uf_min.find(ev.pair_id)] : -1;

            int root_min1 = (ev.c1_mid != -1) ? uf_min.find(ev.c1_mid) : -1;
            int root_min2 = (ev.c2_mid != -1) ? uf_min.find(ev.c2_mid) : -1;

            if (root_min1 != -1 && min_to_out[root_min1] != -1) {
              out_e_min[e_min_count * 2] = s_idx;
              out_e_min[e_min_count * 2 + 1] = min_to_out[root_min1];
              e_min_count++;
            }
            if (root_min1 != root_min2 && root_min2 != -1 && min_to_out[root_min2] != -1) {
              out_e_min[e_min_count * 2] = s_idx;
              out_e_min[e_min_count * 2 + 1] = min_to_out[root_min2];
              e_min_count++;
            }
          }
        },
        [&] {
          for (int i = 0; i < num_surviving_sads; ++i) {
            const auto& ev = max_saddles[i];
            int s_idx = sad_to_out[ev.saddle_id];

            out_p_max[s_idx] = ev.persistence;
            out_ppairs_max[s_idx] = (ev.pair_id != -1) ? max_to_out[uf_max.find(ev.pair_id)] : -1;

            int root_max1 = (ev.c1_mid != -1) ? uf_max.find(ev.c1_mid) : -1;
            int root_max2 = (ev.c2_mid != -1) ? uf_max.find(ev.c2_mid) : -1;

            if (root_max1 != -1 && max_to_out[root_max1] != -1) {
              out_e_max[e_max_count * 2] = s_idx;
              out_e_max[e_max_count * 2 + 1] = max_to_out[root_max1];
              e_max_count++;
            }
            if (root_max1 != root_max2 && root_max2 != -1 && max_to_out[root_max2] != -1) {
              out_e_max[e_max_count * 2] = s_idx;
              out_e_max[e_max_count * 2 + 1] = max_to_out[root_max2];
              e_max_count++;
            }
          }
        },
        [&] {
          // 3. Gather manifold geometries (ridges/valleys) if requested
          if (trace_max_arcs || trace_min_arcs) {
            int num_ridge_faces = 0, num_ridge_vertices = 0;
            int num_arcs = 0;

            for (const auto& ev : min_saddles) {
              int original_saddle_id = out_sad[ev.saddle_id];
              int s_idx = fast_crit_map[original_saddle_id];
              const auto& node = saddle_nodes.nodes[s_idx];
              if (!node.alive) continue;
              if (trace_max_arcs) num_ridge_faces += node.max_arcs[0].length + node.max_arcs[1].length;
              if (trace_min_arcs) num_ridge_vertices += node.min_arcs[0].length + node.min_arcs[1].length;
              num_arcs++;
            }

            // Preallocate exact memory capacity required for geometry coordinates
            int* out_ridge_faces = hlp.out_ridge_faces.request({num_ridge_faces}, opts_i).template data_ptr<int>();
            int* out_ridge_faces_off = hlp.out_ridge_faces_off.request({num_arcs + 1}, opts_i).template data_ptr<int>();
            int* out_arc_faces_off = hlp.out_arc_faces_off.request({num_arcs}, opts_i).template data_ptr<int>();

            int* out_ridge_vertices =
                hlp.out_ridge_vertices.request({num_ridge_vertices}, opts_i).template data_ptr<int>();
            int* out_ridge_vertices_off =
                hlp.out_ridge_vertices_off.request({num_arcs + 1}, opts_i).template data_ptr<int>();
            int* out_arc_vertices_off = hlp.out_arc_vertices_off.request({num_arcs}, opts_i).template data_ptr<int>();

            // Flatten geometric arcs continuously in memory and track array offsets
            out_ridge_faces_off[0] = 0;
            out_ridge_vertices_off[0] = 0;

            int f_idx = 0, v_idx = 0;
            int r_f_off_idx = 1, r_v_off_idx = 1;
            int a_f_off_idx = 0, a_v_off_idx = 0;

            auto flat_max_geom = saddle_nodes.flat_max_geom.get();
            auto flat_min_geom = saddle_nodes.flat_min_geom.get();
            bool gather_max_on_device = flat_max_geom.device().is_cuda();
            bool gather_min_on_device = flat_min_geom.device().is_cuda();
            const int* flat_max_ptr =
                gather_max_on_device ? nullptr : flat_max_geom.template data_ptr<int>();
            const int* flat_min_ptr =
                gather_min_on_device ? nullptr : flat_min_geom.template data_ptr<int>();

            for (const auto& ev : min_saddles) {
              int original_saddle_id = out_sad[ev.saddle_id];
              int s_idx = fast_crit_map[original_saddle_id];
              const auto& node = saddle_nodes.nodes[s_idx];
              if (!node.alive) continue;

              if (trace_max_arcs) {
                for (int k = 0; k < 2; ++k) {
                  int len = node.max_arcs[k].length;
                  if (gather_max_on_device) {
                    for (int j = 0; j < len; ++j) out_ridge_faces[f_idx + j] = node.max_arcs[k].offset + j;
                  } else {
                    const int* start = flat_max_ptr + node.max_arcs[k].offset;
                    std::memcpy(out_ridge_faces + f_idx, start, len * sizeof(int));
                  }
                  f_idx += len;
                  if (k == 0) out_arc_faces_off[a_f_off_idx++] = f_idx;
                }
                out_ridge_faces_off[r_f_off_idx++] = f_idx;
              }

              if (trace_min_arcs) {
                for (int k = 0; k < 2; ++k) {
                  int len = node.min_arcs[k].length;
                  if (gather_min_on_device) {
                    for (int j = 0; j < len; ++j) out_ridge_vertices[v_idx + j] = node.min_arcs[k].offset + j;
                  } else {
                    const int* start = flat_min_ptr + node.min_arcs[k].offset;
                    std::memcpy(out_ridge_vertices + v_idx, start, len * sizeof(int));
                  }
                  v_idx += len;
                  if (k == 0) out_arc_vertices_off[a_v_off_idx++] = v_idx;
                }
                out_ridge_vertices_off[r_v_off_idx++] = v_idx;
              }
            }

            if (gather_max_on_device && num_ridge_faces > 0) {
              auto indices = hlp.out_ridge_faces.get().to(dev);
              auto gathered = flat_max_geom.index_select(0, indices);
              hlp.out_ridge_faces.adopt(std::move(gathered));
            }
            if (gather_min_on_device && num_ridge_vertices > 0) {
              auto indices = hlp.out_ridge_vertices.get().to(dev);
              auto gathered = flat_min_geom.index_select(0, indices);
              hlp.out_ridge_vertices.adopt(std::move(gathered));
            }
          }
        });

    // 4. Bind populated PyTorch buffers into the final structured output natively
    DMSComplex result;
    result.shape = torch::empty({2}, opts_i);
    result.shape.template accessor<int32_t, 1>()[0] = H;
    result.shape.template accessor<int32_t, 1>()[1] = W;

    torch::Tensor d_max_alive = torch::from_blob((void*)max_alive.data(), {(long)crit_maxes.size()}, torch::kUInt8)
                                    .to(dev, /*non_blocking=*/true)
                                    .to(torch::kBool);
    torch::Tensor d_min_alive = torch::from_blob((void*)min_alive.data(), {(long)crit_mins.size()}, torch::kUInt8)
                                    .to(dev, /*non_blocking=*/true)
                                    .to(torch::kBool);

    auto t_max = gradient_data.d_maxes.get().masked_select(d_max_alive);
    auto t_min = gradient_data.d_mins.get().masked_select(d_min_alive);

    auto t_sad = hlp.out_sad_pts.get().slice(0, 0, num_surviving_sads).to(dev);

    result.sad_pts = t_sad;
    result.grad_indices = return_gradient ? gradient_data.d_paired_with.get().clone() : torch::empty({0}, dev_opts_i);

    auto t_e_max = hlp.out_e_max.get().slice(0, 0, e_max_count).to(dev);
    auto t_e_min = hlp.out_e_min.get().slice(0, 0, e_min_count).to(dev);
    auto t_p_max = hlp.out_p_max.get().slice(0, 0, num_surviving_sads).to(dev);
    auto t_p_min = hlp.out_p_min.get().slice(0, 0, num_surviving_sads).to(dev);
    auto t_ppairs_max = hlp.out_ppairs_max.get().slice(0, 0, num_surviving_sads).to(dev);
    auto t_ppairs_min = hlp.out_ppairs_min.get().slice(0, 0, num_surviving_sads).to(dev);

    auto t_ridge_faces = trace_max_arcs ? hlp.out_ridge_faces.get().to(dev) : torch::empty({0}, dev_opts_i);
    auto t_ridge_faces_off =
        trace_max_arcs ? hlp.out_ridge_faces_off.get().to(dev) : torch::empty({0}, dev_opts_i);
    auto t_arc_faces_off = trace_max_arcs ? hlp.out_arc_faces_off.get().to(dev) : torch::empty({0}, dev_opts_i);

    auto t_ridge_vertices = trace_min_arcs ? hlp.out_ridge_vertices.get().to(dev) : torch::empty({0}, dev_opts_i);
    auto t_ridge_vertices_off =
        trace_min_arcs ? hlp.out_ridge_vertices_off.get().to(dev) : torch::empty({0}, dev_opts_i);
    auto t_arc_vertices_off =
        trace_min_arcs ? hlp.out_arc_vertices_off.get().to(dev) : torch::empty({0}, dev_opts_i);

    auto t_face_groups =
        trace_face_groups ? cell_groups.face_groups.get().to(dev).clone() : torch::empty({0, 0}, dev_opts_i);
    auto t_vertex_groups =
        trace_vertex_groups ? cell_groups.vertex_groups.get().to(dev).clone() : torch::empty({0, 0}, dev_opts_i);

    // 5. Permute definitions of Maxima vs Minima depending on whether this is a Primal or Dual complex
    if (IS_DUAL) {
      result.max_pts = t_min;
      result.min_pts = t_max;
      result.e_max = t_e_min;
      result.e_min = t_e_max;
      result.p_max = t_p_min;
      result.p_min = t_p_max;
      result.ppairs_max = t_ppairs_min;
      result.ppairs_min = t_ppairs_max;

      result.peaks = t_vertex_groups;
      result.ridges = t_ridge_vertices;
      result.ridge_offsets = t_ridge_vertices_off;
      result.ridge_arc_offsets = t_arc_vertices_off;

      result.basins = t_face_groups;
      result.valleys = t_ridge_faces;
      result.valley_offsets = t_ridge_faces_off;
      result.valley_arc_offsets = t_arc_faces_off;

    } else {
      result.max_pts = t_max;
      result.min_pts = t_min;
      result.e_max = t_e_max;
      result.e_min = t_e_min;
      result.p_max = t_p_max;
      result.p_min = t_p_min;
      result.ppairs_max = t_ppairs_max;
      result.ppairs_min = t_ppairs_min;

      result.peaks = t_face_groups;
      result.ridges = t_ridge_faces;
      result.ridge_offsets = t_ridge_faces_off;
      result.ridge_arc_offsets = t_arc_faces_off;

      result.basins = t_vertex_groups;
      result.valleys = t_ridge_vertices;
      result.valley_offsets = t_ridge_vertices_off;
      result.valley_arc_offsets = t_arc_vertices_off;
    }

    return result;
  }
};
}  // namespace gpu
