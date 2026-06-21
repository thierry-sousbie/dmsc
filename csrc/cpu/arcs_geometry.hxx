#pragma once

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <torch/extension.h>

#include <vector>

#include "../cell_complex.hxx"
#include "./arcs_geometry_struct.hxx"

namespace cpu {

template <typename Workspace>
void trace_raw_arcs_geometry(Workspace& ws) {
  RECORD_FUNCTION("trace_raw_arcs_geometry_flat", {});
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  const auto& fast_crit_map = ws.hlp.fast_crit_map;
  auto& min_arcs_len = ws.arcs_topology.min_arcs_len;
  auto& max_arcs_len = ws.arcs_topology.max_arcs_len;
  const auto& gradient_data = ws.gradient_data;
  const auto& paired_with = gradient_data.paired_with;
  const auto& crit_saddles = gradient_data.cp.saddles;

  auto& saddle_nodes = ws.saddle_nodes;
  // SaddleNodes saddle_nodes;
  int num_saddles = crit_saddles.size();
  if (num_saddles == 0) return;

  saddle_nodes.nodes.resize(num_saddles);

  // There are 2 arcs per saddle
  std::vector<int> max_offsets(num_saddles * 2 + 1, 0);
  std::vector<int> min_offsets(num_saddles * 2 + 1, 0);

  for (int i = 0; i < num_saddles * 2; ++i) {
    max_offsets[i + 1] = max_offsets[i] + max_arcs_len[i] + 1;
    min_offsets[i + 1] = min_offsets[i] + min_arcs_len[i] + 1;
  }

  auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto flat_max_geom = saddle_nodes.flat_max_geom.request({(long)max_offsets.back()}, cpu_int_opts);
  auto flat_min_geom = saddle_nodes.flat_min_geom.request({(long)min_offsets.back()}, cpu_int_opts);

  // saddle_nodes.flat_max_geom = torch::empty({(long)max_offsets.back()}, cpu_int_opts);
  // saddle_nodes.flat_min_geom = torch::empty({(long)min_offsets.back()}, cpu_int_opts);

  int* flat_max_ptr = flat_max_geom.template data_ptr<int>();
  int* flat_min_ptr = flat_min_geom.template data_ptr<int>();
  SaddleNode* nodes_ptr = saddle_nodes.nodes.data();

  at::parallel_for(0, num_saddles, 256, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      nodes_ptr[i].alive = true;
      int s_id = crit_saddles[i];
      int ey = get_y(s_id, Nx), ex = get_x(s_id, Nx), type = get_type(s_id);

      // ----------------------------
      // MAX ARCS (Faces)
      // ----------------------------
      int f1 = -1, f2 = -1;
      if (type == 1) {
        if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
        if (is_valid_f(ey + 1, ex, H, W)) f2 = cell_id(3, ey + 1, ex, Nx);
      } else {
        if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
        if (is_valid_f(ey, ex + 1, H, W)) f2 = cell_id(3, ey, ex + 1, Nx);
      }

      auto trace_geometry_face = [&](int start_face, int arc_idx) {
        int curr = start_face;
        int base_offset = max_offsets[2 * i + arc_idx];
        int count = 0;

        while (curr != -1) {
          if (get_type(curr) == 3) {
            flat_max_ptr[base_offset + count] = curr;
            count++;
          }
          if (paired_with[curr] == -1) break;

          int edge = paired_with[curr];
          int n_ey = get_y(edge, Nx), n_ex = get_x(edge, Nx);
          int nf1 = -1, nf2 = -1;

          if (get_type(edge) == 1) {
            if (is_valid_f(n_ey, n_ex, H, W)) nf1 = cell_id(3, n_ey, n_ex, Nx);
            if (is_valid_f(n_ey + 1, n_ex, H, W)) nf2 = cell_id(3, n_ey + 1, n_ex, Nx);
          } else {
            if (is_valid_f(n_ey, n_ex, H, W)) nf1 = cell_id(3, n_ey, n_ex, Nx);
            if (is_valid_f(n_ey, n_ex + 1, H, W)) nf2 = cell_id(3, n_ey, n_ex + 1, Nx);
          }
          curr = (nf1 == curr) ? nf2 : nf1;
        }

        int target_idx = (curr != -1) ? fast_crit_map[curr] : -1;
        nodes_ptr[i].max_arcs[arc_idx] = {target_idx, base_offset, count};
      };

      int valid_max_count = 0;
      if (f1 != -1) trace_geometry_face(f1, valid_max_count++);
      if (f2 != -1) trace_geometry_face(f2, valid_max_count++);

      // ----------------------------
      // MIN ARCS (Vertices)
      // ----------------------------
      int v1 = -1, v2 = -1;
      if (type == 1) {
        if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, Nx);
        if (ex < W) v2 = cell_id(0, ey, ex, Nx);
      } else {
        if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, Nx);
        if (ey < H) v2 = cell_id(0, ey, ex, Nx);
      }

      auto trace_geometry_vertex = [&](int start_v, int arc_idx) {
        int curr = start_v;
        int base_offset = min_offsets[2 * i + arc_idx];
        int count = 0;

        while (curr != -1) {
          if (get_type(curr) == 0) {
            flat_min_ptr[base_offset + count] = curr;
            count++;
          }
          if (paired_with[curr] == -1) break;

          int edge = paired_with[curr];
          int n_ey = get_y(edge, Nx), n_ex = get_x(edge, Nx);
          int nv1 = -1, nv2 = -1;

          if (get_type(edge) == 1) {
            if (n_ex - 1 >= 0) nv1 = cell_id(0, n_ey, n_ex - 1, Nx);
            if (n_ex < W) nv2 = cell_id(0, n_ey, n_ex, Nx);
          } else {
            if (n_ey - 1 >= 0) nv1 = cell_id(0, n_ey - 1, n_ex, Nx);
            if (n_ey < H) nv2 = cell_id(0, n_ey, n_ex, Nx);
          }
          curr = (nv1 == curr) ? nv2 : nv1;
        }

        int target_idx = (curr != -1) ? fast_crit_map[curr] : -1;
        nodes_ptr[i].min_arcs[arc_idx] = {target_idx, base_offset, count};
      };

      int valid_min_count = 0;
      if (v1 != -1) trace_geometry_vertex(v1, valid_min_count++);
      if (v2 != -1) trace_geometry_vertex(v2, valid_min_count++);
    }
  });
}
}  // namespace cpu