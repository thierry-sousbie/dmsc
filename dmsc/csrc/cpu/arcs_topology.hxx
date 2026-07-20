#pragma once

#include <tbb/parallel_sort.h>
#include <torch/extension.h>

#include <vector>

#include "../cell_complex.hxx"
#include "./arcs_topology_struct.hxx"
#include "./cell_compare.hxx"

namespace cpu {

struct TraceRes {
  int target;
  int length;
};

template <typename Container>
TraceRes trace_faces_with_length(const Container& gradient_pairs, int start_face, int Nx, int H, int W,
                                 int max_steps = std::numeric_limits<int>::max()) {
  if (start_face == -1) return {-1, 0};
  int curr = start_face, steps = 0;
  while (curr != -1 && gradient_pairs[curr] != -1 && steps < max_steps) {
    steps++;
    int edge = gradient_pairs[curr];
    int ey = get_y(edge, Nx), ex = get_x(edge, Nx);
    int f1 = -1, f2 = -1;
    if (get_type(edge) == 1) {
      if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
      if (is_valid_f(ey + 1, ex, H, W)) f2 = cell_id(3, ey + 1, ex, Nx);
    } else {
      if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
      if (is_valid_f(ey, ex + 1, H, W)) f2 = cell_id(3, ey, ex + 1, Nx);
    }
    curr = (f1 == curr) ? f2 : f1;
  }
  return {curr, steps};
}

template <typename Container>
TraceRes trace_vertices_with_length(const Container& gradient_pairs, int start_v, int Nx, int H, int W,
                                    int max_steps = std::numeric_limits<int>::max()) {
  if (start_v == -1) return {-1, 0};
  int curr = start_v, steps = 0;
  while (curr != -1 && gradient_pairs[curr] != -1 && steps < max_steps) {
    steps++;
    int edge = gradient_pairs[curr];
    int ey = get_y(edge, Nx), ex = get_x(edge, Nx);
    int v1 = -1, v2 = -1;
    if (get_type(edge) == 1) {  // H-edge
      // if (is_valid_v(ey, ex - 1, Nx)) v1 = cell_id(0, ey, ex - 1, Nx);
      // if (is_valid_v(ey, ex, Nx)) v1 = cell_id(0, ey, ex, Nx);
      if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, Nx);
      if (ex < W) v2 = cell_id(0, ey, ex, Nx);
    } else {  // V-Edge
      // if (is_valid_v(ey - 1, ex, Nx)) v1 = cell_id(0, ey - 1, ex, Nx);
      // if (is_valid_v(ey, ex, Nx)) v1 = cell_id(0, ey, ex, Nx);
      if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, Nx);
      if (ey < H) v2 = cell_id(0, ey, ex, Nx);
    }
    curr = (v1 == curr) ? v2 : v1;
  }
  return {curr, steps};
}

template <bool IS_DUAL, typename Workspace, typename scalar_t = float>
void trace_from_saddles(Workspace& ws, const torch::Tensor& scalar_field) {
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  auto& arcs_topology = ws.arcs_topology;
  const auto& paired_with = ws.gradient_data.paired_with;
  const auto& crit_saddles = ws.gradient_data.cp.saddles;
  const auto& fast_crit_map = ws.hlp.fast_crit_map;
  const scalar_t* data = scalar_field.data_ptr<scalar_t>();

  int num_saddles = crit_saddles.size();

  arcs_topology.sorted_max_saddles.resize(num_saddles);
  arcs_topology.sorted_min_saddles.resize(num_saddles);
  arcs_topology.max_arcs_len.assign(num_saddles * 2, 0);
  arcs_topology.min_arcs_len.assign(num_saddles * 2, 0);

  std::vector<size_t> sorted_max_indices(num_saddles);
  std::vector<scalar_t> saddle_value(num_saddles);

  RECORD_FUNCTION("trace_from_saddles", {});
  {
    RECORD_FUNCTION("Sort_by_index", {});
    at::parallel_for(0, num_saddles, 256, [&](int64_t start, int64_t end) {
      for (int64_t j = start; j < end; ++j) {
        int s_id = crit_saddles[j];
        int ey = get_y(s_id, Nx), ex = get_x(s_id, Nx);
        int type = get_type(s_id);
        saddle_value[j] = cell_value<IS_DUAL>(type, ey, ex, H, W, data);
      }
    });
    std::iota(sorted_max_indices.begin(), sorted_max_indices.end(), 0);
    tbb::parallel_sort(sorted_max_indices.begin(), sorted_max_indices.end(), [&](size_t i, size_t j) {
      return value_greater<IS_DUAL>(saddle_value[i], crit_saddles[i], saddle_value[j], crit_saddles[j]);
    });
  }

  at::parallel_for(0, num_saddles, 256, [&](int64_t start, int64_t end) {
    for (int64_t j = start; j < end; ++j) {
      // Maxima
      int max_id = sorted_max_indices[j];
      int s_id = crit_saddles[max_id];
      int ey = get_y(s_id, Nx), ex = get_x(s_id, Nx);
      int type = get_type(s_id);
      scalar_t v_s = saddle_value[max_id];  // cell_value<IS_DUAL>(type, ey, ex, H, W, data);

      int f1 = -1, f2 = -1;
      if (type == 1) {
        if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
        if (is_valid_f(ey + 1, ex, H, W)) f2 = cell_id(3, ey + 1, ex, Nx);
      } else {
        if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
        if (is_valid_f(ey, ex + 1, H, W)) f2 = cell_id(3, ey, ex + 1, Nx);
      }
      TraceRes max1 = (f1 != -1) ? trace_faces_with_length(paired_with.data(), f1, Nx, H, W) : TraceRes{-1, 0};
      TraceRes max2 = (f2 != -1) ? trace_faces_with_length(paired_with.data(), f2, Nx, H, W) : TraceRes{-1, 0};

      arcs_topology.sorted_max_saddles[j] = {-999, -1, -1, 0.0f};
      if (max1.target != -1 && max2.target != -1) {
        arcs_topology.sorted_max_saddles[j] = {
            s_id, fast_crit_map[max1.target], fast_crit_map[max2.target], v_s};
        arcs_topology.max_arcs_len[2 * max_id] = max1.length;
        arcs_topology.max_arcs_len[2 * max_id + 1] = max2.length;
      } else if ((max1.target != -1 && max2.target == -1) || (max1.target == -1 && max2.target != -1)) {
        int max_real = (max1.target != -1) ? max1.target : max2.target;
        arcs_topology.sorted_max_saddles[j] = {s_id, fast_crit_map[max_real], -1, v_s};
        arcs_topology.max_arcs_len[2 * max_id] = (max1.target != -1) ? max1.length : max2.length;
      }
    }
  });

  at::parallel_for(0, num_saddles, 256, [&](int64_t start, int64_t end) {
    for (int64_t j = start; j < end; ++j) {
      // Minima
      int min_id = sorted_max_indices[num_saddles - j - 1];
      int s_id = crit_saddles[min_id];
      int ey = get_y(s_id, Nx), ex = get_x(s_id, Nx);
      int type = get_type(s_id);
      scalar_t v_s = saddle_value[min_id];  // cell_value<IS_DUAL>(type, ey, ex, H, W, data);

      int v1 = -1, v2 = -1;
      if (type == 1) {
        if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, Nx);
        if (ex < W) v2 = cell_id(0, ey, ex, Nx);
      } else {
        if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, Nx);
        if (ey < H) v2 = cell_id(0, ey, ex, Nx);
      }
      TraceRes min1 = (v1 != -1) ? trace_vertices_with_length(paired_with.data(), v1, Nx, H, W) : TraceRes{-1, 0};
      TraceRes min2 = (v2 != -1) ? trace_vertices_with_length(paired_with.data(), v2, Nx, H, W) : TraceRes{-1, 0};

      arcs_topology.sorted_min_saddles[j] = {-999, -1, -1, 0.0f};
      if (min1.target != -1 && min2.target != -1) {
        arcs_topology.sorted_min_saddles[j] = {
            s_id, fast_crit_map[min1.target], fast_crit_map[min2.target], v_s};
        arcs_topology.min_arcs_len[2 * min_id] = min1.length;
        arcs_topology.min_arcs_len[2 * min_id + 1] = min2.length;
      } else if ((min1.target != -1 && min2.target == -1) || (min1.target == -1 && min2.target != -1)) {
        int min_real = (min1.target != -1) ? min1.target : min2.target;
        arcs_topology.sorted_min_saddles[j] = {s_id, fast_crit_map[min_real], -1, v_s};
        arcs_topology.min_arcs_len[2 * min_id] = (min1.target != -1) ? min1.length : min2.length;
      }
    }
  });
}

}  // namespace cpu
