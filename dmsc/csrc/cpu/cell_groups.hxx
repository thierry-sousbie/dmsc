#include <ATen/Parallel.h>
#include <tbb/parallel_invoke.h>
#include <torch/extension.h>

#include "../union_find.hxx"
#include "./cell_groups_struct.hxx"

#define CACHE_IS_EMPTY -2
#define CACHE_DEPTH 16

namespace cpu {

inline int trace_faces(int start_face, const int* paired_with, int H, int W, int Nx, int* out_groups, int* cached_value,
                       int* path_buffer) {
  if (start_face == -1) return -1;
  int curr = start_face;
  int step_count = 0;

  while (curr != -1 && paired_with[curr] != -1) {
    int cy = get_y(curr, Nx);
    int cx = get_x(curr, Nx);
    int flat_idx = cy * Nx + cx;

    // Opportunistic cache check via volatile pointer bypass
    int val = ((volatile int*)out_groups)[flat_idx];
    if (val != CACHE_IS_EMPTY) {
      *cached_value = val;
      if (step_count < CACHE_DEPTH) path_buffer[step_count] = CACHE_IS_EMPTY;
      return curr;
    }

    if (step_count < CACHE_DEPTH) {
      path_buffer[step_count] = flat_idx;
      step_count++;
    }

    int edge = paired_with[curr];
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

  if (step_count < CACHE_DEPTH) path_buffer[step_count] = CACHE_IS_EMPTY;
  return curr;
}

inline int trace_vertices(int start_v, const int* paired_with, int H, int W, int Nx, int* out_groups, int* cached_value,
                          int* path_buffer) {
  if (start_v == -1) return -1;
  int curr = start_v;
  int step_count = 0;

  while (curr != -1 && paired_with[curr] != -1) {
    int cy = get_y(curr, Nx);
    int cx = get_x(curr, Nx);
    int flat_idx = cy * W + cx;

    int val = ((volatile int*)out_groups)[flat_idx];
    if (val != CACHE_IS_EMPTY) {
      *cached_value = val;
      if (step_count < CACHE_DEPTH) path_buffer[step_count] = CACHE_IS_EMPTY;
      return curr;
    }

    if (step_count < CACHE_DEPTH) {
      path_buffer[step_count] = flat_idx;
      step_count++;
    }

    int edge = paired_with[curr];
    int ey = get_y(edge, Nx), ex = get_x(edge, Nx);
    int v1 = -1, v2 = -1;

    if (get_type(edge) == 1) {
      if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, Nx);
      if (ex < W) v2 = cell_id(0, ey, ex, Nx);
    } else {
      if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, Nx);
      if (ey < H) v2 = cell_id(0, ey, ex, Nx);
    }
    curr = (v1 == curr) ? v2 : v1;
  }

  if (step_count < CACHE_DEPTH) path_buffer[step_count] = CACHE_IS_EMPTY;
  return curr;
}

template <typename Workspace>
void compute_cell_groups(Workspace& ws, bool trace_face_groups, bool trace_vertex_groups) {
  RECORD_FUNCTION("cell_groups_cpu", {});
  int H = ws.H;
  int W = ws.W;
  const auto& paired_with = ws.gradient_data.paired_with;
  const auto& fast_crit_map = ws.hlp.fast_crit_map;
  const auto& crit_maxes = ws.gradient_data.cp.maxes;
  const auto& crit_mins = ws.gradient_data.cp.mins;

  auto& uf_max = ws.p_data.uf_max;
  auto& uf_min = ws.p_data.uf_min;
  const auto& max_alive = ws.p_data.max_alive;
  const auto& min_alive = ws.p_data.min_alive;

  int Nx = W + 1;
  int num_faces = (H + 1) * (W + 1);
  int num_vertices = H * W;

  auto i_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  torch::Tensor vertex_groups = ws.cell_groups.vertex_groups.request_full({H, W}, i_opts, CACHE_IS_EMPTY);
  torch::Tensor face_groups = ws.cell_groups.face_groups.request_full({H + 1, W + 1}, i_opts, CACHE_IS_EMPTY);

  auto face_groups_ptr = face_groups.data_ptr<int>();
  auto vertex_groups_ptr = vertex_groups.data_ptr<int>();

  // Assumes gradient_data holds the gradient pairs
  const int* paired_with_ptr = &paired_with[0];
  std::vector<int> fast_region_id(paired_with.size(), -1);
  {
    RECORD_FUNCTION("cell_groups_preproc_cpu", {});
    // flatten union_find structures and build fast_region_id
    tbb::parallel_invoke(
        [&]() {
          for (size_t i = 0; i < uf_max.parent.size(); ++i) uf_max.find(i);
        },
        [&]() {
          for (size_t i = 0; i < uf_min.parent.size(); ++i) uf_min.find(i);
        },
        [&]() {
          int region_counter = 0;
          for (size_t i = 0; i < crit_maxes.size(); ++i) {
            if (max_alive[i]) fast_region_id[crit_maxes[i]] = region_counter++;
          }
        },
        [&]() {
          int region_counter = 0;
          for (size_t i = 0; i < crit_mins.size(); ++i) {
            if (min_alive[i]) fast_region_id[crit_mins[i]] = region_counter++;
          }
        });
  }
  const auto uf_max_parent = uf_max.parent;
  const auto uf_min_parent = uf_min.parent;
  // Compute Face Groups
  if (trace_face_groups) {
    RECORD_FUNCTION("trace_faces_cpu", {});
    at::parallel_for(0, num_faces, 1024, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        int y = i / Nx;
        int x = i % Nx;

        int start_face = cell_id(3, y, x, Nx);
        int final_val = -1;
        int cached_value = CACHE_IS_EMPTY;
        int path_buffer[CACHE_DEPTH];

        int initial_max =
            trace_faces(start_face, paired_with_ptr, H, W, Nx, face_groups_ptr, &cached_value, path_buffer);

        if (cached_value != CACHE_IS_EMPTY) {
          final_val = cached_value;
        } else if (initial_max != -1 && fast_crit_map[initial_max] != -1) {
          // Relies on pre-flattened union-find parent array
          int root_idx = uf_max_parent[fast_crit_map[initial_max]];
          if (root_idx != -1) {
            final_val = fast_region_id[crit_maxes[root_idx]];
          }
        }

        face_groups_ptr[i] = final_val;

        // Path Compression
        if (final_val != -1) {
          for (int k = 0; k < CACHE_DEPTH; ++k) {
            int flat_idx = path_buffer[k];
            if (flat_idx == CACHE_IS_EMPTY) break;
            face_groups_ptr[flat_idx] = final_val;
          }
        }
      }
    });
  }

  // Compute Vertex Groups
  if (trace_vertex_groups) {
    RECORD_FUNCTION("trace_vertices_cpu", {});
    at::parallel_for(0, num_vertices, 1024, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        int y = i / W;
        int x = i % W;

        int start_vertex = cell_id(0, y, x, Nx);
        int final_val = -1;
        int cached_value = CACHE_IS_EMPTY;
        int path_buffer[CACHE_DEPTH];

        int initial_min =
            trace_vertices(start_vertex, paired_with_ptr, H, W, Nx, vertex_groups_ptr, &cached_value, path_buffer);

        if (cached_value != CACHE_IS_EMPTY) {
          final_val = cached_value;
        } else if (initial_min != -1 && fast_crit_map[initial_min] != -1) {
          // Relies on pre-flattened union-find parent array
          int root_idx = uf_min_parent[fast_crit_map[initial_min]];
          if (root_idx != -1) {
            final_val = fast_region_id[crit_mins[root_idx]];
          }
        }

        vertex_groups_ptr[i] = final_val;

        // Path Compression
        if (final_val != -1) {
          for (int k = 0; k < CACHE_DEPTH; ++k) {
            int flat_idx = path_buffer[k];
            if (flat_idx == CACHE_IS_EMPTY) break;
            vertex_groups_ptr[flat_idx] = final_val;
          }
        }
      }
    });
  }
}

}  // namespace cpu