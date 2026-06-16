#include <metal_stdlib>
using namespace metal;

#define CACHE_IS_EMPTY -2
#define CACHE_DEPTH 16  // Number of steps to record for path compression

struct Constants {
  int H;
  int W;
  int Nx;
  int padding;
};

inline int cell_id(int type, int y, int x, int Nx) {
  return type + 4 * (y * Nx + x);
}

inline int get_type(int id) {
  return id % 4;
}

inline int get_y(int id, int Nx) {
  return (id / 4) / Nx;
}

inline int get_x(int id, int Nx) {
  return (id / 4) % Nx;
}

inline bool is_valid_v(int y, int x, int H, int W) {
  return y >= 0 && y < H && x >= 0 && x < W;
}

inline bool is_valid_ehx(int y, int x, int H, int W) {
  return y >= 0 && y < H && x >= 0 && x <= W;
}

inline bool is_valid_evy(int y, int x, int H, int W) {
  return y >= 0 && y <= H && x >= 0 && x < W;
}

inline bool is_valid_f(int y, int x, int H, int W) {
  return y >= 0 && y <= H && x >= 0 && x <= W;
}

// Used to memoize naturally the faces we already computed without requiring any sync
inline bool check_cached(int curr, int Nx, int row_stride, device int* out_groups, thread int* cached_value) {
  int cy = get_y(curr, Nx);
  int cx = get_x(curr, Nx);
  int flat_idx = cy * row_stride + cx;

  // Volatile bypasses the L1/register cache in MSL just like in CUDA
  *cached_value = ((volatile device int*)out_groups)[flat_idx];

  if (*cached_value != CACHE_IS_EMPTY) return true;

  return false;
}

inline int trace_faces(int start_face, device const int* gradient_pairs, int H, int W, int Nx, device int* out_groups,
                       thread int* cached_value, thread int* path_buffer) {
  if (start_face == -1) return -1;
  int curr = start_face;
  int step_count = 0;

  while (curr != -1 && gradient_pairs[curr] != -1) {
    if (check_cached(curr, Nx, Nx, out_groups, cached_value)) {
      if (step_count < CACHE_DEPTH) path_buffer[step_count] = CACHE_IS_EMPTY;
      return curr;
    }

    // Record the flat index into the local register buffer
    if (step_count < CACHE_DEPTH) {
      int cy = get_y(curr, Nx);
      int cx = get_x(curr, Nx);
      path_buffer[step_count] = cy * Nx + cx;
      step_count++;
    }

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

  if (step_count < CACHE_DEPTH) path_buffer[step_count] = CACHE_IS_EMPTY;
  return curr;
}

inline int trace_vertices(int start_v, device const int* gradient_pairs, int H, int W, int Nx, device int* out_groups,
                          thread int* cached_value, thread int* path_buffer) {
  if (start_v == -1) return -1;
  int curr = start_v;
  int step_count = 0;

  while (curr != -1 && gradient_pairs[curr] != -1) {
    if (check_cached(curr, Nx, W, out_groups, cached_value)) {
      if (step_count < CACHE_DEPTH) path_buffer[step_count] = CACHE_IS_EMPTY;
      return curr;
    }

    // Record the flat index into the local register buffer
    if (step_count < CACHE_DEPTH) {
      int cy = get_y(curr, Nx);
      int cx = get_x(curr, Nx);
      path_buffer[step_count] = cy * W + cx;
      step_count++;
    }

    int edge = gradient_pairs[curr];
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

template <bool IS_DUAL>
inline void compute_face_groups_impl(device const int* paired_with, device const int* fast_crit_map,
                                     device const int* uf_max_parent, device const int* crit_maxes,
                                     device const int* fast_region_id, device int* out_groups,
                                     device atomic_uint* global_index, constant Constants& p) {
  uint num_faces = (uint)((p.H + 1) * (p.W + 1));

  while (true) {
    uint id = atomic_fetch_add_explicit(global_index, 1, memory_order_relaxed);
    if (id >= num_faces) break;

    int y = id / p.Nx;
    int x = id % p.Nx;

    int start_face = cell_id(3, y, x, p.Nx);
    int final_val = -1;

    int cached_value = CACHE_IS_EMPTY;
    int path_buffer[CACHE_DEPTH];

    int initial_max = trace_faces(start_face, paired_with, p.H, p.W, p.Nx, out_groups, &cached_value, path_buffer);

    if (cached_value != CACHE_IS_EMPTY) {
      final_val = cached_value;
      out_groups[id] = final_val;
    } else {
      if (initial_max != -1 && fast_crit_map[initial_max] != -1) {
        int root_idx = uf_max_parent[fast_crit_map[initial_max]];
        if (root_idx != -1) {
          int final_max = crit_maxes[root_idx];
          final_val = fast_region_id[final_max];
        }
      }
      out_groups[id] = final_val;
    }

    // Write back the final value to all visited nodes
    if (final_val != -1) {
      for (int i = 0; i < CACHE_DEPTH; ++i) {
        int flat_idx = path_buffer[i];
        if (flat_idx == CACHE_IS_EMPTY) break;
        out_groups[flat_idx] = final_val;
      }
    }
  }
}

template <bool IS_DUAL>
inline void compute_vertex_groups_impl(device const int* paired_with, device const int* fast_crit_map,
                                       device const int* uf_min_parent, device const int* crit_mins,
                                       device const int* fast_region_id, device int* out_groups,
                                       device atomic_uint* global_index, constant Constants& p) {
  uint num_vertices = (uint)(p.H * p.W);

  while (true) {
    uint id = atomic_fetch_add_explicit(global_index, 1, memory_order_relaxed);
    if (id >= num_vertices) break;

    int y = id / p.W;
    int x = id % p.W;

    int start_vertex = cell_id(0, y, x, p.Nx);
    int final_val = -1;

    int cached_value = CACHE_IS_EMPTY;
    int path_buffer[CACHE_DEPTH];

    int initial_min = trace_vertices(start_vertex, paired_with, p.H, p.W, p.Nx, out_groups, &cached_value, path_buffer);

    if (cached_value != CACHE_IS_EMPTY) {
      final_val = cached_value;
      out_groups[id] = final_val;
    } else {
      if (initial_min != -1 && fast_crit_map[initial_min] != -1) {
        int root_idx = uf_min_parent[fast_crit_map[initial_min]];
        if (root_idx != -1) {
          int final_min = crit_mins[root_idx];
          final_val = fast_region_id[final_min];
        }
      }
      out_groups[id] = final_val;
    }

    // Write back the final value to all visited nodes
    if (final_val != -1) {
      for (int i = 0; i < CACHE_DEPTH; ++i) {
        int flat_idx = path_buffer[i];
        if (flat_idx == CACHE_IS_EMPTY) break;
        out_groups[flat_idx] = final_val;
      }
    }
  }
}

kernel void compute_face_groups_primal(device const float* data [[buffer(0)]],
                                       device const int* paired_with [[buffer(1)]],
                                       device const int* fast_crit_map [[buffer(2)]],
                                       device const int* uf_max_parent [[buffer(3)]],
                                       device const int* crit_maxes [[buffer(4)]],
                                       device const int* fast_region_id [[buffer(5)]],
                                       device int* out_groups [[buffer(6)]], constant Constants& p [[buffer(7)]],
                                       device atomic_uint* global_index [[buffer(8)]]) {
  compute_face_groups_impl<false>(paired_with, fast_crit_map, uf_max_parent, crit_maxes, fast_region_id, out_groups,
                                  global_index, p);
}

kernel void compute_face_groups_dual(device const float* data [[buffer(0)]],
                                     device const int* paired_with [[buffer(1)]],
                                     device const int* fast_crit_map [[buffer(2)]],
                                     device const int* uf_max_parent [[buffer(3)]],
                                     device const int* crit_maxes [[buffer(4)]],
                                     device const int* fast_region_id [[buffer(5)]],
                                     device int* out_groups [[buffer(6)]], constant Constants& p [[buffer(7)]],
                                     device atomic_uint* global_index [[buffer(8)]]) {
  compute_face_groups_impl<true>(paired_with, fast_crit_map, uf_max_parent, crit_maxes, fast_region_id, out_groups,
                                 global_index, p);
}

kernel void compute_vertex_groups_primal(device const float* data [[buffer(0)]],
                                         device const int* paired_with [[buffer(1)]],
                                         device const int* fast_crit_map [[buffer(2)]],
                                         device const int* uf_min_parent [[buffer(3)]],
                                         device const int* crit_mins [[buffer(4)]],
                                         device const int* fast_region_id [[buffer(5)]],
                                         device int* out_groups [[buffer(6)]], constant Constants& p [[buffer(7)]],
                                         device atomic_uint* global_index [[buffer(8)]]) {
  compute_vertex_groups_impl<false>(paired_with, fast_crit_map, uf_min_parent, crit_mins, fast_region_id, out_groups,
                                    global_index, p);
}

kernel void compute_vertex_groups_dual(device const float* data [[buffer(0)]],
                                       device const int* paired_with [[buffer(1)]],
                                       device const int* fast_crit_map [[buffer(2)]],
                                       device const int* uf_min_parent [[buffer(3)]],
                                       device const int* crit_mins [[buffer(4)]],
                                       device const int* fast_region_id [[buffer(5)]],
                                       device int* out_groups [[buffer(6)]], constant Constants& p [[buffer(7)]],
                                       device atomic_uint* global_index [[buffer(8)]]) {
  compute_vertex_groups_impl<true>(paired_with, fast_crit_map, uf_min_parent, crit_mins, fast_region_id, out_groups,
                                   global_index, p);
}