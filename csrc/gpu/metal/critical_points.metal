#include <metal_atomic>
#include <metal_stdlib>
using namespace metal;

struct Constants {
  int H;
  int W;
  int Nx;
};

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

kernel void extract_critical_points(device const int* paired_with [[buffer(0)]], device int* out_vertices [[buffer(1)]],
                                    device int* out_edges [[buffer(2)]], device int* out_faces [[buffer(3)]],
                                    device atomic_uint* counters [[buffer(4)]], constant Constants& p [[buffer(5)]],
                                    uint id [[thread_position_in_grid]], ushort lane_id [[thread_index_in_simdgroup]]) {
  int num_cells = 4 * (p.H + 1) * (p.W + 1);

  bool is_vertex = false;
  bool is_edge = false;
  bool is_face = false;

  if (id < (uint)num_cells && paired_with[id] == -1) {
    int type = get_type(id);
    int y = get_y(id, p.Nx);
    int x = get_x(id, p.Nx);

    if (type == 0 && is_valid_v(y, x, p.H, p.W))
      is_vertex = true;
    else if ((type == 1 && is_valid_ehx(y, x, p.H, p.W)) || (type == 2 && is_valid_evy(y, x, p.H, p.W)))
      is_edge = true;
    else if (type == 3 && is_valid_f(y, x, p.H, p.W))
      is_face = true;
  }

  // --- VERTICES ---
  uint v_val = is_vertex ? 1 : 0;
  uint v_offset = simd_prefix_exclusive_sum(v_val);
  uint v_total = simd_sum(v_val);
  uint v_base = 0;

  if (lane_id == 0 && v_total > 0) {
    v_base = atomic_fetch_add_explicit(&counters[0], v_total, memory_order_relaxed);
  }
  v_base = simd_broadcast(v_base, 0);
  if (is_vertex) out_vertices[v_base + v_offset] = id;

  // --- EDGES ---
  uint e_val = is_edge ? 1 : 0;
  uint e_offset = simd_prefix_exclusive_sum(e_val);
  uint e_total = simd_sum(e_val);
  uint e_base = 0;

  if (lane_id == 0 && e_total > 0) {
    e_base = atomic_fetch_add_explicit(&counters[1], e_total, memory_order_relaxed);
  }
  e_base = simd_broadcast(e_base, 0);
  if (is_edge) out_edges[e_base + e_offset] = id;

  // --- FACES ---
  uint f_val = is_face ? 1 : 0;
  uint f_offset = simd_prefix_exclusive_sum(f_val);
  uint f_total = simd_sum(f_val);
  uint f_base = 0;

  if (lane_id == 0 && f_total > 0) {
    f_base = atomic_fetch_add_explicit(&counters[2], f_total, memory_order_relaxed);
  }
  f_base = simd_broadcast(f_base, 0);
  if (is_face) out_faces[f_base + f_offset] = id;
}