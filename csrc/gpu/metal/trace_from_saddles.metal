#include <metal_stdlib>
using namespace metal;

struct Constants {
  int H;
  int W;
  int Nx;
  int num_saddles;
};

struct Coord {
  int y;
  int x;
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

template <bool IS_DUAL = false>
inline bool value_greater(float val1, int id1, float val2, int id2) {
  if (IS_DUAL) {
    if (val1 != val2)
      return val1 < val2;
    else
      return id1 < id2;
  } else {
    if (val1 != val2)
      return val1 > val2;
    else
      return id1 > id2;
  }
}

template <bool IS_DUAL = false>
inline bool v_greater(int y1, int x1, int y2, int x2, device const float* data, int W) {
  int id1 = y1 * W + x1;
  int id2 = y2 * W + x2;
  return value_greater<IS_DUAL>(data[id1], id1, data[id2], id2);
}

template <bool IS_DUAL = false>
inline Coord _check_and_update(int vy, int vx, Coord best, device const float* data, int W) {
  if (best.y == -1) {
    return {vy, vx};
  } else if (v_greater<IS_DUAL>(vy, vx, best.y, best.x, data, W)) {
    return {vy, vx};
  }
  return best;
}

template <bool IS_DUAL = false>
inline Coord get_highest_vertex(int type, int y, int x, int H, int W, device const float* data) {
  Coord best = {-1, -1};

  if (type == 0) {
    best = _check_and_update<IS_DUAL>(y, x, best, data, W);
  } else if (type == 1) {
    if (x - 1 >= 0) best = _check_and_update<IS_DUAL>(y, x - 1, best, data, W);
    if (x < W) best = _check_and_update<IS_DUAL>(y, x, best, data, W);
  } else if (type == 2) {
    if (y - 1 >= 0) best = _check_and_update<IS_DUAL>(y - 1, x, best, data, W);
    if (y < H) best = _check_and_update<IS_DUAL>(y, x, best, data, W);
  } else if (type == 3) {
    if (y - 1 >= 0 && x - 1 >= 0) best = _check_and_update<IS_DUAL>(y - 1, x - 1, best, data, W);
    if (y - 1 >= 0 && x < W) best = _check_and_update<IS_DUAL>(y - 1, x, best, data, W);
    if (y < H && x - 1 >= 0) best = _check_and_update<IS_DUAL>(y, x - 1, best, data, W);
    if (y < H && x < W) best = _check_and_update<IS_DUAL>(y, x, best, data, W);
  }

  return best;
}

template <bool IS_DUAL = false>
inline float cell_value(int type, int y, int x, int H, int W, device const float* data) {
  Coord mv = get_highest_vertex<IS_DUAL>(type, y, x, H, W, data);
  return data[mv.y * W + mv.x];
}

struct TraceRes {
  int target;
  int length;
};

TraceRes trace_faces_with_length(int start_face, device const int* gradient_pairs, int Nx, int H, int W) {
  if (start_face == -1) return {-1, 0};
  int curr = start_face, steps = 0;
  while (curr != -1 && gradient_pairs[curr] != -1) {
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

TraceRes trace_vertices_with_length(int start_v, device const int* gradient_pairs, int Nx, int H, int W) {
  if (start_v == -1) return {-1, 0};
  int curr = start_v, steps = 0;
  while (curr != -1 && gradient_pairs[curr] != -1) {
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

// inline int trace_faces(int start_face, device const int* gradient_pairs, int H, int W, int Nx) {
//   if (start_face == -1) return -1;
//   int curr = start_face;
//   while (curr != -1 && gradient_pairs[curr] != -1) {
//     int edge = gradient_pairs[curr];
//     int ey = get_y(edge, Nx), ex = get_x(edge, Nx);
//     int f1 = -1, f2 = -1;
//     if (get_type(edge) == 1) {
//       if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
//       if (is_valid_f(ey + 1, ex, H, W)) f2 = cell_id(3, ey + 1, ex, Nx);
//     } else {
//       if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
//       if (is_valid_f(ey, ex + 1, H, W)) f2 = cell_id(3, ey, ex + 1, Nx);
//     }
//     curr = (f1 == curr) ? f2 : f1;
//   }
//   return curr;
// }

// inline int trace_vertices(int start_v, device const int* gradient_pairs, int H, int W, int Nx) {
//   if (start_v == -1) return -1;
//   int curr = start_v;
//   while (curr != -1 && gradient_pairs[curr] != -1) {
//     int edge = gradient_pairs[curr];
//     int ey = get_y(edge, Nx), ex = get_x(edge, Nx);
//     int v1 = -1, v2 = -1;
//     if (get_type(edge) == 1) {
//       if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, Nx);
//       if (ex < W) v2 = cell_id(0, ey, ex, Nx);
//     } else {
//       if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, Nx);
//       if (ey < H) v2 = cell_id(0, ey, ex, Nx);
//     }
//     curr = (v1 == curr) ? v2 : v1;
//   }
//   return curr;
// }

// THE MEGAKERNEL IMPLEMENTATION
template <bool IS_DUAL>
inline void trace_from_saddles_impl(device const float* data, device const int* paired_with, device const int* saddles,
                                    device int* max_c1, device int* max_c2, device int* min_c1, device int* min_c2,
                                    device float* out_s_vals, device int* out_max_len, device int* out_min_len,
                                    device atomic_uint* global_saddle_index,  // Added atomic counter
                                    constant Constants& p) {
  // Infinite loop allows threads to continuously fetch new work
  while (true) {
    // 1. Fetch the next available saddle ID atomically
    uint id = atomic_fetch_add_explicit(global_saddle_index, 1, memory_order_relaxed);

    // 2. If we have processed all saddles, this thread can retire
    if (id >= (uint)p.num_saddles) {
      break;
    }

    // 3. Process the saddle (Identical logic, but completely hides warp divergence)
    int s_id = saddles[id];
    int type = get_type(s_id);
    int ey = get_y(s_id, p.Nx);
    int ex = get_x(s_id, p.Nx);

    out_s_vals[id] = cell_value<IS_DUAL>(type, ey, ex, p.H, p.W, data);

    // --- to maxima ---
    int f1 = -1, f2 = -1;
    if (type == 1) {
      if (is_valid_f(ey, ex, p.H, p.W)) f1 = cell_id(3, ey, ex, p.Nx);
      if (is_valid_f(ey + 1, ex, p.H, p.W)) f2 = cell_id(3, ey + 1, ex, p.Nx);
    } else {
      if (is_valid_f(ey, ex, p.H, p.W)) f1 = cell_id(3, ey, ex, p.Nx);
      if (is_valid_f(ey, ex + 1, p.H, p.W)) f2 = cell_id(3, ey, ex + 1, p.Nx);
    }

    TraceRes max1 = (f1 != -1) ? trace_faces_with_length(f1, paired_with, p.Nx, p.H, p.W) : TraceRes{-1, 0};
    TraceRes max2 = (f2 != -1) ? trace_faces_with_length(f2, paired_with, p.Nx, p.H, p.W) : TraceRes{-1, 0};

    if ((max1.target != -1 && max2.target == -1) || (max1.target == -1 && max2.target != -1)) {
      max_c1[id] = (max1.target != -1) ? max1.target : max2.target;
      max_c2[id] = -1;
      out_max_len[2 * id] = (max1.target != -1) ? max1.length : max2.length;
      out_max_len[2 * id + 1] = 0;
    } else {
      max_c1[id] = max1.target;
      max_c2[id] = max2.target;
      out_max_len[2 * id] = max1.length;
      out_max_len[2 * id + 1] = max2.length;
    }

    // --- to minima ---
    int v1 = -1, v2 = -1;
    if (type == 1) {
      if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, p.Nx);
      if (ex < p.W) v2 = cell_id(0, ey, ex, p.Nx);
    } else {
      if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, p.Nx);
      if (ey < p.H) v2 = cell_id(0, ey, ex, p.Nx);
    }

    TraceRes min1 = (v1 != -1) ? trace_vertices_with_length(v1, paired_with, p.Nx, p.H, p.W) : TraceRes{-1, 0};
    TraceRes min2 = (v2 != -1) ? trace_vertices_with_length(v2, paired_with, p.Nx, p.H, p.W) : TraceRes{-1, 0};

    if ((min1.target != -1 && min2.target == -1) || (min1.target == -1 && min2.target != -1)) {
      min_c1[id] = (min1.target != -1) ? min1.target : min2.target;
      min_c2[id] = -1;
      out_min_len[2 * id] = (min1.target != -1) ? min1.length : min2.length;
      out_min_len[2 * id + 1] = 0;
    } else {
      min_c1[id] = min1.target;
      min_c2[id] = min2.target;
      out_min_len[2 * id] = min1.length;
      out_min_len[2 * id + 1] = min2.length;
    }
  }
}

kernel void trace_from_saddles_primal(device const float* data [[buffer(0)]],
                                      device const int* paired_with [[buffer(1)]],
                                      device const int* saddles [[buffer(2)]], device int* max_c1 [[buffer(3)]],
                                      device int* max_c2 [[buffer(4)]], device int* min_c1 [[buffer(5)]],
                                      device int* min_c2 [[buffer(6)]], device float* out_s_vals [[buffer(7)]],
                                      device int* out_max_len [[buffer(8)]], device int* out_min_len [[buffer(9)]],
                                      constant Constants& p [[buffer(10)]],
                                      device atomic_uint* global_saddle_index [[buffer(11)]]) {
  trace_from_saddles_impl<false>(data, paired_with, saddles, max_c1, max_c2, min_c1, min_c2, out_s_vals, out_max_len,
                                 out_min_len, global_saddle_index, p);
}

kernel void trace_from_saddles_dual(device const float* data [[buffer(0)]], device const int* paired_with [[buffer(1)]],
                                    device const int* saddles [[buffer(2)]], device int* max_c1 [[buffer(3)]],
                                    device int* max_c2 [[buffer(4)]], device int* min_c1 [[buffer(5)]],
                                    device int* min_c2 [[buffer(6)]], device float* out_s_vals [[buffer(7)]],
                                    device int* out_max_len [[buffer(8)]], device int* out_min_len [[buffer(9)]],
                                    constant Constants& p [[buffer(10)]],
                                    device atomic_uint* global_saddle_index [[buffer(11)]]) {
  trace_from_saddles_impl<true>(data, paired_with, saddles, max_c1, max_c2, min_c1, min_c2, out_s_vals, out_max_len,
                                out_min_len, global_saddle_index, p);
}