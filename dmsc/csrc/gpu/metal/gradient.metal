#include <metal_stdlib>

using namespace metal;

// Discrete Morse cell complex navigation
// when dual is true, Faces <-> vertices and Vertical edge <-> Horizontal edge
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

struct Constants {
  int H;
  int W;
  int Nx;
};

struct Coord {
  int y;
  int x;
};

// n.b shared memory version + no constexpr in metal
template <bool IS_DUAL = false>
inline bool v_greater_shared(int gy1, int gx1, int gy2, int gx2, threadgroup const float* tile, int start_y,
                             int start_x, int W) {
  // Map global (y, x) down to the 18x18 tile coordinate
  float val1 = tile[(gy1 - start_y) * 18 + (gx1 - start_x)];
  float val2 = tile[(gy2 - start_y) * 18 + (gx2 - start_x)];
  if (IS_DUAL) {
    if (val1 != val2) return val1 < val2;
    return (gy1 * W + gx1) < (gy2 * W + gx2);
  } else {
    if (val1 != val2) return val1 > val2;
    return (gy1 * W + gx1) > (gy2 * W + gx2);
  }
}

template <bool IS_DUAL>
inline Coord _check_and_update(int vy, int vx, Coord current_best, threadgroup const float* tile, int start_y,
                               int start_x, int W) {
  if (current_best.y == -1) {
    return {vy, vx};
  } else if (v_greater_shared<IS_DUAL>(vy, vx, current_best.y, current_best.x, tile, start_y, start_x, W)) {
    return {vy, vx};
  }
  return current_best;
}

template <bool IS_DUAL = false>
inline Coord get_highest_vertex_shared(int type, int y, int x, int H, int W, threadgroup const float* tile, int start_y,
                                       int start_x) {
  Coord best = {-1, -1};

  if (type == 0) {
    best = _check_and_update<IS_DUAL>(y, x, best, tile, start_y, start_x, W);
  } else if (type == 1) {
    if (x - 1 >= 0) best = _check_and_update<IS_DUAL>(y, x - 1, best, tile, start_y, start_x, W);
    if (x < W) best = _check_and_update<IS_DUAL>(y, x, best, tile, start_y, start_x, W);
  } else if (type == 2) {
    if (y - 1 >= 0) best = _check_and_update<IS_DUAL>(y - 1, x, best, tile, start_y, start_x, W);
    if (y < H) best = _check_and_update<IS_DUAL>(y, x, best, tile, start_y, start_x, W);
  } else if (type == 3) {
    if (y - 1 >= 0 && x - 1 >= 0) best = _check_and_update<IS_DUAL>(y - 1, x - 1, best, tile, start_y, start_x, W);
    if (y - 1 >= 0 && x < W) best = _check_and_update<IS_DUAL>(y - 1, x, best, tile, start_y, start_x, W);
    if (y < H && x - 1 >= 0) best = _check_and_update<IS_DUAL>(y, x - 1, best, tile, start_y, start_x, W);
    if (y < H && x < W) best = _check_and_update<IS_DUAL>(y, x, best, tile, start_y, start_x, W);
  }

  return best;
}

template <bool IS_DUAL = false>
inline void compute_discrete_gradient_impl(device const float* data, device int* paired_with, constant Constants& p,
                                           uint2 tid_in_group, uint2 gid, uint2 dim, threadgroup float* tile) {
  int x = gid.x * dim.x + tid_in_group.x;
  int y = gid.y * dim.y + tid_in_group.y;
  int block_start_x = (int)(gid.x * dim.x) - 1;
  int block_start_y = (int)(gid.y * dim.y) - 1;

  // 256 threads load 324 pixels (16x16 + 1 pixel boundary)
  int local_id = tid_in_group.y * dim.x + tid_in_group.x;
  for (int i = local_id; i < 324; i += 256) {
    int ly = i / 18;
    int lx = i % 18;
    int gy = block_start_y + ly;
    int gx = block_start_x + lx;

    if (gy >= 0 && gy < p.H && gx >= 0 && gx < p.W) {
      tile[i] = data[gy * p.W + gx];
    } else {
      tile[i] = -INFINITY;  // Padding for out-of-bounds, this could remain uninitialized
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (x >= p.W || y >= p.H) return;

  int L_edges[4];
  int L_faces[4];
  int fedges[4];
  int L_e_cnt = 0;
  int L_f_cnt = 0;

  Coord mv;
  mv = get_highest_vertex_shared<IS_DUAL>(1, y, x, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(1, y, x, p.Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(1, y, x + 1, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(1, y, x + 1, p.Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(2, y, x, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(2, y, x, p.Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(2, y + 1, x, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(2, y + 1, x, p.Nx);

  mv = get_highest_vertex_shared<IS_DUAL>(3, y, x, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y, x, p.Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(3, y, x + 1, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y, x + 1, p.Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(3, y + 1, x, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y + 1, x, p.Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(3, y + 1, x + 1, p.H, p.W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y + 1, x + 1, p.Nx);

  int best_e = -1;
  Coord best_other_v = {-1, -1};

  for (int i = 0; i < L_e_cnt; ++i) {
    int e_id = L_edges[i];
    int ey = get_y(e_id, p.Nx), ex = get_x(e_id, p.Nx);
    int type = get_type(e_id);
    Coord other_v = {-1, -1};

    if (type == 1) {
      if (ex - 1 == x && ex < p.W)
        other_v = {ey, ex};
      else if (ex == x && ex - 1 >= 0)
        other_v = {ey, ex - 1};
    } else {
      if (ey - 1 == y && ey < p.H)
        other_v = {ey, ex};
      else if (ey == y && ey - 1 >= 0)
        other_v = {ey - 1, ex};
    }

    if (other_v.y != -1) {
      if (best_e == -1 || v_greater_shared<IS_DUAL>(best_other_v.y, best_other_v.x, other_v.y, other_v.x, tile,
                                                    block_start_y, block_start_x, p.W)) {
        best_e = e_id;
        best_other_v = other_v;
      }
    }
  }

  if (best_e != -1) {
    int v_id = cell_id(0, y, x, p.Nx);
    paired_with[v_id] = best_e;
    paired_with[best_e] = v_id;
  }

  for (int pass = 0; pass < 4; ++pass) {
    bool changed = false;
    for (int i = 0; i < L_f_cnt; ++i) {
      int f = L_faces[i];
      if (paired_with[f] != -1) continue;

      int fy = get_y(f, p.Nx), fx = get_x(f, p.Nx);
      int f_cnt = 0;
      if (is_valid_ehx(fy - 1, fx, p.H, p.W)) fedges[f_cnt++] = cell_id(1, fy - 1, fx, p.Nx);
      if (is_valid_ehx(fy, fx, p.H, p.W)) fedges[f_cnt++] = cell_id(1, fy, fx, p.Nx);
      if (is_valid_evy(fy, fx - 1, p.H, p.W)) fedges[f_cnt++] = cell_id(2, fy, fx - 1, p.Nx);
      if (is_valid_evy(fy, fx, p.H, p.W)) fedges[f_cnt++] = cell_id(2, fy, fx, p.Nx);

      int unp_e = -1, unp_count = 0;
      for (int k = 0; k < f_cnt; ++k) {
        int e = fedges[k];
        bool in_L = false;
        for (int j = 0; j < L_e_cnt; ++j) {
          if (L_edges[j] == e) {
            in_L = true;
            break;
          }
        }
        if (in_L && paired_with[e] == -1) {
          unp_count++;
          unp_e = e;
        }
      }

      if (unp_count == 1) {
        paired_with[f] = unp_e;
        paired_with[unp_e] = f;
        changed = true;
      }
    }

    // warp voting (simd == warp), <=> __any_sync()
    if (!simd_any(changed)) break;
  }
}

kernel void compute_discrete_gradient_primal(device const float* data [[buffer(0)]],
                                             device int* paired_with [[buffer(1)]], constant Constants& p [[buffer(2)]],
                                             uint2 tid_in_group [[thread_position_in_threadgroup]],
                                             uint2 gid [[threadgroup_position_in_grid]],
                                             uint2 dim [[threads_per_threadgroup]]) {
  threadgroup float tile[18 * 18];
  compute_discrete_gradient_impl<false>(data, paired_with, p, tid_in_group, gid, dim, tile);
}

kernel void compute_discrete_gradient_dual(device const float* data [[buffer(0)]],
                                           device int* paired_with [[buffer(1)]], constant Constants& p [[buffer(2)]],
                                           uint2 tid_in_group [[thread_position_in_threadgroup]],
                                           uint2 gid [[threadgroup_position_in_grid]],
                                           uint2 dim [[threads_per_threadgroup]]) {
  threadgroup float tile[18 * 18];
  compute_discrete_gradient_impl<true>(data, paired_with, p, tid_in_group, gid, dim, tile);
}
