#include <metal_stdlib>
using namespace metal;

struct Constants {
  int H;
  int W;
  int Nx;
  int num_saddles;
};

struct Arc {
  int target;
  int offset;
  int length;
};

struct SaddleNode {
  int alive;
  Arc max_arcs[2];
  Arc min_arcs[2];
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

inline int2 trace_geom_face(int start_face, device const int* paired_with, device int* flat_geom, int base_offset,
                            int H, int W, int Nx) {
  int curr = start_face;
  int count = 0;
  while (curr != -1) {
    if (get_type(curr) == 3) {
      flat_geom[base_offset + count] = curr;
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
  // Return both the final target cell and the total arc length
  return int2(curr, count);
}

inline int2 trace_geom_vertex(int start_v, device const int* paired_with, device int* flat_geom, int base_offset, int H,
                              int W, int Nx) {
  int curr = start_v;
  int count = 0;
  while (curr != -1) {
    if (get_type(curr) == 0) {
      flat_geom[base_offset + count] = curr;
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
  return int2(curr, count);
}

kernel void trace_raw_arcs_geometry_kernel(
    device const int* paired_with [[buffer(0)]], device const int* fast_crit_map [[buffer(1)]],
    device const int* crit_saddles [[buffer(2)]], device const int* max_offsets [[buffer(3)]],
    device const int* min_offsets [[buffer(4)]], device int* flat_max_geom [[buffer(5)]],
    device int* flat_min_geom [[buffer(6)]], device SaddleNode* saddle_nodes [[buffer(7)]],
    constant Constants& p [[buffer(8)]], constant bool& trace_max_arcs [[buffer(9)]],
    constant bool& trace_min_arcs [[buffer(10)]], uint id [[thread_position_in_grid]]) {
  if (id >= (uint)p.num_saddles) return;

  int s_id = crit_saddles[id];
  int ey = get_y(s_id, p.Nx), ex = get_x(s_id, p.Nx), type = get_type(s_id);

  // Initialize the struct directly in VRAM
  saddle_nodes[id].alive = 1;
  for (int i = 0; i < 2; ++i) {
    saddle_nodes[id].max_arcs[i] = {-1, 0, 0};
    saddle_nodes[id].min_arcs[i] = {-1, 0, 0};
  }

  // MAX ARCS
  if (trace_max_arcs) {
    int f1 = -1, f2 = -1;
    if (type == 1) {
      if (is_valid_f(ey, ex, p.H, p.W)) f1 = cell_id(3, ey, ex, p.Nx);
      if (is_valid_f(ey + 1, ex, p.H, p.W)) f2 = cell_id(3, ey + 1, ex, p.Nx);
    } else {
      if (is_valid_f(ey, ex, p.H, p.W)) f1 = cell_id(3, ey, ex, p.Nx);
      if (is_valid_f(ey, ex + 1, p.H, p.W)) f2 = cell_id(3, ey, ex + 1, p.Nx);
    }

    int valid_max = 0;
    if (f1 != -1) {
      int arc_idx = valid_max++;
      int base_off = max_offsets[2 * id + arc_idx];
      int2 result = trace_geom_face(f1, paired_with, flat_max_geom, base_off, p.H, p.W, p.Nx);
      int target = (result.x != -1) ? fast_crit_map[result.x] : -1;
      saddle_nodes[id].max_arcs[arc_idx] = {target, base_off, result.y};
    }
    if (f2 != -1) {
      int arc_idx = valid_max++;
      int base_off = max_offsets[2 * id + arc_idx];
      int2 result = trace_geom_face(f2, paired_with, flat_max_geom, base_off, p.H, p.W, p.Nx);
      int target = (result.x != -1) ? fast_crit_map[result.x] : -1;
      saddle_nodes[id].max_arcs[arc_idx] = {target, base_off, result.y};
    }
  }

  // MIN ARCS
  if (trace_min_arcs) {
    int v1 = -1, v2 = -1;
    if (type == 1) {
      if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, p.Nx);
      if (ex < p.W) v2 = cell_id(0, ey, ex, p.Nx);
    } else {
      if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, p.Nx);
      if (ey < p.H) v2 = cell_id(0, ey, ex, p.Nx);
    }

    int valid_min = 0;
    if (v1 != -1) {
      int arc_idx = valid_min++;
      int base_off = min_offsets[2 * id + arc_idx];
      int2 result = trace_geom_vertex(v1, paired_with, flat_min_geom, base_off, p.H, p.W, p.Nx);
      int target = (result.x != -1) ? fast_crit_map[result.x] : -1;
      saddle_nodes[id].min_arcs[arc_idx] = {target, base_off, result.y};
    }
    if (v2 != -1) {
      int arc_idx = valid_min++;
      int base_off = min_offsets[2 * id + arc_idx];
      int2 result = trace_geom_vertex(v2, paired_with, flat_min_geom, base_off, p.H, p.W, p.Nx);
      int target = (result.x != -1) ? fast_crit_map[result.x] : -1;
      saddle_nodes[id].min_arcs[arc_idx] = {target, base_off, result.y};
    }
  }
}