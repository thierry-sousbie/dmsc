#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "../cell_complex.hxx"

namespace cpu {

template <bool IS_DUAL = false>
bool value_less(float val1, int id1, float val2, int id2) {
  if constexpr (!IS_DUAL) {
    return std::tie(val1, id1) < std::tie(val2, id2);
  } else {
    return std::tie(val1, id1) > std::tie(val2, id2);
  }
}

template <bool IS_DUAL = false>
bool value_greater(float val1, int id1, float val2, int id2) {
  return value_less<!IS_DUAL>(val1, id1, val2, id2);
}

template <bool IS_DUAL = false>
bool v_greater(int y1, int x1, int y2, int x2, const float* data, int W) {
  // compare values of two vertices (pixels)at coordinates (x1, y1) and (x2, y2) an returns true if the first is larger
  int id1 = y1 * W + x1;
  int id2 = y2 * W + x2;
  // float val1 = data[y1 * W + x1];
  // float val2 = data[y2 * W + x2];
  return value_greater<IS_DUAL>(data[id1], id1, data[id2], id2);
}

template <bool IS_DUAL = false>
std::pair<int, int> get_highest_vertex_yx(int type, int y, int x, int H, int W, const float* data) {
  int best_y = -1, best_x = -1;

  auto check_and_update = [&](int vy, int vx) {
    if (best_y == -1) {
      best_y = vy;
      best_x = vx;
    } else if (v_greater<IS_DUAL>(vy, vx, best_y, best_x, data, W)) {
      best_y = vy;
      best_x = vx;
    }
  };

  if (type == 0) {
    check_and_update(y, x);
  } else if (type == 1) {  // Horizontal edge
    if (x - 1 >= 0) check_and_update(y, x - 1);
    if (x < W) check_and_update(y, x);
  } else if (type == 2) {  // Vertical edge
    if (y - 1 >= 0) check_and_update(y - 1, x);
    if (y < H) check_and_update(y, x);
  } else if (type == 3) {
    if (y - 1 >= 0 && x - 1 >= 0) check_and_update(y - 1, x - 1);
    if (y - 1 >= 0 && x < W) check_and_update(y - 1, x);
    if (y < H && x - 1 >= 0) check_and_update(y, x - 1);
    if (y < H && x < W) check_and_update(y, x);
  }

  return {best_y, best_x};
}

template <bool IS_DUAL = false>
std::pair<int, int> get_lowest_vertex_yx(int type, int y, int x, int H, int W, const float* data) {
  return get_highest_vertex_yx<!IS_DUAL>(type, y, x, H, W, data);
}

template <bool IS_DUAL = false>
int get_lowest_vertex(int type, int y, int x, int H, int W, int Nx, const float* data) {
  auto yx = get_lowest_vertex_yx<IS_DUAL>(type, y, x, H, W, data);
  return cell_id(0, yx.first, yx.second, Nx);
}

template <bool IS_DUAL = false>
int get_highest_vertex(int type, int y, int x, int H, int W, int Nx, const float* data) {
  auto yx = get_highest_vertex_yx<IS_DUAL>(type, y, x, H, W, data);
  return cell_id(0, yx.first, yx.second, Nx);
}

template <bool IS_DUAL = false>
float cell_value(int type, int y, int x, int H, int W, const float* data) {
  // the value of a cell is that of the highest of its vertices
  auto mv = get_highest_vertex_yx<IS_DUAL>(type, y, x, H, W, data);
  return data[mv.first * W + mv.second];
}

template <bool IS_DUAL = false>
int get_highest_face(int y, int x, int H, int W, int Nx, const float* data) {
  int f_coords[4][2] = {{y, x}, {y, x + 1}, {y + 1, x}, {y + 1, x + 1}};
  int best_f_id = -1;
  float best_val = IS_DUAL ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();

  for (int i = 0; i < 4; ++i) {
    int fy = f_coords[i][0], fx = f_coords[i][1];
    if (fy >= 0 && fy <= H && fx >= 0 && fx <= W) {
      float v = cell_value<IS_DUAL>(3, fy, fx, H, W, data);
      int f_id = cell_id(3, fy, fx, Nx);

      bool is_better = false;
      if (best_f_id == -1) {
        is_better = true;
      } else {
        is_better = value_greater<IS_DUAL>(v, f_id, best_val, best_f_id);
      }
      if (is_better) {
        best_val = v;
        best_f_id = f_id;
      }
    }
  }
  return best_f_id;
}

template <bool IS_DUAL = false>
int get_lowest_face(int y, int x, int H, int W, int Nx, const float* data) {
  return get_highest_face<!IS_DUAL>(y, x, H, W, Nx, data);
}

}  // namespace cpu