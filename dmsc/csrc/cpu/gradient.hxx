#pragma once

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <torch/extension.h>

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "./cell_compare.hxx"
#include "./gradient_struct.hxx"

#define DEFAULT_GRADIENT_BLOCK_SIZE 128

namespace cpu {

template <bool IS_DUAL, typename scalar_t>
void compute_gradient(int block_size, int H, int W, int Nx, const scalar_t* data, std::vector<int>& paired_with) {
  // --- DISCRETE GRADIENT ---
  // See "Theory and Algorithms for Constructing Discrete Morse Complexes from Grayscale Digital Images" by Robins et
  // al. (2011)
  RECORD_FUNCTION("compute_gradient_cpu", {});
  int num_blocks_y = (H + block_size - 1) / block_size;
  int num_blocks_x = (W + block_size - 1) / block_size;
  int total_blocks = num_blocks_y * num_blocks_x;

  paired_with.assign(4 * (H + 1) * (W + 1), -1);

  at::parallel_for(0, total_blocks, 1, [&](int64_t start, int64_t end) {
    std::vector<int> queue;
    queue.reserve(16);

    int L_edges[4];
    int L_faces[4];
    int fedges[4];

    for (int64_t b = start; b < end; ++b) {
      int by = b / num_blocks_x;
      int bx = b % num_blocks_x;
      int y_start = by * block_size;
      int y_end = std::min(y_start + block_size, H);
      int x_start = bx * block_size;
      int x_end = std::min(x_start + block_size, W);

      // consider vertex with coordinates (x,y) and compute lower star
      for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
          int L_e_cnt = 0;
          int L_f_cnt = 0;

          // Check that the current vertex is the highest of each cells that contain it
          // Horizontal edges
          if (get_highest_vertex_yx<IS_DUAL>(1, y, x, H, W, data) == std::make_pair(y, x))
            L_edges[L_e_cnt++] = cell_id(1, y, x, Nx);
          if (get_highest_vertex_yx<IS_DUAL>(1, y, x + 1, H, W, data) == std::make_pair(y, x))
            L_edges[L_e_cnt++] = cell_id(1, y, x + 1, Nx);

          // Vertical edges
          if (get_highest_vertex_yx<IS_DUAL>(2, y, x, H, W, data) == std::make_pair(y, x))
            L_edges[L_e_cnt++] = cell_id(2, y, x, Nx);
          if (get_highest_vertex_yx<IS_DUAL>(2, y + 1, x, H, W, data) == std::make_pair(y, x))
            L_edges[L_e_cnt++] = cell_id(2, y + 1, x, Nx);

          // Pixel corners
          if (get_highest_vertex_yx<IS_DUAL>(3, y, x, H, W, data) == std::make_pair(y, x))
            L_faces[L_f_cnt++] = cell_id(3, y, x, Nx);
          if (get_highest_vertex_yx<IS_DUAL>(3, y, x + 1, H, W, data) == std::make_pair(y, x))
            L_faces[L_f_cnt++] = cell_id(3, y, x + 1, Nx);
          if (get_highest_vertex_yx<IS_DUAL>(3, y + 1, x, H, W, data) == std::make_pair(y, x))
            L_faces[L_f_cnt++] = cell_id(3, y + 1, x, Nx);
          if (get_highest_vertex_yx<IS_DUAL>(3, y + 1, x + 1, H, W, data) == std::make_pair(y, x))
            L_faces[L_f_cnt++] = cell_id(3, y + 1, x + 1, Nx);

          int best_e = -1;                              // index of best edge (the one with lowest value)
          std::pair<int, int> best_other_v = {-1, -1};  // vertex coord (Y,X)

          for (int i = 0; i < L_e_cnt; ++i) {
            int e_id = L_edges[i];
            int ey = get_y(e_id, Nx), ex = get_x(e_id, Nx);
            int type = get_type(e_id);
            std::pair<int, int> other_v = {-1, -1};

            if (type == 1) {
              if (ex - 1 == x && ex < W)
                other_v = {ey, ex};
              else if (ex == x && ex - 1 >= 0)
                other_v = {ey, ex - 1};
            } else {
              if (ey - 1 == y && ey < H)
                other_v = {ey, ex};
              else if (ey == y && ey - 1 >= 0)
                other_v = {ey - 1, ex};
            }

            if (other_v.first != -1) {
              // keep the edge with SMALLEST other vertex
              if (best_e == -1 ||
                  v_greater<IS_DUAL>(best_other_v.first, best_other_v.second, other_v.first, other_v.second, data, W)) {
                best_e = e_id;
                best_other_v = other_v;
              }
            }
          }

          // Pair the edge if we got one
          if (best_e != -1) {
            int v_id = cell_id(0, y, x, Nx);
            paired_with[v_id] = best_e;
            paired_with[best_e] = v_id;
          }

          // Now pair faces and edges
          queue.clear();
          for (int i = 0; i < L_f_cnt; ++i) queue.push_back(L_faces[i]);

          size_t queue_head = 0;
          while (queue_head < queue.size()) {
            int f = queue[queue_head++];
            if (paired_with[f] != -1) continue;

            int fy = get_y(f, Nx), fx = get_x(f, Nx);
            int f_cnt = 0;
            // the four edges bounding face f
            if (is_valid_ehx(fy - 1, fx, H, W)) fedges[f_cnt++] = cell_id(1, fy - 1, fx, Nx);
            if (is_valid_ehx(fy, fx, H, W)) fedges[f_cnt++] = cell_id(1, fy, fx, Nx);
            if (is_valid_evy(fy, fx - 1, H, W)) fedges[f_cnt++] = cell_id(2, fy, fx - 1, Nx);
            if (is_valid_evy(fy, fx, H, W)) fedges[f_cnt++] = cell_id(2, fy, fx, Nx);

            // Count how many edges are unpaired in the lower star
            int unp_e = -1, unp_count = 0;
            for (int k = 0; k < f_cnt; ++k) {
              int e = fedges[k];
              bool in_L = false;  // in lower star
              for (int j = 0; j < L_e_cnt; ++j) {
                if (L_edges[j] == e) {
                  in_L = true;
                  break;
                }
              }
              if (in_L && paired_with[e] == -1) {
                unp_count++;
                unp_e = e;  // we only care if there is exactly one
              }
            }

            if (unp_count == 1) {
              // only one unpaired edge in the lower, pair it
              paired_with[f] = unp_e;
              paired_with[unp_e] = f;

              int ey = get_y(unp_e, Nx), ex = get_x(unp_e, Nx);

              auto check_and_push = [&](int new_f) {
                if (new_f != f) {
                  bool in_faces = false;  // Check if new_f is in the lower star
                  for (int j = 0; j < L_f_cnt; ++j) {
                    if (L_faces[j] == new_f) {
                      in_faces = true;
                      break;
                    }
                  }

                  if (in_faces) {
                    // new_f is indeed in the lower star
                    if (paired_with[new_f] == -1) {
                      bool already = false;
                      // add it to the queue if it's unpaired and is not already there
                      for (size_t qi = queue_head; qi < queue.size(); ++qi) {
                        if (queue[qi] == new_f) {
                          already = true;
                          break;
                        }
                      }
                      if (!already) queue.push_back(new_f);
                    }
                  }
                }
              };

              // possibly add the two faces bounding the newly paired edge
              if (get_type(unp_e) == 1) {
                if (is_valid_f(ey, ex, H, W)) check_and_push(cell_id(3, ey, ex, Nx));
                if (is_valid_f(ey + 1, ex, H, W)) check_and_push(cell_id(3, ey + 1, ex, Nx));
              } else {
                if (is_valid_f(ey, ex, H, W)) check_and_push(cell_id(3, ey, ex, Nx));
                if (is_valid_f(ey, ex + 1, H, W)) check_and_push(cell_id(3, ey, ex + 1, Nx));
              }
            }
          }
        }
      }
    }
  });
}

template <bool IS_DUAL, typename Workspace, typename scalar_t = float>
void update_gradient_helpers(Workspace& ws, const scalar_t* data) {
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  const auto& cp = ws.gradient_data.cp;
  auto& fast_crit_map = ws.hlp.fast_crit_map;
  auto& crit_max_vals = ws.hlp.crit_max_vals;
  auto& crit_min_vals = ws.hlp.crit_min_vals;

  RECORD_FUNCTION("helpers_fast_crit_map", {});
  fast_crit_map.assign(ws.num_cells, -1);
  crit_min_vals.resize(cp.mins.size());
  crit_max_vals.resize(cp.maxes.size());
  // printf("(IS_DUAL=%d): MAXES: %ld / MINS: %ld\n", (int)IS_DUAL, cp.maxes.size(), cp.mins.size());
  tbb::parallel_invoke(
      [&]() {
        for (size_t i = 0; i < cp.maxes.size(); ++i) {
          fast_crit_map[cp.maxes[i]] = i;
          crit_max_vals[i] = cell_value<IS_DUAL>(3, get_y(cp.maxes[i], Nx), get_x(cp.maxes[i], Nx), H, W, data);
        }
      },
      [&]() {
        for (size_t i = 0; i < cp.mins.size(); ++i) {
          fast_crit_map[cp.mins[i]] = i;
          crit_min_vals[i] = cell_value<IS_DUAL>(0, get_y(cp.mins[i], Nx), get_x(cp.mins[i], Nx), H, W, data);
        }
      },
      [&]() {
        for (size_t i = 0; i < cp.saddles.size(); ++i) {
          fast_crit_map[cp.saddles[i]] = i;
        }
      });
}

template <typename Workspace>
void update_gradient_helpers_from_values(Workspace& ws, const float* max_values, const float* min_values) {
  const auto& cp = ws.gradient_data.cp;
  auto& fast_crit_map = ws.hlp.fast_crit_map;
  auto& crit_max_vals = ws.hlp.crit_max_vals;
  auto& crit_min_vals = ws.hlp.crit_min_vals;

  RECORD_FUNCTION("helpers_fast_crit_map", {});
  fast_crit_map.assign(ws.num_cells, -1);
  crit_min_vals.resize(cp.mins.size());
  crit_max_vals.resize(cp.maxes.size());

  tbb::parallel_invoke(
      [&]() {
        for (size_t i = 0; i < cp.maxes.size(); ++i) {
          fast_crit_map[cp.maxes[i]] = i;
          crit_max_vals[i] = max_values[i];
        }
      },
      [&]() {
        for (size_t i = 0; i < cp.mins.size(); ++i) {
          fast_crit_map[cp.mins[i]] = i;
          crit_min_vals[i] = min_values[i];
        }
      },
      [&]() {
        for (size_t i = 0; i < cp.saddles.size(); ++i) fast_crit_map[cp.saddles[i]] = i;
      });
}

template <bool IS_DUAL, typename Workspace, typename scalar_t = float>
void compute_gradient_and_crit_points(Workspace& ws, const torch::Tensor& scalar_field,
                                      int block_size = DEFAULT_GRADIENT_BLOCK_SIZE) {
  const scalar_t* data = scalar_field.data_ptr<scalar_t>();
  GradientData& gd = ws.gradient_data;

  compute_gradient<IS_DUAL>(block_size, ws.H, ws.W, ws.Nx, data, gd.paired_with);
  extract_critical_points(gd.paired_with, ws.H, ws.W, ws.Nx, gd.cp.mins, gd.cp.saddles, gd.cp.maxes);
  update_gradient_helpers<IS_DUAL>(ws, data);
}

}  // namespace cpu
