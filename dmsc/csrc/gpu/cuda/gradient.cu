#include <cuda_runtime.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <iostream>
#include <stdexcept>

#include "../../cell_complex.hxx"

struct Coord {
  int y;
  int x;
};

template <bool IS_DUAL>
__device__ __forceinline__ bool v_greater_shared(int gy1, int gx1, int gy2, int gx2, const float* tile, int start_y,
                                                 int start_x, int W) {
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
__device__ __forceinline__ Coord get_highest_vertex_shared(int type, int y, int x, int H, int W, const float* tile,
                                                           int start_y, int start_x) {
  Coord best = {-1, -1};

  auto check_and_update = [=](int vy, int vx, Coord current_best) -> Coord {
    if (current_best.y == -1) {
      return {vy, vx};
    } else if (v_greater_shared<IS_DUAL>(vy, vx, current_best.y, current_best.x, tile, start_y, start_x, W)) {
      return {vy, vx};
    }
    return current_best;
  };

  if (type == 0) {
    best = check_and_update(y, x, best);
  } else if (type == 1) {
    if (x - 1 >= 0) best = check_and_update(y, x - 1, best);
    if (x < W) best = check_and_update(y, x, best);
  } else if (type == 2) {
    if (y - 1 >= 0) best = check_and_update(y - 1, x, best);
    if (y < H) best = check_and_update(y, x, best);
  } else if (type == 3) {
    if (y - 1 >= 0 && x - 1 >= 0) best = check_and_update(y - 1, x - 1, best);
    if (y - 1 >= 0 && x < W) best = check_and_update(y - 1, x, best);
    if (y < H && x - 1 >= 0) best = check_and_update(y, x - 1, best);
    if (y < H && x < W) best = check_and_update(y, x, best);
  }

  return best;
}

template <bool IS_DUAL>
__global__ void compute_discrete_gradient_kernel(const float* data, int* paired_with, int H, int W, int Nx) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x * blockDim.x;
  int by = blockIdx.y * blockDim.y;

  int x = bx + tx;
  int y = by + ty;

  int block_start_x = bx - 1;
  int block_start_y = by - 1;

  __shared__ float tile[18 * 18];

  int local_id = ty * blockDim.x + tx;
  for (int i = local_id; i < 324; i += 256) {
    int ly = i / 18;
    int lx = i % 18;
    int gy = block_start_y + ly;
    int gx = block_start_x + lx;

    if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
      tile[i] = data[gy * W + gx];
    } else {
      tile[i] = IS_DUAL ? CUDART_INF_F : -CUDART_INF_F;  // this could remain uninitialized
    }
  }

  __syncthreads();

  if (x >= W || y >= H) return;

  int L_edges[4];
  int L_faces[4];
  int fedges[4];
  int L_e_cnt = 0;
  int L_f_cnt = 0;

  Coord mv;
  mv = get_highest_vertex_shared<IS_DUAL>(1, y, x, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(1, y, x, Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(1, y, x + 1, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(1, y, x + 1, Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(2, y, x, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(2, y, x, Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(2, y + 1, x, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_edges[L_e_cnt++] = cell_id(2, y + 1, x, Nx);

  mv = get_highest_vertex_shared<IS_DUAL>(3, y, x, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y, x, Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(3, y, x + 1, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y, x + 1, Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(3, y + 1, x, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y + 1, x, Nx);
  mv = get_highest_vertex_shared<IS_DUAL>(3, y + 1, x + 1, H, W, tile, block_start_y, block_start_x);
  if (mv.y == y && mv.x == x) L_faces[L_f_cnt++] = cell_id(3, y + 1, x + 1, Nx);

  int best_e = -1;
  Coord best_other_v = {-1, -1};

  for (int i = 0; i < L_e_cnt; ++i) {
    int e_id = L_edges[i];
    int ey = get_y(e_id, Nx), ex = get_x(e_id, Nx);
    int type = get_type(e_id);
    Coord other_v = {-1, -1};

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

    if (other_v.y != -1) {
      if (best_e == -1 || v_greater_shared<IS_DUAL>(best_other_v.y, best_other_v.x, other_v.y, other_v.x, tile,
                                                    block_start_y, block_start_x, W)) {
        best_e = e_id;
        best_other_v = other_v;
      }
    }
  }

  if (best_e != -1) {
    int v_id = cell_id(0, y, x, Nx);
    paired_with[v_id] = best_e;
    paired_with[best_e] = v_id;
  }

  for (int pass = 0; pass < 4; ++pass) {
    bool changed = false;

    for (int i = 0; i < L_f_cnt; ++i) {
      int f = L_faces[i];
      if (paired_with[f] != -1) continue;

      int fy = get_y(f, Nx), fx = get_x(f, Nx);
      int f_cnt = 0;

      if (is_valid_ehx(fy - 1, fx, H, W)) fedges[f_cnt++] = cell_id(1, fy - 1, fx, Nx);
      if (is_valid_ehx(fy, fx, H, W)) fedges[f_cnt++] = cell_id(1, fy, fx, Nx);
      if (is_valid_evy(fy, fx - 1, H, W)) fedges[f_cnt++] = cell_id(2, fy, fx - 1, Nx);
      if (is_valid_evy(fy, fx, H, W)) fedges[f_cnt++] = cell_id(2, fy, fx, Nx);

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

    if (!__any_sync(__activemask(), changed)) break;
  }
}

void launch_gradient_cuda(const float* data, int* paired_with, int H, int W, bool is_dual) {
  dim3 block(16, 16);
  dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
  int Nx = W + 1;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (is_dual) {
    compute_discrete_gradient_kernel<true><<<grid, block, 0, stream>>>(data, paired_with, H, W, Nx);
  } else {
    compute_discrete_gradient_kernel<false><<<grid, block, 0, stream>>>(data, paired_with, H, W, Nx);
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
