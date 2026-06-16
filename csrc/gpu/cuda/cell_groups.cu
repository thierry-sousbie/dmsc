#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

#include "../../cell_complex.hxx"

#define CACHE_IS_EMPTY -2
#define CACHE_DEPTH 16  // Number of steps to record for path compression (should fit in registers)

// Used to memoize naturally the faces we already computed wihtout requiring any sync
// This maximize chances long pathes get cut early, limiting warp divergence
__device__ __forceinline__ bool check_cached(int curr, int Nx, int row_stride, const int* __restrict__ out_groups,
                                             int* cached_value) {
  int cy = get_y(curr, Nx);
  int cx = get_x(curr, Nx);
  int flat_idx = cy * row_stride + cx;

  // Volatile to make sure it s not optimized out
  *cached_value = ((volatile int*)out_groups)[flat_idx];

  if (*cached_value != CACHE_IS_EMPTY) return true;

  return false;
}

__device__ __forceinline__ int trace_faces(int start_face, const int* __restrict__ paired_with, int H, int W, int Nx,
                                           const int* __restrict__ out_groups, int* cached_value, int* path_buffer) {
  if (start_face == -1) return -1;
  int curr = start_face;
  int step_count = 0;

  while (curr != -1 && paired_with[curr] != -1) {
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

    int edge = paired_with[curr];
    int ey = get_y(edge, Nx);
    int ex = get_x(edge, Nx);
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

__device__ __forceinline__ int trace_vertices(int start_v, const int* __restrict__ paired_with, int H, int W, int Nx,
                                              const int* __restrict__ out_groups, int* cached_value, int* path_buffer) {
  if (start_v == -1) return -1;
  int curr = start_v;
  int step_count = 0;

  while (curr != -1 && paired_with[curr] != -1) {
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

    int edge = paired_with[curr];
    int ey = get_y(edge, Nx);
    int ex = get_x(edge, Nx);
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
__global__ void compute_face_groups_kernel(const int* __restrict__ paired_with, const int* __restrict__ fast_crit_map,
                                           const int* __restrict__ uf_max_parent, const int* __restrict__ crit_maxes,
                                           const int* __restrict__ fast_region_id, int* __restrict__ out_groups, int H,
                                           int W, int Nx, int* __restrict__ global_index) {
  int num_faces = (H + 1) * (W + 1);

  while (true) {
    int id = atomicAdd(global_index, 1);
    if (id >= num_faces) break;

    int y = id / Nx;
    int x = id % Nx;

    // Start trace directly from the face
    int start_face = cell_id(3, y, x, Nx);
    int final_val = -1;

    int cached_value = CACHE_IS_EMPTY;
    int path_buffer[CACHE_DEPTH];

    int initial_max = trace_faces(start_face, paired_with, H, W, Nx, out_groups, &cached_value, path_buffer);
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
__global__ void compute_vertex_groups_kernel(const int* __restrict__ paired_with, const int* __restrict__ fast_crit_map,
                                             const int* __restrict__ uf_min_parent, const int* __restrict__ crit_mins,
                                             const int* __restrict__ fast_region_id, int* __restrict__ out_groups,
                                             int H, int W, int Nx, int* __restrict__ global_index) {
  int num_vertices = H * W;

  while (true) {
    int id = atomicAdd(global_index, 1);
    if (id >= num_vertices) break;

    int y = id / W;
    int x = id % W;

    // Start trace directly from the vertex!
    int start_vertex = cell_id(0, y, x, Nx);
    int final_val = -1;

    int cached_value = CACHE_IS_EMPTY;
    int path_buffer[CACHE_DEPTH];

    int initial_min = trace_vertices(start_vertex, paired_with, H, W, Nx, out_groups, &cached_value, path_buffer);
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

// n.b. out_groups MUST to be initialized to CACHE_IS_EMPTY when calling !!!!!!
void launch_cell_groups_cuda(const float* data,  // Kept to avoid breaking C++ signature
                             const int* paired_with, const int* fast_crit_map, const int* uf_parent, const int* crits,
                             const int* fast_region_id, int* out_groups, int H, int W, int Nx, bool is_dual,
                             bool trace_faces) {
  // Allocate and zero the native CUDA atomic counter
  int* d_counter;
  cudaMalloc(&d_counter, sizeof(int));
  cudaMemset(d_counter, 0, sizeof(int));

  // Launch a 1D grid, 256 blocks * 256 threads is enough to saturate any GPU
  int threads = 256;
  int blocks = 256;

  if (!trace_faces) {
    if (is_dual) {
      compute_vertex_groups_kernel<true><<<blocks, threads>>>(paired_with, fast_crit_map, uf_parent, crits,
                                                              fast_region_id, out_groups, H, W, Nx, d_counter);
    } else {
      compute_vertex_groups_kernel<false><<<blocks, threads>>>(paired_with, fast_crit_map, uf_parent, crits,
                                                               fast_region_id, out_groups, H, W, Nx, d_counter);
    }
  } else {
    if (is_dual) {
      compute_face_groups_kernel<true><<<blocks, threads>>>(paired_with, fast_crit_map, uf_parent, crits,
                                                            fast_region_id, out_groups, H, W, Nx, d_counter);
    } else {
      compute_face_groups_kernel<false><<<blocks, threads>>>(paired_with, fast_crit_map, uf_parent, crits,
                                                             fast_region_id, out_groups, H, W, Nx, d_counter);
    }
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    cudaFree(d_counter);
    throw std::runtime_error(std::string("Cell Groups Kernel Failed: ") + cudaGetErrorString(err));
  }

  cudaFree(d_counter);
}