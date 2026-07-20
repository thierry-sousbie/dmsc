#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

// Core Structs (match C++ host definition)
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

struct TraceResult {
  int target;
  int length;
};

// cell Helpers
__device__ inline int cell_id(int type, int y, int x, int Nx) {
  return type + 4 * (y * Nx + x);
}
__device__ inline int get_type(int id) {
  return id % 4;
}
__device__ inline int get_y(int id, int Nx) {
  return (id / 4) / Nx;
}
__device__ inline int get_x(int id, int Nx) {
  return (id / 4) % Nx;
}
__device__ inline bool is_valid_v(int y, int x, int H, int W) {
  return y >= 0 && y < H && x >= 0 && x < W;
}
__device__ inline bool is_valid_ehx(int y, int x, int H, int W) {
  return y >= 0 && y < H && x >= 0 && x <= W;
}
__device__ inline bool is_valid_evy(int y, int x, int H, int W) {
  return y >= 0 && y <= H && x >= 0 && x < W;
}
__device__ inline bool is_valid_f(int y, int x, int H, int W) {
  return y >= 0 && y <= H && x >= 0 && x <= W;
}

__device__ inline TraceResult trace_geom_face(int start_face, const int* paired_with, int* flat_geom, int base_offset,
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

  return {curr, count};
}

__device__ inline TraceResult trace_geom_vertex(int start_v, const int* paired_with, int* flat_geom, int base_offset,
                                                int H, int W, int Nx) {
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

  return {curr, count};
}

__global__ void trace_raw_arcs_geometry_kernel(const int* paired_with, const int* fast_crit_map,
                                               const int* crit_saddles, const int* max_offsets, const int* min_offsets,
                                               int* flat_max_geom, int* flat_min_geom, SaddleNode* saddle_nodes, int H,
                                               int W, int Nx, int num_saddles, bool trace_max_arcs,
                                               bool trace_min_arcs) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num_saddles) return;

  int s_id = crit_saddles[id];
  int ey = get_y(s_id, Nx), ex = get_x(s_id, Nx), type = get_type(s_id);

  saddle_nodes[id].alive = 1;
  for (int i = 0; i < 2; ++i) {
    saddle_nodes[id].max_arcs[i] = {-1, 0, 0};
    saddle_nodes[id].min_arcs[i] = {-1, 0, 0};
  }

  // MAX ARCS (Faces)
  if (trace_max_arcs) {
    int f1 = -1, f2 = -1;
    if (type == 1) {
      if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
      if (is_valid_f(ey + 1, ex, H, W)) f2 = cell_id(3, ey + 1, ex, Nx);
    } else {
      if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
      if (is_valid_f(ey, ex + 1, H, W)) f2 = cell_id(3, ey, ex + 1, Nx);
    }

    int valid_max = 0;
    if (f1 != -1) {
      int arc_idx = valid_max++;
      int base_off = max_offsets[2 * id + arc_idx];
      TraceResult result = trace_geom_face(f1, paired_with, flat_max_geom, base_off, H, W, Nx);
      int target = (result.target != -1) ? fast_crit_map[result.target] : -1;
      saddle_nodes[id].max_arcs[arc_idx] = {target, base_off, result.length};
    }
    if (f2 != -1) {
      int arc_idx = valid_max++;
      int base_off = max_offsets[2 * id + arc_idx];
      TraceResult result = trace_geom_face(f2, paired_with, flat_max_geom, base_off, H, W, Nx);
      int target = (result.target != -1) ? fast_crit_map[result.target] : -1;
      saddle_nodes[id].max_arcs[arc_idx] = {target, base_off, result.length};
    }
  }

  // MIN ARCS (Vertices)
  if (trace_min_arcs) {
    int v1 = -1, v2 = -1;
    if (type == 1) {
      if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, Nx);
      if (ex < W) v2 = cell_id(0, ey, ex, Nx);
    } else {
      if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, Nx);
      if (ey < H) v2 = cell_id(0, ey, ex, Nx);
    }

    int valid_min = 0;
    if (v1 != -1) {
      int arc_idx = valid_min++;
      int base_off = min_offsets[2 * id + arc_idx];
      TraceResult result = trace_geom_vertex(v1, paired_with, flat_min_geom, base_off, H, W, Nx);
      int target = (result.target != -1) ? fast_crit_map[result.target] : -1;
      saddle_nodes[id].min_arcs[arc_idx] = {target, base_off, result.length};
    }
    if (v2 != -1) {
      int arc_idx = valid_min++;
      int base_off = min_offsets[2 * id + arc_idx];
      TraceResult result = trace_geom_vertex(v2, paired_with, flat_min_geom, base_off, H, W, Nx);
      int target = (result.target != -1) ? fast_crit_map[result.target] : -1;
      saddle_nodes[id].min_arcs[arc_idx] = {target, base_off, result.length};
    }
  }
}

void launch_trace_raw_arcs_geometry_cuda(const int* d_paired_with, const int* d_fast_crit_map,
                                         const int* d_crit_saddles, const int* d_max_offsets, const int* d_min_offsets,
                                         int* d_flat_max, int* d_flat_min, void* d_saddle_nodes, int H, int W, int Nx,
                                         int num_saddles, bool trace_max_arcs, bool trace_min_arcs) {
  int threads = 256;
  int blocks = (num_saddles + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  trace_raw_arcs_geometry_kernel<<<blocks, threads, 0, stream>>>(
      d_paired_with, d_fast_crit_map, d_crit_saddles, d_max_offsets, d_min_offsets, d_flat_max, d_flat_min,
      (SaddleNode*)d_saddle_nodes, H, W, Nx, num_saddles, trace_max_arcs, trace_min_arcs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
