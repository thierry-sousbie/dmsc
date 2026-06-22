#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>

#include "../../cell_complex.hxx"

struct Coord {
  int y;
  int x;
};

// Return struct for tracing
struct TraceRes {
  int target;
  int length;
};

template <bool IS_DUAL = false>
__device__ __forceinline__ bool value_greater(float val1, int id1, float val2, int id2) {
  if (IS_DUAL) {
    if (val1 != val2) return val1 < val2;
    return id1 < id2;
  } else {
    if (val1 != val2) return val1 > val2;
    return id1 > id2;
  }
}

template <bool IS_DUAL = false>
__device__ __forceinline__ bool v_greater(int y1, int x1, int y2, int x2, const float* __restrict__ data, int W) {
  int id1 = y1 * W + x1;
  int id2 = y2 * W + x2;
  return value_greater<IS_DUAL>(data[id1], id1, data[id2], id2);
}

template <bool IS_DUAL = false>
__device__ __forceinline__ Coord _check_and_update(int vy, int vx, Coord best, const float* __restrict__ data, int W) {
  if (best.y == -1) {
    return {vy, vx};
  } else if (v_greater<IS_DUAL>(vy, vx, best.y, best.x, data, W)) {
    return {vy, vx};
  }
  return best;
}

template <bool IS_DUAL = false>
__device__ __forceinline__ Coord get_highest_vertex(int type, int y, int x, int H, int W,
                                                    const float* __restrict__ data) {
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
__device__ __forceinline__ float cell_value(int type, int y, int x, int H, int W, const float* __restrict__ data) {
  Coord mv = get_highest_vertex<IS_DUAL>(type, y, x, H, W, data);
  return data[mv.y * W + mv.x];
}

__device__ TraceRes trace_faces_with_length(int start_face, const int* __restrict__ paired_with, int H, int W, int Nx) {
  if (start_face == -1) return TraceRes{-1, 0};
  int curr = start_face;
  int steps = 0;

  while (curr != -1 && paired_with[curr] != -1) {
    steps++;
    int edge = paired_with[curr];
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
  return TraceRes{curr, steps};
}

__device__ TraceRes trace_vertices_with_length(int start_v, const int* __restrict__ paired_with, int H, int W, int Nx) {
  if (start_v == -1) return TraceRes{-1, 0};
  int curr = start_v;
  int steps = 0;

  while (curr != -1 && paired_with[curr] != -1) {
    steps++;
    int edge = paired_with[curr];
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
  return TraceRes{curr, steps};
}

template <bool IS_DUAL = false>
__global__ void trace_saddles_kernel(const float* __restrict__ data, const int* __restrict__ paired_with,
                                     const int* __restrict__ saddles, int* __restrict__ max_c1,
                                     int* __restrict__ max_c2, int* __restrict__ min_c1, int* __restrict__ min_c2,
                                     float* __restrict__ s_vals, int* __restrict__ max_len, int* __restrict__ min_len,
                                     int H, int W, int Nx, int num_saddles, int* __restrict__ global_saddle_index) {
    while (true) {
    int id = atomicAdd(global_saddle_index, 1);
    if (id >= num_saddles) break;

    int s_id = saddles[id];
    int type = get_type(s_id);
    int ey = get_y(s_id, Nx);
    int ex = get_x(s_id, Nx);

    s_vals[id] = cell_value<IS_DUAL>(type, ey, ex, H, W, data);

    // --- MAXIMA ---
    int f1 = -1, f2 = -1;
    if (type == 1) {
      if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
      if (is_valid_f(ey + 1, ex, H, W)) f2 = cell_id(3, ey + 1, ex, Nx);
    } else {
      if (is_valid_f(ey, ex, H, W)) f1 = cell_id(3, ey, ex, Nx);
      if (is_valid_f(ey, ex + 1, H, W)) f2 = cell_id(3, ey, ex + 1, Nx);
    }

    TraceRes max1 = (f1 != -1) ? trace_faces_with_length(f1, paired_with, H, W, Nx) : TraceRes{-1, 0};
    TraceRes max2 = (f2 != -1) ? trace_faces_with_length(f2, paired_with, H, W, Nx) : TraceRes{-1, 0};

    if ((max1.target != -1 && max2.target == -1) || (max1.target == -1 && max2.target != -1)) {
      max_c1[id] = (max1.target != -1) ? max1.target : max2.target;
      max_c2[id] = -1;
      max_len[2 * id] = (max1.target != -1) ? max1.length : max2.length;
      max_len[2 * id + 1] = 0;
    } else {
      max_c1[id] = max1.target;
      max_c2[id] = max2.target;
      max_len[2 * id] = max1.length;
      max_len[2 * id + 1] = max2.length;
    }

    // --- MINIMA ---
    int v1 = -1, v2 = -1;
    if (type == 1) {
      if (ex - 1 >= 0) v1 = cell_id(0, ey, ex - 1, Nx);
      if (ex < W) v2 = cell_id(0, ey, ex, Nx);
    } else {
      if (ey - 1 >= 0) v1 = cell_id(0, ey - 1, ex, Nx);
      if (ey < H) v2 = cell_id(0, ey, ex, Nx);
    }

    TraceRes min1 = (v1 != -1) ? trace_vertices_with_length(v1, paired_with, H, W, Nx) : TraceRes{-1, 0};
    TraceRes min2 = (v2 != -1) ? trace_vertices_with_length(v2, paired_with, H, W, Nx) : TraceRes{-1, 0};

    if ((min1.target != -1 && min2.target == -1) || (min1.target == -1 && min2.target != -1)) {
      min_c1[id] = (min1.target != -1) ? min1.target : min2.target;
      min_c2[id] = -1;
      min_len[2 * id] = (min1.target != -1) ? min1.length : min2.length;
      min_len[2 * id + 1] = 0;
    } else {
      min_c1[id] = min1.target;
      min_c2[id] = min2.target;
      min_len[2 * id] = min1.length;
      min_len[2 * id + 1] = min2.length;
    }
  }
}

// Ensure the driver signature aligns perfectly with the Metal equivalent
struct TracedSaddlesTensors {
  torch::Tensor saddles;
  torch::Tensor max_c1;
  torch::Tensor max_c2;
  torch::Tensor min_c1;
  torch::Tensor min_c2;
  torch::Tensor s_vals;
  torch::Tensor max_len;
  torch::Tensor min_len;
};

TracedSaddlesTensors launch_trace_from_saddles_cuda(torch::Tensor& active_field, torch::Tensor& paired_with,
                                                    torch::Tensor& saddles_tensor, int H, int W, int Nx, bool is_dual) {
  int num_saddles = saddles_tensor.numel();
  if (num_saddles == 0) return {};

  auto int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  // The kernel overwrites everything
  torch::Tensor max_c1 = torch::empty({num_saddles}, int_opts);
  torch::Tensor max_c2 = torch::empty({num_saddles}, int_opts);
  torch::Tensor min_c1 = torch::empty({num_saddles}, int_opts);
  torch::Tensor min_c2 = torch::empty({num_saddles}, int_opts);
  torch::Tensor s_vals = torch::empty({num_saddles}, float_opts);
  torch::Tensor max_len = torch::empty({num_saddles * 2}, int_opts);
  torch::Tensor min_len = torch::empty({num_saddles * 2}, int_opts);

  // The atomic counter must start at 0
  torch::Tensor global_counter = torch::zeros({1}, int_opts);

  int threads = 256;
  int blocks = 256;  // Fixed grid dimension

  if (is_dual) {
    trace_saddles_kernel<true><<<blocks, threads>>>(
        active_field.data_ptr<float>(), paired_with.data_ptr<int>(), saddles_tensor.data_ptr<int>(),
        max_c1.data_ptr<int>(), max_c2.data_ptr<int>(), min_c1.data_ptr<int>(), min_c2.data_ptr<int>(),
        s_vals.data_ptr<float>(), max_len.data_ptr<int>(), min_len.data_ptr<int>(), H, W, Nx, num_saddles,
        global_counter.data_ptr<int>());
  } else {
    trace_saddles_kernel<false><<<blocks, threads>>>(
        active_field.data_ptr<float>(), paired_with.data_ptr<int>(), saddles_tensor.data_ptr<int>(),
        max_c1.data_ptr<int>(), max_c2.data_ptr<int>(), min_c1.data_ptr<int>(), min_c2.data_ptr<int>(),
        s_vals.data_ptr<float>(), max_len.data_ptr<int>(), min_len.data_ptr<int>(), H, W, Nx, num_saddles,
        global_counter.data_ptr<int>());
  }

  return {saddles_tensor, max_c1, max_c2, min_c1, min_c2, s_vals, max_len, min_len};
}