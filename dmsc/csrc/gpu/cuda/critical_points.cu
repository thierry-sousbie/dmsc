#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../../cell_complex.hxx"
#include "../gradient_struct.hxx"

__global__ void extract_critical_points_kernel(const int* __restrict__ paired_with, int* __restrict__ out_vertices,
                                               int* __restrict__ out_edges, int* __restrict__ out_faces,
                                               int* __restrict__ counters,  // [0]=vertices, [1]=edges, [2]=faces
                                               int H, int W, int Nx) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_cells = 4 * (H + 1) * (W + 1);

  bool is_vertex = false;
  bool is_edge = false;
  bool is_face = false;

  if (id < num_cells && paired_with[id] == -1) {
    int type = get_type(id);
    int y = get_y(id, Nx);
    int x = get_x(id, Nx);

    if (type == 0 && is_valid_v(y, x, H, W))
      is_vertex = true;
    else if ((type == 1 && is_valid_ehx(y, x, H, W)) || (type == 2 && is_valid_evy(y, x, H, W)))
      is_edge = true;
    else if (type == 3 && is_valid_f(y, x, H, W))
      is_face = true;
  }

  int lane_id = threadIdx.x % 32;

  // --- VERTICES ---
  unsigned int v_ballot = __ballot_sync(0xFFFFFFFF, is_vertex);
  int v_total = __popc(v_ballot);
  int v_offset = __popc(v_ballot & ((1u << lane_id) - 1));

  int v_base = 0;
  if (lane_id == 0 && v_total > 0) {
    v_base = atomicAdd(&counters[0], v_total);
  }
  v_base = __shfl_sync(0xFFFFFFFF, v_base, 0);

  if (is_vertex) out_vertices[v_base + v_offset] = id;

  // --- EDGES ---
  unsigned int e_ballot = __ballot_sync(0xFFFFFFFF, is_edge);
  int e_total = __popc(e_ballot);
  int e_offset = __popc(e_ballot & ((1u << lane_id) - 1));

  int e_base = 0;
  if (lane_id == 0 && e_total > 0) {
    e_base = atomicAdd(&counters[1], e_total);
  }
  e_base = __shfl_sync(0xFFFFFFFF, e_base, 0);

  if (is_edge) out_edges[e_base + e_offset] = id;

  // --- FACES ---
  unsigned int f_ballot = __ballot_sync(0xFFFFFFFF, is_face);
  int f_total = __popc(f_ballot);
  int f_offset = __popc(f_ballot & ((1u << lane_id) - 1));

  int f_base = 0;
  if (lane_id == 0 && f_total > 0) {
    f_base = atomicAdd(&counters[2], f_total);
  }
  f_base = __shfl_sync(0xFFFFFFFF, f_base, 0);

  if (is_face) out_faces[f_base + f_offset] = id;
}

template <bool FACES, bool IS_DUAL>
__global__ void gather_critical_values_kernel(const float* data, const int* cell_ids, float* values, int count, int H,
                                              int W, int Nx) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= count) return;

  int cell = cell_ids[id] / 4;
  int y = cell / Nx;
  int x = cell % Nx;
  if constexpr (!FACES) {
    values[id] = data[y * W + x];
    return;
  }

  float value = IS_DUAL ? CUDART_INF_F : -CUDART_INF_F;
  if (y > 0 && x > 0) value = IS_DUAL ? fminf(value, data[(y - 1) * W + x - 1])
                                       : fmaxf(value, data[(y - 1) * W + x - 1]);
  if (y > 0 && x < W) value = IS_DUAL ? fminf(value, data[(y - 1) * W + x])
                                       : fmaxf(value, data[(y - 1) * W + x]);
  if (y < H && x > 0) value = IS_DUAL ? fminf(value, data[y * W + x - 1])
                                       : fmaxf(value, data[y * W + x - 1]);
  if (y < H && x < W) value = IS_DUAL ? fminf(value, data[y * W + x]) : fmaxf(value, data[y * W + x]);
  values[id] = value;
}

torch::Tensor launch_gather_critical_values_cuda(const torch::Tensor& data, const torch::Tensor& cell_ids, int H, int W,
                                                 int Nx, bool faces, bool is_dual) {
  torch::Tensor values = torch::empty({cell_ids.numel()}, data.options());
  int count = cell_ids.numel();
  if (count == 0) return values;

  int threads = 256;
  int blocks = (count + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (faces && is_dual) {
    gather_critical_values_kernel<true, true><<<blocks, threads, 0, stream>>>(
        data.data_ptr<float>(), cell_ids.data_ptr<int>(), values.data_ptr<float>(), count, H, W, Nx);
  } else if (faces) {
    gather_critical_values_kernel<true, false><<<blocks, threads, 0, stream>>>(
        data.data_ptr<float>(), cell_ids.data_ptr<int>(), values.data_ptr<float>(), count, H, W, Nx);
  } else {
    gather_critical_values_kernel<false, false><<<blocks, threads, 0, stream>>>(
        data.data_ptr<float>(), cell_ids.data_ptr<int>(), values.data_ptr<float>(), count, H, W, Nx);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return values;
}

gpu::CriticalPointsAsTensors launch_extract_critical_points_cuda(const int* d_paired_with, int H, int W, int Nx) {
  int num_cells = 4 * (H + 1) * (W + 1);
  int max_expected = num_cells / 3;

  auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32);

  torch::Tensor d_vertices = torch::empty({max_expected}, opts);
  torch::Tensor d_edges = torch::empty({max_expected}, opts);
  torch::Tensor d_faces = torch::empty({max_expected}, opts);
  torch::Tensor d_counters = torch::zeros({3}, opts);

  int threads = 256;
  int blocks = (num_cells + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  extract_critical_points_kernel<<<blocks, threads, 0, stream>>>(
      d_paired_with, d_vertices.data_ptr<int>(), d_edges.data_ptr<int>(), d_faces.data_ptr<int>(),
      d_counters.data_ptr<int>(), H, W, Nx);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  int h_counters[3];
  C10_CUDA_CHECK(cudaMemcpyAsync(h_counters, d_counters.data_ptr<int>(), 3 * sizeof(int), cudaMemcpyDeviceToHost, stream));
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  int vertex_count = h_counters[0];
  int edge_count = h_counters[1];
  int face_count = h_counters[2];

  torch::Tensor exact_vertices = d_vertices.slice(0, 0, vertex_count);
  torch::Tensor exact_edges = d_edges.slice(0, 0, edge_count);
  torch::Tensor exact_faces = d_faces.slice(0, 0, face_count);

  return {exact_faces, exact_edges, exact_vertices};
}
