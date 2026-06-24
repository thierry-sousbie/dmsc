#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

  extract_critical_points_kernel<<<blocks, threads>>>(d_paired_with, d_vertices.data_ptr<int>(),
                                                      d_edges.data_ptr<int>(), d_faces.data_ptr<int>(),
                                                      d_counters.data_ptr<int>(), H, W, Nx);

  int h_counters[3];
  cudaMemcpy(h_counters, d_counters.data_ptr<int>(), 3 * sizeof(int), cudaMemcpyDeviceToHost);

  int vertex_count = h_counters[0];
  int edge_count = h_counters[1];
  int face_count = h_counters[2];

  torch::Tensor exact_vertices = d_vertices.slice(0, 0, vertex_count);
  torch::Tensor exact_edges = d_edges.slice(0, 0, edge_count);
  torch::Tensor exact_faces = d_faces.slice(0, 0, face_count);

  return {exact_faces, exact_edges, exact_vertices};
}