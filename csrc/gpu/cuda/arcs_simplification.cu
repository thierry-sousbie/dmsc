#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

// GPUDAGNode and GPUPathRef in gpu namespace
#include "../arcs_simplification_struct.hxx"

using namespace gpu;

__device__ inline int read_only_find(int i, const int* parent) {
  int curr = i;
  while (curr != -1 && parent[curr] != curr) {
    curr = parent[curr];
  }
  return curr;
}

__device__ inline GPUPathRef merge_paths_cuda(GPUPathRef p1, GPUPathRef p2, GPUDAGNode* dag, int* dag_sz,
                                              const int* base_lens, int N2) {
  if (p1.id == -1) return p2;
  if (p2.id == -1) return p1;

  int len1 = (p1.id < N2) ? base_lens[p1.id] : dag[p1.id - N2].total_len;
  int len2 = (p2.id < N2) ? base_lens[p2.id] : dag[p2.id - N2].total_len;

  int alloc_idx = atomicAdd(dag_sz, 1);  // warps write concurently here
  int new_id = alloc_idx + N2;

  dag[alloc_idx].left_id = p1.id;
  dag[alloc_idx].right_id = p2.id;
  dag[alloc_idx].left_fwd = p1.fwd;
  dag[alloc_idx].right_fwd = p2.fwd;
  dag[alloc_idx].total_len = len1 + len2;

  return {new_id, 1};
}

__device__ inline GPUPathRef read_only_find_with_weight(int i, const int* parent, const GPUPathRef* weights,
                                                        GPUDAGNode* dag, int* dag_sz, const int* base_lens, int N2,
                                                        int stop_node) {
  if (i == -1) return GPUPathRef{-1, 1};

  GPUPathRef acc = weights[i];
  int curr = parent[i];

  while (curr != -1 && parent[curr] != curr && curr != stop_node) {
    if (weights[curr].id != -1) {
      acc = merge_paths_cuda(acc, weights[curr], dag, dag_sz, base_lens, N2);
    }
    curr = parent[curr];
  }
  return acc;
}

__global__ void evaluate_cancellations_kernel(const int* cancels, const int* init_t, const int* parent_ptrs,
                                              const uint8_t* alive, const uint8_t* pending, int* ready_count,
                                              int* ready_list, int* ready_R0, int* ready_R1, int num_cancels,
                                              int num_extrema) {
  // Grid-stride loop
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < num_cancels; id += blockDim.x * gridDim.x) {
    if (pending[id] == 0) continue;

    int s = cancels[id * 2];
    int dead = cancels[id * 2 + 1];

    if (alive[s] == 0 || dead < 0 || dead >= num_extrema) continue;

    int T0 = init_t[2 * s];
    int T1 = init_t[2 * s + 1];

    int R0 = read_only_find(T0, parent_ptrs);
    int R1 = read_only_find(T1, parent_ptrs);

    if ((R0 == dead && R0 != R1) || (R1 == dead && R0 != R1)) {
      int write_idx = atomicAdd(ready_count, 1);
      ready_list[write_idx] = id;
      ready_R0[write_idx] = R0;
      ready_R1[write_idx] = R1;
    }
  }
}

__global__ void contract_cancellations_kernel(const int* ready_list, const int* ready_R0, const int* ready_R1,
                                              const int* cancels, const int* init_t, const int* base_lens,
                                              int* parent_ptrs, GPUPathRef* weights, GPUDAGNode* dag, int* dag_sz,
                                              uint8_t* alive, uint8_t* pending, int num_ready, int N2) {
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < num_ready; id += blockDim.x * gridDim.x) {
    int cancel_idx = ready_list[id];
    int s = cancels[cancel_idx * 2];
    int dead = cancels[cancel_idx * 2 + 1];

    int T0 = init_t[2 * s];
    int T1 = init_t[2 * s + 1];

    int R0 = ready_R0[id];
    int R1 = ready_R1[id];

    if (R0 == dead && R0 != R1) {
      GPUPathRef wT0 = (T0 == -1 || T0 == R0) ? GPUPathRef{-1, 1} : weights[T0];
      GPUPathRef wT1 = (T1 == -1 || T1 == R1) ? GPUPathRef{-1, 1} : weights[T1];

      GPUPathRef Full0 = merge_paths_cuda({2 * s, 1}, wT0, dag, dag_sz, base_lens, N2);
      GPUPathRef Full1 = merge_paths_cuda({2 * s + 1, 1}, wT1, dag, dag_sz, base_lens, N2);
      weights[R0] = merge_paths_cuda({Full0.id, Full0.fwd == 0 ? 1 : 0}, Full1, dag, dag_sz, base_lens, N2);

      parent_ptrs[R0] = R1;

      alive[s] = 0;
      pending[cancel_idx] = 0;
    } else if (R1 == dead && R0 != R1) {
      GPUPathRef wT0 = (T0 == -1 || T0 == R0) ? GPUPathRef{-1, 1} : weights[T0];
      GPUPathRef wT1 = (T1 == -1 || T1 == R1) ? GPUPathRef{-1, 1} : weights[T1];

      GPUPathRef Full0 = merge_paths_cuda({2 * s, 1}, wT0, dag, dag_sz, base_lens, N2);
      GPUPathRef Full1 = merge_paths_cuda({2 * s + 1, 1}, wT1, dag, dag_sz, base_lens, N2);
      weights[R1] = merge_paths_cuda({Full1.id, Full1.fwd == 0 ? 1 : 0}, Full0, dag, dag_sz, base_lens, N2);

      parent_ptrs[R1] = R0;

      alive[s] = 0;
      pending[cancel_idx] = 0;
    }
  }
}

__global__ void compress_paths_step_kernel(const int* parent_in, const GPUPathRef* weights_in, int* parent_out,
                                           GPUPathRef* weights_out, GPUDAGNode* dag, int* dag_sz, const int* base_lens,
                                           int* changed, int num_extrema, int N2) {
  for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < num_extrema; id += blockDim.x * gridDim.x) {
    int p = parent_in[id];

    if (p != id && p != -1) {
      int pp = parent_in[p];
      if (pp != p && pp != -1) {
        GPUPathRef wp = weights_in[p];
        GPUPathRef w_new = merge_paths_cuda(weights_in[id], wp, dag, dag_sz, base_lens, N2);

        parent_out[id] = pp;
        weights_out[id] = w_new;
        *changed = 1;  // Thread-safe collision write
        continue;
      }
    }

    // Fallback: carry over
    parent_out[id] = p;
    weights_out[id] = weights_in[id];
  }
}

// ------------------------------------------
// 4. Host Dispatcher
// ------------------------------------------
void launch_simplify_arcs_cuda(torch::Tensor d_cancels, torch::Tensor d_init_t, torch::Tensor d_parent_ptrs,
                               torch::Tensor d_weights, torch::Tensor d_dag, torch::Tensor d_base_lens,
                               torch::Tensor d_alive, torch::Tensor d_pending, int num_cancels, int num_extrema, int N2,
                               int* host_dag_sz, cudaStream_t stream) {
  // auto stream = at::cuda::getCurrentCUDAStream();
  int threads = 256;

  // Allocate temporary scalar buffers for cross-bridge polling
  int* d_ready_count;
  int* d_ready_list;
  int* d_ready_R0;
  int* d_ready_R1;
  int* d_mtl_dag_sz;
  int* d_changed;
  cudaMallocAsync(&d_ready_count, sizeof(int), stream);
  cudaMallocAsync(&d_ready_list, num_cancels * sizeof(int), stream);
  cudaMallocAsync(&d_ready_R0, num_cancels * sizeof(int), stream);
  cudaMallocAsync(&d_ready_R1, num_cancels * sizeof(int), stream);
  cudaMallocAsync(&d_mtl_dag_sz, sizeof(int), stream);
  cudaMallocAsync(&d_changed, sizeof(int), stream);

  cudaMemcpyAsync(d_mtl_dag_sz, host_dag_sz, sizeof(int), cudaMemcpyHostToDevice, stream);

  // Allocate alt buffers for pointer jumping once outside the loop
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(d_parent_ptrs.device());
  torch::Tensor d_parent_alt = torch::empty_like(d_parent_ptrs);
  torch::Tensor d_weights_alt = torch::empty_like(d_weights);

  // Iterative Contraction Loop
  int iteration = 0;
  while (iteration < num_cancels) {
    cudaMemsetAsync(d_ready_count, 0, sizeof(int), stream);

    int blocks_eval = (num_cancels + threads - 1) / threads;
    evaluate_cancellations_kernel<<<blocks_eval, threads, 0, stream>>>(
        d_cancels.data_ptr<int>(), d_init_t.data_ptr<int>(), d_parent_ptrs.data_ptr<int>(), d_alive.data_ptr<uint8_t>(),
        d_pending.data_ptr<uint8_t>(), d_ready_count, d_ready_list, d_ready_R0, d_ready_R1, num_cancels, num_extrema);

    // Sync to read back the ready count
    int h_ready = 0;
    cudaMemcpyAsync(&h_ready, d_ready_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_ready == 0) break;

    int blocks_contract = (h_ready + threads - 1) / threads;
    contract_cancellations_kernel<<<blocks_contract, threads, 0, stream>>>(
        d_ready_list, d_ready_R0, d_ready_R1, d_cancels.data_ptr<int>(), d_init_t.data_ptr<int>(),
        d_base_lens.data_ptr<int>(), d_parent_ptrs.data_ptr<int>(), (GPUPathRef*)d_weights.data_ptr(),
        (GPUDAGNode*)d_dag.data_ptr(), d_mtl_dag_sz, d_alive.data_ptr<uint8_t>(), d_pending.data_ptr<uint8_t>(),
        h_ready, N2);


    bool use_alt = false;
    int pass = 0;

    int blocks_comp = (num_extrema + threads - 1) / threads;
    while (pass < 32) {
      cudaMemsetAsync(d_changed, 0, sizeof(int), stream);

      int* p_in = use_alt ? d_parent_alt.data_ptr<int>() : d_parent_ptrs.data_ptr<int>();
      int* p_out = use_alt ? d_parent_ptrs.data_ptr<int>() : d_parent_alt.data_ptr<int>();
      GPUPathRef* w_in = use_alt ? (GPUPathRef*)d_weights_alt.data_ptr() : (GPUPathRef*)d_weights.data_ptr();
      GPUPathRef* w_out = use_alt ? (GPUPathRef*)d_weights.data_ptr() : (GPUPathRef*)d_weights_alt.data_ptr();

      compress_paths_step_kernel<<<blocks_comp, threads, 0, stream>>>(
          p_in, w_in, p_out, w_out, (GPUDAGNode*)d_dag.data_ptr(), d_mtl_dag_sz, d_base_lens.data_ptr<int>(), d_changed,
          num_extrema, N2);

      int h_changed = 0;
      cudaMemcpyAsync(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);

      use_alt = !use_alt;
      pass++;

      if (h_changed == 0) break;
    }

    // If final data is in alt buffers, copy back to primary PyTorch tensors
    if (use_alt) {
      d_parent_ptrs.copy_(d_parent_alt, /*non_blocking=*/true);
      d_weights.copy_(d_weights_alt, /*non_blocking=*/true);
      cudaStreamSynchronize(stream);
    }

    iteration++;
  }

  // Pull final DAG size
  cudaMemcpyAsync(host_dag_sz, d_mtl_dag_sz, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Cleanup local allocations
  cudaFree(d_ready_count);
  cudaFree(d_ready_list);
  cudaFree(d_ready_R0);
  cudaFree(d_ready_R1);
  cudaFree(d_mtl_dag_sz);
  cudaFree(d_changed);
}