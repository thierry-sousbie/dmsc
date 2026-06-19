#include <ATen/Parallel.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

// #include "../cpu/arcs_simplification.hxx"
#include "../cpu/arcs_simplification.hxx"
#include "../cpu/persistence_struct.hxx"
#include "./arcs_geometry_struct.hxx"
#include "./arcs_simplification_struct.hxx"
// #include "./gradient_struct.hxx"
#include "./trace_saddles_helpers.hxx"

void launch_gradient_cuda(const float* data, int* paired_with, int H, int W, bool is_dual);
void launch_extract_critical_points_cuda(const int* d_paired_with, int H, int W, int Nx, std::vector<int>& crit_mins,
                                         std::vector<int>& crit_saddles, std::vector<int>& crit_maxes, bool is_dual);
void launch_cell_groups_cuda(const float* data, const int* paired_with, const int* fast_crit_map, const int* uf_parent,
                             const int* crits, const int* fast_region_id, int* out_groups, int H, int W, int Nx,
                             bool is_dual, bool trace_faces);
gpu::TracedSaddlesTensors launch_trace_from_saddles_cuda(torch::Tensor& active_field, torch::Tensor& paired_with,
                                                         torch::Tensor& saddles_tensor, int H, int W, int Nx,
                                                         bool is_dual);

void launch_trace_raw_arcs_geometry_cuda(const int* d_paired_with, const int* d_fast_crit_map,
                                         const int* d_crit_saddles, const int* d_max_offsets, const int* d_min_offsets,
                                         int* d_flat_max, int* d_flat_min, void* d_saddle_nodes, int H, int W, int Nx,
                                         int num_saddles);

void launch_simplify_arcs_cuda(torch::Tensor d_cancels, torch::Tensor d_init_t, torch::Tensor d_parent_ptrs,
                               torch::Tensor d_weights, torch::Tensor d_dag, torch::Tensor d_base_lens,
                               torch::Tensor d_alive, torch::Tensor d_pending, int num_cancels, int num_extrema, int N2,
                               int* host_dag_sz, cudaStream_t stream);

namespace gpu {

template <bool IS_DUAL = false, typename Workspace>
void compute_gradient(Workspace& ws, torch::Tensor& active_field) {
  RECORD_FUNCTION("compute_gradient_and_crit_points_gpu", {});
  auto t0 = std::chrono::high_resolution_clock::now();

  int H = active_field.size(0);
  int W = active_field.size(1);
  int Nx = W + 1;
  int num_cells = 4 * (H + 1) * (W + 1);

  std::vector<int> paired_with(num_cells, -1);
  CriticalPoints cp;

  active_field = active_field.contiguous();
  ws.d_data = active_field;  // Keep a handle to the data on GPU
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(active_field.device());
  torch::Tensor d_paired_with = ws.gradient_data.d_paired_with.request_full({num_cells}, opts, -1);

  auto t1 = std::chrono::high_resolution_clock::now();

  launch_gradient_cuda(active_field.data_ptr<float>(), d_paired_with.data_ptr<int>(), H, W, IS_DUAL);

  auto t2 = std::chrono::high_resolution_clock::now();
  // is_dual is false because we computed minus the discrete gradient over the dual cells
  launch_extract_critical_points_cuda(d_paired_with.data_ptr<int>(), H, W, Nx, cp.mins, cp.saddles, cp.maxes, false);

  auto t3 = std::chrono::high_resolution_clock::now();

  // The CUDA extraction populated cp.saddles on the CPU. We push it back to the GPU as a Tensor
  // so it's ready immediately for Phase 2 trace_from_saddles.
  torch::Tensor d_saddles = ws.gradient_data.d_saddles.copy_from_cpu_ptr((void*)cp.saddles.data(), {(long)cp.saddles.size()}, opts);

  // Copy paired_with back to CPU (Needed for Phase 3 simplification)
  torch::Tensor cpu_paired = d_paired_with.cpu();
  std::vector<int> paired_with(num_cells);
  std::memcpy(paired_with.data(), cpu_paired.data_ptr<int>(), num_cells * sizeof(int));

  auto& fast_crit_map = ws.hlp.fast_crit_map;
  fast_crit_map.assign(num_cells, -1);
  for (size_t i = 0; i < cp.maxes.size(); ++i) fast_crit_map[cp.maxes[i]] = i;
  for (size_t i = 0; i < cp.mins.size(); ++i) fast_crit_map[cp.mins[i]] = i;
  for (size_t i = 0; i < cp.saddles.size(); ++i) fast_crit_map[cp.saddles[i]] = i;

  auto t4 = std::chrono::high_resolution_clock::now();
  ws.gradient_data.cp = std::move(cp);
  ws.gradient_data.paired_with = std::move(paired_with);
}

template <bool IS_DUAL = false, typename Workspace>
void compute_cell_groups(Workspace& ws) {
  RECORD_FUNCTION("cell_groups_gpu", {});
  const auto& gdata = ws.gradient_data;
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  int num_cells = 4 * (H + 1) * (W + 1);
  int num_faces = (H + 1) * (W + 1);
  int num_vertices = (H) * (W);

  auto& uf_max = ws.p_data.uf_max;
  auto& uf_min = ws.p_data.uf_min;
  const auto& max_alive = ws.p_data.max_alive;
  const auto& min_alive = ws.p_data.min_alive;
  const auto& crit_maxes = ws.gradient_data.cp.maxes;
  const auto& crit_mins = ws.gradient_data.cp.mins;
  const auto& fast_crit_map = ws.hlp.fast_crit_map;
  size_t num_crit_maxes = crit_maxes.size();
  size_t num_crit_mins = crit_mins.size();

  std::vector<int> fast_region_id(num_cells, -1);
  {
    RECORD_FUNCTION("cell_groups_preproc_cpu", {});
    // flatten union_find structures and build fast_region_id
    tbb::parallel_invoke(
        [&]() {
          for (size_t i = 0; i < uf_max.parent.size(); ++i) uf_max.find(i);
        },
        [&]() {
          for (size_t i = 0; i < uf_min.parent.size(); ++i) uf_min.find(i);
        },
        [&]() {
          int region_counter = 0;
          for (size_t i = 0; i < crit_maxes.size(); ++i) {
            if (max_alive[i]) fast_region_id[crit_maxes[i]] = region_counter++;
          }
        },
        [&]() {
          int region_counter = 0;
          for (size_t i = 0; i < crit_mins.size(); ++i) {
            if (min_alive[i]) fast_region_id[crit_mins[i]] = region_counter++;
          }
        });
  }

  auto i_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

  torch::Tensor d_fast_crit_map = torch::from_blob((void*)fast_crit_map.data(), {num_cells}, torch::kInt32).to(torch::kCUDA);

  torch::Tensor d_uf_max_parent =
      torch::from_blob((void*)uf_max.parent.data(), {(long)num_crit_maxes}, torch::kInt32).to(torch::kCUDA);
  torch::Tensor d_crit_maxes =
      torch::from_blob((void*)crit_maxes.data(), {(long)num_crit_maxes}, torch::kInt32).to(torch::kCUDA);

  torch::Tensor d_uf_min_parent =
      torch::from_blob((void*)uf_min.parent.data(), {(long)num_crit_mins}, torch::kInt32).to(torch::kCUDA);
  torch::Tensor d_crit_mins = torch::from_blob((void*)crit_mins.data(), {(long)num_crit_mins}, torch::kInt32).to(torch::kCUDA);

  torch::Tensor d_fast_region = torch::from_blob((void*)fast_region_id.data(), {num_cells}, torch::kInt32).to(torch::kCUDA);

  // Pre-allocate output on CUDA, -2 means unknonw
  torch::Tensor d_out_face_groups = torch::full({H + 1, W + 1}, -2, i_opts);
  torch::Tensor d_out_vertex_groups = torch::full({H, W}, -2, i_opts);
  {
    RECORD_FUNCTION("trace_faces", {});
    launch_cell_groups_cuda(ws.d_data.data_ptr<float>(), gdata.d_paired_with.get().data_ptr<int>(), d_fast_crit_map.data_ptr<int>(), d_uf_max_parent.data_ptr<int>(), d_crit_maxes.data_ptr<int>(),
                            d_fast_region.data_ptr<int>(), d_out_face_groups.data_ptr<int>(), H, W, Nx, IS_DUAL, /*trace_faces=*/true);
  }
  {
    RECORD_FUNCTION("trace_vertices", {});
    launch_cell_groups_cuda(ws.d_data.data_ptr<float>(), gdata.d_paired_with.get().data_ptr<int>(), d_fast_crit_map.data_ptr<int>(), d_uf_min_parent.data_ptr<int>(), d_crit_mins.data_ptr<int>(),
                            d_fast_region.data_ptr<int>(), d_out_vertex_groups.data_ptr<int>(), H, W, Nx, IS_DUAL, /*trace_faces=*/false);
  }

  // Bring the result back to CPU memory synchronously
  {
    RECORD_FUNCTION("Memcpy_Device_To_Host", {});
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor out_face_groups = ws.cell_groups.face_groups.request({H + 1, W + 1}, cpu_opts);
    torch::Tensor out_vertex_groups = ws.cell_groups.vertex_groups.request({H, W}, cpu_opts);

    // 2. Direct copy (GPU -> Pre-existing CPU RAM)
    out_face_groups.copy_(d_out_face_groups, /*non_blocking=*/false);
    out_vertex_groups.copy_(d_out_vertex_groups, /*non_blocking=*/false);
  }
}

template <bool IS_DUAL = false, typename Workspace>
void trace_from_saddles(Workspace& ws) {
  RECORD_FUNCTION("trace_from_saddles", {});
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  int saddles_count = ws.gradient_data.cp.saddles.size();

  auto traced_saddles_tensors =
      launch_trace_from_saddles_cuda(ws.d_data, ws.gradient_data.d_paired_with.get(),
                                      ws.gradient_data.d_saddles.get(), saddles_count, H, W, Nx, IS_DUAL);

  ws.arcs_topology = tensors_to_sad_events<IS_DUAL>(traced_saddles_tensors, Nx);
}

template <typename Workspace>
void trace_raw_arcs_geometry(Workspace& ws, const GradientData& gdata, const int* fast_crit_map,
                             const std::vector<int>& max_arcs_len, const std::vector<int>& min_arcs_len) {
  int H = ws.d_data.size(0);
  int W = ws.d_data.size(1);
  int Nx = W + 1;
  int num_cells = 4 * (H + 1) * (W + 1);
  RECORD_FUNCTION("trace_raw_arcs_geometry_gpu", {});
  auto dev = ws.d_data.device();

  torch::Tensor d_fast_crit_map =
      torch::from_blob((void*)fast_crit_map, {num_cells}, torch::kInt32).to(dev, /*non_blocking=*/true);

  auto& out = ws.saddle_nodes;
  int num_saddles = max_arcs_len.size() / 2;
  if (num_saddles == 0) return out;

  std::vector<int> max_offsets(num_saddles * 2 + 1, 0);
  std::vector<int> min_offsets(num_saddles * 2 + 1, 0);

  {
    RECORD_FUNCTION("prepare", {});
    for (int i = 0; i < num_saddles * 2; ++i) {
      max_offsets[i + 1] = max_offsets[i] + max_arcs_len[i] + 1;
      min_offsets[i + 1] = min_offsets[i] + min_arcs_len[i] + 1;
    }
    out.nodes.resize(num_saddles);
  }

  auto cuda_opts = torch::TensorOptions().device(dev);
  auto int_opts = cuda_opts.dtype(torch::kInt32);

  // Allocate UNINITIALIZED memory natively on CUDA
  torch::Tensor d_saddle_nodes =
      torch::empty({(long)(num_saddles * sizeof(SaddleNode))}, cuda_opts.dtype(torch::kUInt8));
  torch::Tensor d_flat_max = torch::empty({(long)max_offsets.back()}, int_opts);
  torch::Tensor d_flat_min = torch::empty({(long)min_offsets.back()}, int_opts);

  torch::Tensor d_max_offsets =
      torch::from_blob(max_offsets.data(), {num_saddles * 2 + 1}, torch::kInt32).to(dev, /*non_blocking=*/true);
  torch::Tensor d_min_offsets =
      torch::from_blob(min_offsets.data(), {num_saddles * 2 + 1}, torch::kInt32).to(dev, /*non_blocking=*/true);

  {
    RECORD_FUNCTION("kernel", {});
    // Extract raw pointers and dispatch to standard CUDA kernel
    launch_trace_raw_arcs_geometry_cuda(
        gdata.d_paired_with.get().data_ptr<int>(), d_fast_crit_map.data_ptr<int>(), gdata.d_saddles.get().data_ptr<int>(),
        d_max_offsets.data_ptr<int>(), d_min_offsets.data_ptr<int>(), d_flat_max.data_ptr<int>(),
        d_flat_min.data_ptr<int>(), d_saddle_nodes.data_ptr<uint8_t>(), H, W, Nx, num_saddles);
  }

  // out.flat_max_geom = d_flat_max.cpu();
  // out.flat_min_geom = d_flat_min.cpu();
  out.flat_max_geom.copy_from_tensor(d_flat_max, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
  out.flat_min_geom.copy_from_tensor(d_flat_min, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));

  torch::from_blob(out.nodes.data(), {(long)(num_saddles * sizeof(SaddleNode))}, torch::kUInt8).copy_(d_saddle_nodes);
}

template <typename Workspace>
void simplify_arcs_geometry(Workspace& ws, SaddleNodes& sn, int num_crit_maxes, int num_crit_mins,
                            std::vector<cpu::CancelEvent>& min_cancellations,
                            std::vector<cpu::CancelEvent>& max_cancellations) {
  RECORD_FUNCTION("simplify_arcs_geometry_cuda_dispatch", {});

  int num_saddles = sn.nodes.size();
  int N2 = num_saddles * 2;

  std::vector<int> max_parent(num_crit_maxes), min_parent(num_crit_mins);
  std::vector<GPUPathRef> max_weight(num_crit_maxes, {-1, 1}), min_weight(num_crit_mins, {-1, 1});

  for (int i = 0; i < num_crit_maxes; ++i) max_parent[i] = i;
  for (int i = 0; i < num_crit_mins; ++i) min_parent[i] = i;

  std::vector<int> base_max_len(N2, 0), base_min_len(N2, 0);
  std::vector<int> base_max_offset(N2, 0), base_min_offset(N2, 0);
  std::vector<int> init_t_max(N2, -1), init_t_min(N2, -1);

  {
    RECORD_FUNCTION("Extract_Base_Data_CPU", {});
    at::parallel_for(0, num_saddles, 1024, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        int idx = static_cast<int>(i);
        init_t_max[2 * idx] = sn.nodes[idx].max_arcs[0].target;
        init_t_max[2 * idx + 1] = sn.nodes[idx].max_arcs[1].target;
        init_t_min[2 * idx] = sn.nodes[idx].min_arcs[0].target;
        init_t_min[2 * idx + 1] = sn.nodes[idx].min_arcs[1].target;

        base_max_len[2 * idx] = sn.nodes[idx].max_arcs[0].length;
        base_max_len[2 * idx + 1] = sn.nodes[idx].max_arcs[1].length;
        base_max_offset[2 * idx] = sn.nodes[idx].max_arcs[0].offset;
        base_max_offset[2 * idx + 1] = sn.nodes[idx].max_arcs[1].offset;

        base_min_len[2 * idx] = sn.nodes[idx].min_arcs[0].length;
        base_min_len[2 * idx + 1] = sn.nodes[idx].min_arcs[1].length;
        base_min_offset[2 * idx] = sn.nodes[idx].min_arcs[0].offset;
        base_min_offset[2 * idx + 1] = sn.nodes[idx].min_arcs[1].offset;
      }
    });
  }

  auto dev = torch::kCUDA;
  auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);

  // MAXIMA PREP
  int num_max_cancels = max_cancellations.size();
  size_t safe_max_dag = std::max<size_t>(MIN_DAG_ALLOC_TOTAL, min_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR);

  torch::Tensor d_max_init_t = torch::from_blob(init_t_max.data(), {N2}, torch::kInt32).to(dev, true);
  torch::Tensor d_max_base_lens = torch::from_blob(base_max_len.data(), {N2}, torch::kInt32).to(dev, true);
  torch::Tensor d_max_parent = torch::from_blob(max_parent.data(), {num_crit_maxes}, torch::kInt32).to(dev, true);
  torch::Tensor d_max_weights =
      torch::from_blob(max_weight.data(), {(long)(num_crit_maxes * sizeof(GPUPathRef))}, torch::kUInt8).to(dev, true);
  torch::Tensor d_max_dag = torch::empty({(long)(safe_max_dag * sizeof(GPUDAGNode))}, byte_opts);

  std::vector<int> flat_max_cancels(num_max_cancels * 2);
  for (int i = 0; i < num_max_cancels; ++i) {
    flat_max_cancels[i * 2] = max_cancellations[i].s_idx;
    flat_max_cancels[i * 2 + 1] = max_cancellations[i].t_idx;
  }
  torch::Tensor d_max_cancels =
      torch::from_blob(flat_max_cancels.data(), {num_max_cancels * 2}, torch::kInt32).to(dev, true);
  torch::Tensor d_max_alive = torch::ones({num_saddles}, byte_opts);
  torch::Tensor d_max_pending = torch::ones({num_max_cancels}, byte_opts);
  int max_dag_sz = 0;

  // MINIMA PREP
  int num_min_cancels = min_cancellations.size();
  size_t safe_min_dag = std::max<size_t>(MIN_DAG_ALLOC_TOTAL, min_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR);

  torch::Tensor d_min_init_t = torch::from_blob(init_t_min.data(), {N2}, torch::kInt32).to(dev, true);
  torch::Tensor d_min_base_lens = torch::from_blob(base_min_len.data(), {N2}, torch::kInt32).to(dev, true);
  torch::Tensor d_min_parent = torch::from_blob(min_parent.data(), {num_crit_mins}, torch::kInt32).to(dev, true);
  torch::Tensor d_min_weights =
      torch::from_blob(min_weight.data(), {(long)(num_crit_mins * sizeof(GPUPathRef))}, torch::kUInt8).to(dev, true);
  torch::Tensor d_min_dag = torch::empty({(long)(safe_min_dag * sizeof(GPUDAGNode))}, byte_opts);

  std::vector<int> flat_min_cancels(num_min_cancels * 2);
  for (int i = 0; i < num_min_cancels; ++i) {
    flat_min_cancels[i * 2] = min_cancellations[i].s_idx;
    flat_min_cancels[i * 2 + 1] = min_cancellations[i].t_idx;
  }
  torch::Tensor d_min_cancels =
      torch::from_blob(flat_min_cancels.data(), {num_min_cancels * 2}, torch::kInt32).to(dev, true);
  torch::Tensor d_min_alive = torch::ones({num_saddles}, byte_opts);
  torch::Tensor d_min_pending = torch::ones({num_min_cancels}, byte_opts);
  int min_dag_sz = 0;

  // DISPATCH, min and max being independant
  at::cuda::CUDAStream stream_max = at::cuda::getStreamFromPool();
  at::cuda::CUDAStream stream_min = at::cuda::getStreamFromPool();
  {
    RECORD_FUNCTION("CUDA_Iterative_Contraction", {});
    launch_simplify_arcs_cuda(d_max_cancels, d_max_init_t, d_max_parent, d_max_weights, d_max_dag, d_max_base_lens,
                              d_max_alive, d_max_pending, num_max_cancels, num_crit_maxes, N2, &max_dag_sz,
                              stream_max.stream());
    launch_simplify_arcs_cuda(d_min_cancels, d_min_init_t, d_min_parent, d_min_weights, d_min_dag, d_min_base_lens,
                              d_min_alive, d_min_pending, num_min_cancels, num_crit_mins, N2, &min_dag_sz,
                              stream_min.stream());
  }
  stream_max.synchronize();
  stream_min.synchronize();

  // RETRIEVE AND ASSEMBLE
  torch::Tensor cpu_max_parent = d_max_parent.cpu();
  std::memcpy(max_parent.data(), cpu_max_parent.data_ptr<int>(), num_crit_maxes * sizeof(int));
  torch::Tensor cpu_min_parent = d_min_parent.cpu();
  std::memcpy(min_parent.data(), cpu_min_parent.data_ptr<int>(), num_crit_mins * sizeof(int));

  std::vector<uint8_t> max_alive(num_saddles);
  torch::Tensor cpu_max_alive = d_max_alive.cpu();
  std::memcpy(max_alive.data(), cpu_max_alive.data_ptr<uint8_t>(), num_saddles * sizeof(uint8_t));

  std::vector<uint8_t> min_alive(num_saddles);
  torch::Tensor cpu_min_alive = d_min_alive.cpu();
  std::memcpy(min_alive.data(), cpu_min_alive.data_ptr<uint8_t>(), num_saddles * sizeof(uint8_t));

  torch::Tensor cpu_max_weights = d_max_weights.cpu();
  std::memcpy(max_weight.data(), cpu_max_weights.data_ptr(), num_crit_maxes * sizeof(GPUPathRef));
  torch::Tensor cpu_min_weights = d_min_weights.cpu();
  std::memcpy(min_weight.data(), cpu_min_weights.data_ptr(), num_crit_mins * sizeof(GPUPathRef));

  size_t req_max_dag = max_dag_sz + N2;
  std::vector<GPUDAGNode> cpu_max_dag(req_max_dag);
  if (max_dag_sz > 0) {
    torch::Tensor cpu_max_dag_tensor = d_max_dag.cpu();
    std::memcpy(cpu_max_dag.data(), cpu_max_dag_tensor.data_ptr(), max_dag_sz * sizeof(GPUDAGNode));
  }

  size_t req_min_dag = min_dag_sz + N2;
  std::vector<GPUDAGNode> cpu_min_dag(req_min_dag);
  if (min_dag_sz > 0) {
    torch::Tensor cpu_min_dag_tensor = d_min_dag.cpu();
    std::memcpy(cpu_min_dag.data(), cpu_min_dag_tensor.data_ptr(), min_dag_sz * sizeof(GPUDAGNode));
  }

  cpu::assemble_simplified_geometry<GPUDAGNode, GPUPathRef>(
      ws, max_alive, min_alive, init_t_max, init_t_min, base_max_len, base_min_len, base_max_offset, base_min_offset,
      max_parent, min_parent, max_weight, min_weight, cpu_max_dag, max_dag_sz, cpu_min_dag, min_dag_sz);
}
}  // namespace gpu