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

#include "../cpu/arcs_simplification.hxx"
#include "../cpu/gradient.hxx"
#include "../cpu/persistence_struct.hxx"
#include "./arcs_geometry_struct.hxx"
#include "./arcs_simplification_struct.hxx"
#include "./arcs_topology_helpers.hxx"
#include "./gradient_struct.hxx"

#define MAX_DAG_ALLOC_PER_PAIR_GPU 15

void launch_gradient_cuda(const float* data, int* paired_with, int H, int W, bool is_dual);
torch::Tensor launch_gather_critical_values_cuda(const torch::Tensor& data, const torch::Tensor& cell_ids, int H, int W,
                                                 int Nx, bool faces, bool is_dual);
gpu::CriticalPointsAsTensors launch_extract_critical_points_cuda(const int* d_paired_with, int H, int W, int Nx);
void launch_cell_groups_cuda(const float* data, const int* paired_with, const int* fast_crit_map, const int* uf_parent,
                             const int* crits, const int* fast_region_id, int* out_groups, int H, int W, int Nx,
                             bool is_dual, bool trace_faces);
gpu::TracedSaddlesTensors launch_trace_from_saddles_cuda(torch::Tensor& active_field, torch::Tensor& paired_with,
                                                         torch::Tensor& saddles_tensor, int H, int W, int Nx,
                                                         bool is_dual);

void launch_trace_raw_arcs_geometry_cuda(const int* d_paired_with, const int* d_crit_saddles,
                                         const int* d_max_offsets, const int* d_min_offsets, int* d_flat_max,
                                         int* d_flat_min, int H, int W, int Nx, int num_saddles, bool trace_max_arcs,
                                         bool trace_min_arcs);

void launch_simplify_arcs_cuda(torch::Tensor d_cancels, torch::Tensor d_init_t, torch::Tensor d_parent_ptrs,
                               torch::Tensor d_weights, torch::Tensor d_dag, torch::Tensor d_base_lens,
                               torch::Tensor d_alive, torch::Tensor d_pending, int num_cancels, int num_extrema, int N2,
                               int* host_dag_sz, cudaStream_t stream);

namespace gpu {

CriticalPoints tensors_to_critical_points(const gpu::CriticalPointsAsTensors& cpt) {
  CriticalPoints cp;

  torch::Tensor cpu_mins = cpt.mins.cpu().contiguous();
  torch::Tensor cpu_saddles = cpt.saddles.cpu().contiguous();
  torch::Tensor cpu_maxes = cpt.maxes.cpu().contiguous();

  cp.mins.assign(cpu_mins.data_ptr<int>(), cpu_mins.data_ptr<int>() + cpu_mins.numel());
  cp.saddles.assign(cpu_saddles.data_ptr<int>(), cpu_saddles.data_ptr<int>() + cpu_saddles.numel());
  cp.maxes.assign(cpu_maxes.data_ptr<int>(), cpu_maxes.data_ptr<int>() + cpu_maxes.numel());

  return cp;
}

template <bool IS_DUAL = false, typename Workspace>
void compute_gradient(Workspace& ws, const torch::Tensor& scalar_field) {
  RECORD_FUNCTION("compute_gradient_and_crit_points_gpu", {});
  int H = ws.H;
  int W = ws.W;
  int Nx = W + 1;
  int num_cells = ws.num_cells;

  ws.d_data = scalar_field.contiguous();
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(scalar_field.device());
  torch::Tensor d_paired_with = ws.gradient_data.d_paired_with.request_full({num_cells}, opts, -1);

  {
    RECORD_FUNCTION("KERNEL_GRADIENT", {});
    launch_gradient_cuda(ws.d_data.template data_ptr<float>(), d_paired_with.data_ptr<int>(), H, W, IS_DUAL);
  }

  // Compute crit points on GPU
  auto cpt = launch_extract_critical_points_cuda(d_paired_with.data_ptr<int>(), H, W, Nx);
  ws.gradient_data.cp = tensors_to_critical_points(cpt);

  ws.gradient_data.d_maxes.copy_from_tensor(cpt.maxes, opts);
  ws.gradient_data.d_mins.copy_from_tensor(cpt.mins, opts);
  ws.gradient_data.d_saddles.copy_from_tensor(cpt.saddles, opts);

  {
    RECORD_FUNCTION("gather_critical_values", {});
    torch::Tensor max_values =
        launch_gather_critical_values_cuda(ws.d_data, cpt.maxes, H, W, Nx, /*faces=*/true, IS_DUAL).cpu();
    torch::Tensor min_values =
        launch_gather_critical_values_cuda(ws.d_data, cpt.mins, H, W, Nx, /*faces=*/false, IS_DUAL).cpu();
    cpu::update_gradient_helpers_from_values(ws, max_values.data_ptr<float>(), min_values.data_ptr<float>());
  }
}

template <bool IS_DUAL = false, typename Workspace>
void compute_cell_groups(Workspace& ws, bool trace_face_groups, bool trace_vertex_groups) {
  RECORD_FUNCTION("cell_groups_gpu", {});
  const auto& gdata = ws.gradient_data;
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  int num_cells = 4 * (H + 1) * (W + 1);

  auto& uf_max = ws.p_data.uf_max;
  auto& uf_min = ws.p_data.uf_min;
  const auto& max_alive = ws.p_data.max_alive;
  const auto& min_alive = ws.p_data.min_alive;
  const auto& crit_maxes = ws.gradient_data.cp.maxes;
  const auto& crit_mins = ws.gradient_data.cp.mins;
  size_t num_crit_maxes = crit_maxes.size();
  size_t num_crit_mins = crit_mins.size();

  {
    RECORD_FUNCTION("cell_groups_preproc_cpu", {});
    // flatten union_find structures
    tbb::parallel_invoke(
        [&]() {
          for (size_t i = 0; i < uf_max.parent.size(); ++i) uf_max.find(i);
        },
        [&]() {
          for (size_t i = 0; i < uf_min.parent.size(); ++i) uf_min.find(i);
        });
  }

  auto dev = ws.d_data.device();
  auto i_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);

  torch::Tensor d_crit_maxes = gdata.d_maxes.get();
  torch::Tensor d_crit_mins = gdata.d_mins.get();
  torch::Tensor d_crit_saddles = gdata.d_saddles.get();

  torch::Tensor d_fast_crit_map = ws.d_fast_crit_map;
  if (!d_fast_crit_map.defined()) {
    d_fast_crit_map = torch::full({num_cells}, -1, i_opts);
    if (num_crit_maxes > 0) d_fast_crit_map.index_put_({d_crit_maxes}, torch::arange((long)num_crit_maxes, i_opts));
    if (num_crit_mins > 0) d_fast_crit_map.index_put_({d_crit_mins}, torch::arange((long)num_crit_mins, i_opts));
    if (d_crit_saddles.numel() > 0)
      d_fast_crit_map.index_put_({d_crit_saddles}, torch::arange((long)d_crit_saddles.numel(), i_opts));
  }

  torch::Tensor d_uf_max_parent = torch::from_blob((void*)uf_max.parent.data(), {(long)num_crit_maxes}, torch::kInt32)
                                      .to(dev, /*non_blocking=*/true);

  torch::Tensor d_uf_min_parent = torch::from_blob((void*)uf_min.parent.data(), {(long)num_crit_mins}, torch::kInt32)
                                      .to(dev, /*non_blocking=*/true);

  auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);

  torch::Tensor d_max_alive = torch::from_blob((void*)max_alive.data(), {(long)num_crit_maxes}, torch::kUInt8)
                                  .to(dev, /*non_blocking=*/true)
                                  .to(torch::kBool);
  torch::Tensor d_min_alive = torch::from_blob((void*)min_alive.data(), {(long)num_crit_mins}, torch::kUInt8)
                                  .to(dev, /*non_blocking=*/true)
                                  .to(torch::kBool);

  torch::Tensor d_fast_region = ws.hlp.fast_region_id.request_full({num_cells}, i_opts, -1);

  // Compute region prefix sums natively on device
  torch::Tensor region_ids_max = d_max_alive.to(torch::kInt32).cumsum(0, torch::kInt32) - 1;
  d_fast_region.index_put_({d_crit_maxes.masked_select(d_max_alive)}, region_ids_max.masked_select(d_max_alive));

  torch::Tensor region_ids_min = d_min_alive.to(torch::kInt32).cumsum(0, torch::kInt32) - 1;
  d_fast_region.index_put_({d_crit_mins.masked_select(d_min_alive)}, region_ids_min.masked_select(d_min_alive));

  // Pre-allocate output on CUDA, -2 means unknown
  torch::Tensor d_out_face_groups = ws.cell_groups.face_groups.request_full({H + 1, W + 1}, i_opts, -2);
  torch::Tensor d_out_vertex_groups = ws.cell_groups.vertex_groups.request_full({H, W}, i_opts, -2);
  torch::Tensor d_paired_with = gdata.d_paired_with.get();
  if (trace_face_groups) {
    RECORD_FUNCTION("trace_faces", {});
    launch_cell_groups_cuda(ws.d_data.template data_ptr<float>(), d_paired_with.data_ptr<int>(),
                            d_fast_crit_map.data_ptr<int>(), d_uf_max_parent.data_ptr<int>(),
                            d_crit_maxes.data_ptr<int>(), d_fast_region.data_ptr<int>(),
                            d_out_face_groups.data_ptr<int>(), H, W, Nx, IS_DUAL, /*trace_faces=*/true);
  }
  if (trace_vertex_groups) {
    RECORD_FUNCTION("trace_vertices", {});
    launch_cell_groups_cuda(ws.d_data.template data_ptr<float>(), d_paired_with.data_ptr<int>(),
                            d_fast_crit_map.data_ptr<int>(), d_uf_min_parent.data_ptr<int>(),
                            d_crit_mins.data_ptr<int>(), d_fast_region.data_ptr<int>(),
                            d_out_vertex_groups.data_ptr<int>(), H, W, Nx, IS_DUAL, /*trace_faces=*/false);
  }
}

template <bool IS_DUAL = false, typename Workspace>
void trace_from_saddles(Workspace& ws) {
  RECORD_FUNCTION("trace_from_saddles", {});
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;

  torch::Tensor d_paired_with = ws.gradient_data.d_paired_with.get();
  torch::Tensor d_saddles = ws.gradient_data.d_saddles.get();
  auto traced_saddles_tensors = launch_trace_from_saddles_cuda(ws.d_data, d_paired_with, d_saddles, H, W, Nx, IS_DUAL);

  tensors_to_sad_events<IS_DUAL>(ws, traced_saddles_tensors);
}

template <typename Workspace>
void trace_raw_arcs_geometry(Workspace& ws, bool trace_max_arcs, bool trace_min_arcs) {
  RECORD_FUNCTION("trace_raw_arcs_geometry_gpu", {});
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  int num_cells = ws.num_cells;
  const auto& gdata = ws.gradient_data;
  const auto& max_arcs_len = ws.arcs_topology.max_arcs_len;
  const auto& min_arcs_len = ws.arcs_topology.min_arcs_len;

  auto dev = ws.d_data.device();
  auto int_opts = torch::TensorOptions().device(dev).dtype(torch::kInt32);
  torch::Tensor d_fast_crit_map = torch::full({num_cells}, -1, int_opts);
  auto d_maxes = gdata.d_maxes.get();
  auto d_mins = gdata.d_mins.get();
  torch::Tensor d_saddles = gdata.d_saddles.get();
  if (d_maxes.numel() > 0) d_fast_crit_map.index_put_({d_maxes}, torch::arange(d_maxes.numel(), int_opts));
  if (d_mins.numel() > 0) d_fast_crit_map.index_put_({d_mins}, torch::arange(d_mins.numel(), int_opts));
  if (d_saddles.numel() > 0) d_fast_crit_map.index_put_({d_saddles}, torch::arange(d_saddles.numel(), int_opts));
  ws.d_fast_crit_map = d_fast_crit_map;

  auto& out = ws.saddle_nodes;
  int num_saddles = max_arcs_len.size() / 2;
  if (num_saddles == 0) return;

  std::vector<int> max_offsets(num_saddles * 2 + 1, 0);
  std::vector<int> min_offsets(num_saddles * 2 + 1, 0);

  {
    RECORD_FUNCTION("prepare", {});
    for (int i = 0; i < num_saddles * 2; ++i) {
      if (trace_max_arcs) {
        max_offsets[i + 1] = max_offsets[i] + max_arcs_len[i] + 1;
      } else {
        max_offsets[i + 1] = 0;
      }
      if (trace_min_arcs) {
        min_offsets[i + 1] = min_offsets[i] + min_arcs_len[i] + 1;
      } else {
        min_offsets[i + 1] = 0;
      }
    }
    out.nodes.assign(num_saddles, SaddleNode{});
    for (int i = 0; i < num_saddles; ++i) {
      out.nodes[i].alive = 1;
      for (int k = 0; k < 2; ++k) {
        out.nodes[i].max_arcs[k] = {-1, max_offsets[2 * i + k], trace_max_arcs ? max_arcs_len[2 * i + k] : 0};
        out.nodes[i].min_arcs[k] = {-1, min_offsets[2 * i + k], trace_min_arcs ? min_arcs_len[2 * i + k] : 0};
      }
    }

    auto assign_targets = [&](const std::vector<SadEvent>& events, bool max_arcs) {
      for (const auto& event : events) {
        if (event.saddle_id < 0) continue;
        int saddle_idx = ws.hlp.fast_crit_map[event.saddle_id];
        if (saddle_idx < 0) continue;
        Arc* arcs = max_arcs ? out.nodes[saddle_idx].max_arcs : out.nodes[saddle_idx].min_arcs;
        arcs[0].target = event.c1_mid;
        arcs[1].target = event.c2_mid;
      }
    };
    if (trace_max_arcs) assign_targets(ws.arcs_topology.sorted_max_saddles, true);
    if (trace_min_arcs) assign_targets(ws.arcs_topology.sorted_min_saddles, false);
  }

  // Allocate UNINITIALIZED memory natively on CUDA
  torch::Tensor d_flat_max = torch::empty({(long)max_offsets.back()}, int_opts);
  torch::Tensor d_flat_min = torch::empty({(long)min_offsets.back()}, int_opts);

  torch::Tensor d_max_offsets =
      torch::from_blob(max_offsets.data(), {num_saddles * 2 + 1}, torch::kInt32).to(dev, /*non_blocking=*/true);
  torch::Tensor d_min_offsets =
      torch::from_blob(min_offsets.data(), {num_saddles * 2 + 1}, torch::kInt32).to(dev, /*non_blocking=*/true);
  torch::Tensor d_paired_with = gdata.d_paired_with.get();
  {
    RECORD_FUNCTION("kernel", {});
    // Extract raw pointers and dispatch to standard CUDA kernel
    launch_trace_raw_arcs_geometry_cuda(
        d_paired_with.data_ptr<int>(), d_saddles.data_ptr<int>(), d_max_offsets.data_ptr<int>(),
        d_min_offsets.data_ptr<int>(),
        trace_max_arcs ? d_flat_max.data_ptr<int>() : nullptr, trace_min_arcs ? d_flat_min.data_ptr<int>() : nullptr,
        H, W, Nx, num_saddles, trace_max_arcs, trace_min_arcs);
  }

  // Geometry is not consumed by CPU persistence. Keep the potentially large
  // coordinate arenas on-device; simplification resolves their path topology on
  // the CPU and gathers the surviving paths directly on CUDA.
  out.flat_max_geom.copy_from_tensor(d_flat_max, int_opts);
  out.flat_min_geom.copy_from_tensor(d_flat_min, int_opts);

}

template <typename Workspace>
void simplify_arcs_geometry(Workspace& ws, bool trace_max_arcs, bool trace_min_arcs) {
  RECORD_FUNCTION("simplify_arcs_geometry_cuda", {});
  auto& min_cancellations = ws.p_data.min_cancellations;
  auto& max_cancellations = ws.p_data.max_cancellations;
  auto& sn = ws.saddle_nodes;
  int num_crit_maxes = ws.gradient_data.cp.maxes.size();
  int num_crit_mins = ws.gradient_data.cp.mins.size();
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
        if (trace_max_arcs) {
          init_t_max[2 * idx] = sn.nodes[idx].max_arcs[0].target;
          init_t_max[2 * idx + 1] = sn.nodes[idx].max_arcs[1].target;
          base_max_len[2 * idx] = sn.nodes[idx].max_arcs[0].length;
          base_max_len[2 * idx + 1] = sn.nodes[idx].max_arcs[1].length;
          base_max_offset[2 * idx] = sn.nodes[idx].max_arcs[0].offset;
          base_max_offset[2 * idx + 1] = sn.nodes[idx].max_arcs[1].offset;
        }

        if (trace_min_arcs) {
          init_t_min[2 * idx] = sn.nodes[idx].min_arcs[0].target;
          init_t_min[2 * idx + 1] = sn.nodes[idx].min_arcs[1].target;
          base_min_len[2 * idx] = sn.nodes[idx].min_arcs[0].length;
          base_min_len[2 * idx + 1] = sn.nodes[idx].min_arcs[1].length;
          base_min_offset[2 * idx] = sn.nodes[idx].min_arcs[0].offset;
          base_min_offset[2 * idx + 1] = sn.nodes[idx].min_arcs[1].offset;
        }
      }
    });
  }

  auto dev = torch::kCUDA;
  auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);

  // MAXIMA PREP
  int num_max_cancels = max_cancellations.size();
  size_t safe_max_dag = std::max<size_t>(MIN_DAG_ALLOC_TOTAL, max_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR_GPU);

  // min and max are independant
  at::cuda::CUDAStream stream_max = at::cuda::getStreamFromPool();
  at::cuda::CUDAStream stream_min = at::cuda::getStreamFromPool();

  int max_dag_sz = 0;
  torch::Tensor d_max_init_t, d_max_base_lens, d_max_parent, d_max_weights, d_max_cancels, d_max_alive, d_max_pending,
      d_max_dag;

  std::vector<int> flat_max_cancels(num_max_cancels * 2);
  for (int i = 0; i < num_max_cancels; ++i) {
    flat_max_cancels[i * 2] = max_cancellations[i].s_idx;
    flat_max_cancels[i * 2 + 1] = max_cancellations[i].t_idx;
  }

  {
    at::cuda::CUDAStreamGuard guard(stream_max);
    d_max_init_t = torch::from_blob(init_t_max.data(), {N2}, torch::kInt32).to(dev, true);
    d_max_base_lens = torch::from_blob(base_max_len.data(), {N2}, torch::kInt32).to(dev, true);
    d_max_parent = torch::from_blob(max_parent.data(), {num_crit_maxes}, torch::kInt32).to(dev, true);
    d_max_weights =
        torch::from_blob(max_weight.data(), {(long)(num_crit_maxes * sizeof(GPUPathRef))}, torch::kUInt8).to(dev, true);
    d_max_dag = torch::empty({(long)(safe_max_dag * sizeof(GPUDAGNode))}, byte_opts);
    d_max_cancels = torch::from_blob(flat_max_cancels.data(), {num_max_cancels * 2}, torch::kInt32).to(dev, true);
    d_max_alive = torch::ones({num_saddles}, byte_opts);
    d_max_pending = torch::ones({num_max_cancels}, byte_opts);
  }

  int num_min_cancels = min_cancellations.size();
  size_t safe_min_dag = std::max<size_t>(MIN_DAG_ALLOC_TOTAL, min_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR_GPU);

  int min_dag_sz = 0;
  torch::Tensor d_min_init_t, d_min_base_lens, d_min_parent, d_min_weights, d_min_cancels, d_min_alive, d_min_pending,
      d_min_dag;

  std::vector<int> flat_min_cancels(num_min_cancels * 2);
  for (int i = 0; i < num_min_cancels; ++i) {
    flat_min_cancels[i * 2] = min_cancellations[i].s_idx;
    flat_min_cancels[i * 2 + 1] = min_cancellations[i].t_idx;
  }

  {
    at::cuda::CUDAStreamGuard guard(stream_min);
    d_min_init_t = torch::from_blob(init_t_min.data(), {N2}, torch::kInt32).to(dev, true);
    d_min_base_lens = torch::from_blob(base_min_len.data(), {N2}, torch::kInt32).to(dev, true);
    d_min_parent = torch::from_blob(min_parent.data(), {num_crit_mins}, torch::kInt32).to(dev, true);
    d_min_weights =
        torch::from_blob(min_weight.data(), {(long)(num_crit_mins * sizeof(GPUPathRef))}, torch::kUInt8).to(dev, true);
    d_min_dag = torch::empty({(long)(safe_min_dag * sizeof(GPUDAGNode))}, byte_opts);
    d_min_cancels = torch::from_blob(flat_min_cancels.data(), {num_min_cancels * 2}, torch::kInt32).to(dev, true);
    d_min_alive = torch::ones({num_saddles}, byte_opts);
    d_min_pending = torch::ones({num_min_cancels}, byte_opts);
  }
  {
    RECORD_FUNCTION("CUDA_Iterative_Contraction", {});
    if (trace_max_arcs) {
      launch_simplify_arcs_cuda(d_max_cancels, d_max_init_t, d_max_parent, d_max_weights, d_max_dag, d_max_base_lens,
                                d_max_alive, d_max_pending, num_max_cancels, num_crit_maxes, N2, &max_dag_sz,
                                stream_max.stream());
    }
    if (trace_min_arcs) {
      launch_simplify_arcs_cuda(d_min_cancels, d_min_init_t, d_min_parent, d_min_weights, d_min_dag, d_min_base_lens,
                                d_min_alive, d_min_pending, num_min_cancels, num_crit_mins, N2, &min_dag_sz,
                                stream_min.stream());
    }
  }
  stream_max.synchronize();
  stream_min.synchronize();

  // RETRIEVE AND ASSEMBLE
  // Use copy_ instead of .cpu() to avoid memory allocation overhead
  torch::from_blob(max_parent.data(), {num_crit_maxes}, torch::kInt32).copy_(d_max_parent, /*non_blocking=*/false);
  torch::from_blob(min_parent.data(), {num_crit_mins}, torch::kInt32).copy_(d_min_parent, /*non_blocking=*/false);

  std::vector<uint8_t> max_alive(num_saddles);
  torch::from_blob(max_alive.data(), {num_saddles}, torch::kUInt8).copy_(d_max_alive, /*non_blocking=*/false);

  std::vector<uint8_t> min_alive(num_saddles);
  torch::from_blob(min_alive.data(), {num_saddles}, torch::kUInt8).copy_(d_min_alive, /*non_blocking=*/false);

  torch::from_blob(max_weight.data(), {(long)(num_crit_maxes * sizeof(GPUPathRef))}, torch::kUInt8)
      .copy_(d_max_weights, /*non_blocking=*/false);
  torch::from_blob(min_weight.data(), {(long)(num_crit_mins * sizeof(GPUPathRef))}, torch::kUInt8)
      .copy_(d_min_weights, /*non_blocking=*/false);

  size_t required_max_dag = max_dag_sz + num_crit_maxes + N2;
  std::vector<GPUDAGNode> max_dag(required_max_dag);
  if (max_dag_sz > 0) {
    torch::from_blob(max_dag.data(), {(long)(max_dag_sz * sizeof(GPUDAGNode))}, torch::kUInt8)
        .copy_(d_max_dag.slice(0, 0, max_dag_sz * sizeof(GPUDAGNode)), /*non_blocking=*/false);
  }

  size_t required_min_dag = min_dag_sz + num_crit_mins + N2;
  std::vector<GPUDAGNode> min_dag(required_min_dag);
  if (min_dag_sz > 0) {
    torch::from_blob(min_dag.data(), {(long)(min_dag_sz * sizeof(GPUDAGNode))}, torch::kUInt8)
        .copy_(d_min_dag.slice(0, 0, min_dag_sz * sizeof(GPUDAGNode)), /*non_blocking=*/false);
  }

  // flatten the union find tree
  {
    RECORD_FUNCTION("Final_Path_Compression_Sweep", {});

    auto MergePathsCPU = [&](GPUPathRef p1, GPUPathRef p2, GPUDAGNode* dag, int& dag_sz,
                             const int* base_lens) -> GPUPathRef {
      if (p1.id == -1) return p2;
      if (p2.id == -1) return p1;
      int len1 = (p1.id < N2) ? base_lens[p1.id] : dag[p1.id - N2].total_len;
      int len2 = (p2.id < N2) ? base_lens[p2.id] : dag[p2.id - N2].total_len;
      int new_id = dag_sz + N2;
      dag[dag_sz++] = {p1.id, p2.id, p1.fwd, p2.fwd, len1 + len2, 0};  // 0 is the _padding
      return {new_id, 1};
    };

    auto FindCPU = [&](auto& self, int i, int* parent, GPUPathRef* weight, GPUDAGNode* dag, int& dag_sz,
                       const int* blen) -> int {
      if (i == -1) return -1;
      if (parent[i] == i) return i;
      int p = parent[i];
      int root = self(self, p, parent, weight, dag, dag_sz, blen);
      if (p != -1 && weight[p].id != -1) {
        weight[i] = MergePathsCPU(weight[i], weight[p], dag, dag_sz, blen);
      }
      parent[i] = root;
      return root;
    };

    tbb::parallel_invoke(
        [&]() {
          for (int i = 0; i < num_crit_maxes; ++i) {
            FindCPU(FindCPU, i, max_parent.data(), max_weight.data(), max_dag.data(), max_dag_sz, base_max_len.data());
          }
        },
        [&]() {
          for (int i = 0; i < num_crit_mins; ++i) {
            FindCPU(FindCPU, i, min_parent.data(), min_weight.data(), min_dag.data(), min_dag_sz, base_min_len.data());
          }
        });
  }

  cpu::assemble_simplified_geometry<GPUDAGNode, GPUPathRef>(
      ws, max_alive, min_alive, init_t_max, init_t_min, base_max_len, base_min_len, base_max_offset, base_min_offset,
      max_parent, min_parent, max_weight, min_weight, max_dag, max_dag_sz, min_dag, min_dag_sz);
}
}  // namespace gpu
