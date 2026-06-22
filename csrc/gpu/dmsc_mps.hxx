#include <ATen/record_function.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <iostream>

#include "../cpu/arcs_simplification.hxx"
#include "../cpu/gradient.hxx"
#include "../cpu/persistence_struct.hxx"
#include "./arcs_geometry_struct.hxx"
#include "./arcs_simplification_struct.hxx"
#include "./arcs_topology_helpers.hxx"
#include "./gradient_struct.hxx"

#define MAX_DAG_ALLOC_PER_PAIR_GPU 15

void launch_gradient_metal(torch::Tensor d_data, torch::Tensor d_paired_with, int H, int W, bool is_dual);
gpu::CriticalPointsAsTensors launch_extract_critical_points_metal(torch::Tensor d_paired_with, int H, int W, int Nx);
gpu::TracedSaddlesTensors launch_trace_from_saddles_metal(torch::Tensor d_data, torch::Tensor d_paired_with,
                                                          torch::Tensor d_saddles, int H, int W, int Nx, bool is_dual);
void launch_cell_groups_metal(torch::Tensor d_data, torch::Tensor d_paired_with, torch::Tensor d_fast_crit_map,
                              torch::Tensor d_uf_parent, torch::Tensor d_crits, torch::Tensor d_fast_region_id,
                              torch::Tensor d_out_groups, int H, int W, int Nx, bool is_dual, bool trace_faces);

void launch_trace_raw_arcs_geometry_metal(torch::Tensor d_paired_with, torch::Tensor d_fast_crit_map,
                                          torch::Tensor d_crit_saddles, torch::Tensor d_max_offsets,
                                          torch::Tensor d_min_offsets, torch::Tensor d_flat_max,
                                          torch::Tensor d_flat_min, torch::Tensor d_saddle_nodes, int H, int W, int Nx,
                                          int num_saddles, bool trace_max_arcs, bool trace_min_arcs);

void launch_simplify_arcs_metal(torch::Tensor d_cancels, torch::Tensor d_init_t, torch::Tensor d_parent_ptrs,
                                torch::Tensor d_weights, torch::Tensor d_dag, torch::Tensor d_base_lens,
                                torch::Tensor d_alive, torch::Tensor d_pending, int num_cancels, int num_extrema,
                                int N2, int* host_dag_sz);

namespace gpu {

CriticalPoints tensors_to_critical_points(const gpu::CriticalPointsAsTensors& cpt) {
  RECORD_FUNCTION("tensors_to_critical_points", {});
  CriticalPoints cp;

  torch::Tensor cpu_mins = cpt.mins.cpu().contiguous();
  torch::Tensor cpu_saddles = cpt.saddles.cpu().contiguous();
  torch::Tensor cpu_maxes = cpt.maxes.cpu().contiguous();

  cp.mins.assign(cpu_mins.data_ptr<int>(), cpu_mins.data_ptr<int>() + cpu_mins.numel());
  cp.saddles.assign(cpu_saddles.data_ptr<int>(), cpu_saddles.data_ptr<int>() + cpu_saddles.numel());
  cp.maxes.assign(cpu_maxes.data_ptr<int>(), cpu_maxes.data_ptr<int>() + cpu_maxes.numel());

  return cp;
}

// TODO: we don t need to return all this data, it waste time, remove it at some point
template <bool IS_DUAL = false, typename Workspace>
void compute_gradient(Workspace& ws, const torch::Tensor& scalar_field) {
  RECORD_FUNCTION("compute_gradient_and_crit_points_gpu", {});
  int H = ws.H;
  int W = ws.W;
  int Nx = W + 1;
  int num_cells = ws.num_cells;

  // Allocate PyTorch Tensors on MPS
  ws.d_data = scalar_field.to(torch::kMPS).contiguous();
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS);
  torch::Tensor d_paired_with = ws.gradient_data.d_paired_with.request_full({num_cells}, opts, -1);

  {
    RECORD_FUNCTION("KERNEL_GRADIENT", {});
    launch_gradient_metal(ws.d_data, d_paired_with, H, W, IS_DUAL);
  }

  {
    RECORD_FUNCTION("KERNEL_CRIT_POINTS", {});
    auto cpt = launch_extract_critical_points_metal(d_paired_with, H, W, Nx);
    ws.gradient_data.cp = tensors_to_critical_points(cpt);

    // Push critical points to GPU natively
    ws.gradient_data.d_maxes.copy_from_tensor(cpt.maxes, opts);
    ws.gradient_data.d_mins.copy_from_tensor(cpt.mins, opts);
    ws.gradient_data.d_saddles.copy_from_tensor(cpt.saddles, opts);
  }

  // update fast_crit_map and other helpers (CPU)
  torch::Tensor scalar_field_cpu = ws.d_data.cpu();
  cpu::update_gradient_helpers<IS_DUAL>(ws, scalar_field_cpu.data_ptr<float>());
}

template <bool IS_DUAL = false, typename Workspace>
void compute_cell_groups(Workspace& ws, bool trace_face_groups, bool trace_vertex_groups) {
  RECORD_FUNCTION("cell_groups_gpu", {});
  const auto& gdata = ws.gradient_data;
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;
  int num_cells = ws.num_cells;

  auto& uf_max = ws.p_data.uf_max;
  auto& uf_min = ws.p_data.uf_min;
  const auto& max_alive = ws.p_data.max_alive;
  const auto& min_alive = ws.p_data.min_alive;
  const auto& crit_maxes = ws.gradient_data.cp.maxes;
  const auto& crit_mins = ws.gradient_data.cp.mins;
  const auto& fast_crit_map = ws.hlp.fast_crit_map;
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

  torch::Tensor d_fast_crit_map = torch::full({num_cells}, -1, i_opts);
  if (num_crit_maxes > 0) d_fast_crit_map.index_put_({d_crit_maxes}, torch::arange((long)num_crit_maxes, i_opts));
  if (num_crit_mins > 0) d_fast_crit_map.index_put_({d_crit_mins}, torch::arange((long)num_crit_mins, i_opts));
  if (d_crit_saddles.numel() > 0)
    d_fast_crit_map.index_put_({d_crit_saddles}, torch::arange((long)d_crit_saddles.numel(), i_opts));

  torch::Tensor d_uf_max_parent = torch::from_blob((void*)uf_max.parent.data(), {(long)num_crit_maxes}, torch::kInt32)
                                      .to(dev, /*non_blocking=*/true);

  torch::Tensor d_uf_min_parent = torch::from_blob((void*)uf_min.parent.data(), {(long)num_crit_mins}, torch::kInt32)
                                      .to(dev, /*non_blocking=*/true);

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

  // Pre-allocate output on MPS, -2 means unknown
  torch::Tensor d_out_face_groups = ws.cell_groups.face_groups.request_full({H + 1, W + 1}, i_opts, -2);
  torch::Tensor d_out_vertex_groups = ws.cell_groups.vertex_groups.request_full({H, W}, i_opts, -2);
  if (trace_face_groups) {
    RECORD_FUNCTION("trace_faces", {});
    launch_cell_groups_metal(ws.d_data, gdata.d_paired_with.get(), d_fast_crit_map, d_uf_max_parent, d_crit_maxes,
                             d_fast_region, d_out_face_groups, H, W, Nx, IS_DUAL, /*trace_faces=*/true);
  }
  if (trace_vertex_groups) {
    RECORD_FUNCTION("trace_vertices", {});
    launch_cell_groups_metal(ws.d_data, gdata.d_paired_with.get(), d_fast_crit_map, d_uf_min_parent, d_crit_mins,
                             d_fast_region, d_out_vertex_groups, H, W, Nx, IS_DUAL, /*trace_faces=*/false);
  }
}

// NOTE: trace_from_saddles now accepts GradientData by reference so it can access the MPS tensors
template <bool IS_DUAL = false, typename Workspace>
void trace_from_saddles(Workspace& ws) {
  RECORD_FUNCTION("trace_from_saddles", {});
  int H = ws.H;
  int W = ws.W;
  int Nx = ws.Nx;

  auto traced_saddles_tensors = launch_trace_from_saddles_metal(ws.d_data, ws.gradient_data.d_paired_with.get(),
                                                                ws.gradient_data.d_saddles.get(), H, W, Nx, IS_DUAL);

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

  torch::Tensor d_fast_crit_map = torch::from_blob((void*)ws.hlp.fast_crit_map.data(), {num_cells}, torch::kInt32)
                                      .to(torch::kMPS, /*non_blocking=*/true);

  auto& out = ws.saddle_nodes;
  int num_saddles = max_arcs_len.size() / 2;
  if (num_saddles == 0) return;

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

  auto mps_opts = torch::TensorOptions().device(torch::kMPS);
  auto int_opts = mps_opts.dtype(torch::kInt32);

  // Allocate UNINITIALIZED memory on the GPU
  torch::Tensor d_saddle_nodes =
      torch::empty({(long)(num_saddles * sizeof(SaddleNode))}, mps_opts.dtype(torch::kUInt8));
  torch::Tensor d_flat_max = torch::empty({(long)max_offsets.back()}, int_opts);
  torch::Tensor d_flat_min = torch::empty({(long)min_offsets.back()}, int_opts);

  torch::Tensor d_max_offsets =
      torch::from_blob(max_offsets.data(), {num_saddles * 2 + 1}, torch::kInt32).to(torch::kMPS, /*non_blocking=*/true);
  torch::Tensor d_min_offsets =
      torch::from_blob(min_offsets.data(), {num_saddles * 2 + 1}, torch::kInt32).to(torch::kMPS, /*non_blocking=*/true);

  {
    RECORD_FUNCTION("kernel", {});
    launch_trace_raw_arcs_geometry_metal(gdata.d_paired_with.get(), d_fast_crit_map, gdata.d_saddles.get(),
                                         d_max_offsets, d_min_offsets, trace_max_arcs ? d_flat_max : torch::Tensor(),
                                         trace_min_arcs ? d_flat_min : torch::Tensor(), d_saddle_nodes, H, W, Nx,
                                         num_saddles, trace_max_arcs, trace_min_arcs);
  }

  // out.flat_max_geom = d_flat_max.cpu();
  // out.flat_min_geom = d_flat_min.cpu();
  out.flat_max_geom.copy_from_tensor(d_flat_max, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
  out.flat_min_geom.copy_from_tensor(d_flat_min, torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));

  torch::from_blob(out.nodes.data(), {(long)(num_saddles * sizeof(SaddleNode))}, torch::kUInt8).copy_(d_saddle_nodes);
}

template <typename Workspace>
void simplify_arcs_geometry(Workspace& ws, bool trace_max_arcs, bool trace_min_arcs) {
  RECORD_FUNCTION("simplify_arcs_geometry_metal", {});
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

  // Extract Base Data Concurrently
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

  // 3. Move data to Metal PyTorch Tensors
  auto dev = torch::kMPS;
  // auto i_opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
  auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(dev);

  // Move the data for maxima and minima ...
  // TODO: optimize
  int num_max_cancels = max_cancellations.size();
  // WARNING: this could be a lot of memory !
  size_t safe_max_dag = std::max(MIN_DAG_ALLOC_TOTAL, max_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR_GPU);
  // printf("Allocating %ld MB for DAG\n", sizeof(GPUDAGNode) * safe_max_dag / 1024 / 1024);

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

  int num_min_cancels = min_cancellations.size();
  size_t safe_min_dag = std::max(MIN_DAG_ALLOC_TOTAL, min_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR_GPU);

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

  {
    RECORD_FUNCTION("Metal_Iterative_Contraction", {});

    if (trace_max_arcs) {
      launch_simplify_arcs_metal(d_max_cancels, d_max_init_t, d_max_parent, d_max_weights, d_max_dag, d_max_base_lens,
                                 d_max_alive, d_max_pending, num_max_cancels, num_crit_maxes, N2, &max_dag_sz);
    }
    if (trace_min_arcs) {
      launch_simplify_arcs_metal(d_min_cancels, d_min_init_t, d_min_parent, d_min_weights, d_min_dag, d_min_base_lens,
                                 d_min_alive, d_min_pending, num_min_cancels, num_crit_mins, N2, &min_dag_sz);
    }
  }

  // Retrieve data and prepare it for assemble_simplified_geometry hapenning on CPU
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

  // Allocate enough room for the GPU nodes + CPU Compression Sweep + CPU Prefix Sum
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

  // CPU version, templated with GPU structures
  cpu::assemble_simplified_geometry<GPUDAGNode, GPUPathRef>(
      ws, max_alive, min_alive, init_t_max, init_t_min, base_max_len, base_min_len, base_max_offset, base_min_offset,
      max_parent, min_parent, max_weight, min_weight, max_dag, max_dag_sz, min_dag, min_dag_sz);
}
}  // namespace gpu