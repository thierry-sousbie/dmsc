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
#include "./arcs_simplification_struct.hxx"
#include "./trace_saddles_helpers.hxx"

namespace gpu {
// Exactly as defined in metal/metal_backend.mm
struct CriticalPointsAsTensors {
  torch::Tensor mins;
  torch::Tensor saddles;
  torch::Tensor maxes;
};
}  // namespace gpu

void launch_gradient_metal(torch::Tensor d_data, torch::Tensor d_paired_with, int H, int W, bool is_dual);
gpu::CriticalPointsAsTensors launch_extract_critical_points_metal(torch::Tensor d_paired_with, int H, int W, int Nx,
                                                                  bool is_dual);
gpu::TracedSaddlesTensors launch_trace_from_saddles_metal(torch::Tensor d_data, torch::Tensor d_paired_with,
                                                          torch::Tensor d_saddles, int sad_count, int H, int W, int Nx,
                                                          bool is_dual);
void launch_cell_groups_metal(torch::Tensor d_data, torch::Tensor d_paired_with, torch::Tensor d_fast_crit_map,
                              torch::Tensor d_uf_parent, torch::Tensor d_crits, torch::Tensor d_fast_region_id,
                              torch::Tensor d_out_groups, int H, int W, int Nx, bool is_dual, bool trace_faces);

void launch_trace_raw_arcs_geometry_metal(torch::Tensor d_paired_with, torch::Tensor d_fast_crit_map,
                                          torch::Tensor d_crit_saddles, torch::Tensor d_max_offsets,
                                          torch::Tensor d_min_offsets, torch::Tensor d_flat_max,
                                          torch::Tensor d_flat_min, torch::Tensor d_saddle_nodes, int H, int W, int Nx,
                                          int num_saddles);

void launch_simplify_arcs_metal(torch::Tensor d_cancels, torch::Tensor d_init_t, torch::Tensor d_parent_ptrs,
                                torch::Tensor d_weights, torch::Tensor d_dag, torch::Tensor d_base_lens,
                                torch::Tensor d_alive, torch::Tensor d_pending, int num_cancels, int num_extrema,
                                int N2, int* host_dag_sz);

namespace gpu {

struct CriticalPoints {
  std::vector<int> maxes;
  std::vector<int> saddles;
  std::vector<int> mins;
};

struct GradientData {
  CriticalPoints cp;
  std::vector<int> paired_with;

  // Handles for GPU memory. keep data between calls so we don t need to reallocate
  torch::Tensor d_data;
  torch::Tensor d_paired_with;
  torch::Tensor d_saddles;
};

CriticalPoints tensors_to_critical_points(const CriticalPointsAsTensors& cpt) {
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
template <bool IS_DUAL = false>
GradientData compute_gradient(torch::Tensor& active_field) {
  RECORD_FUNCTION("compute_gradient_and_crit_points_gpu", {});
  int H = active_field.size(0);
  int W = active_field.size(1);
  int Nx = W + 1;
  int num_cells = 4 * (H + 1) * (W + 1);

  // 1. Allocate PyTorch Tensors on MPS
  torch::Tensor d_data = active_field.to(torch::kMPS).contiguous();
  torch::Tensor d_paired_with =
      torch::empty({num_cells}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS));

  // 2. Dispatch
  {
    RECORD_FUNCTION("KERNEL_GRADIENT", {});
    launch_gradient_metal(d_data, d_paired_with, H, W, IS_DUAL);
  }
  // 3. Copy paired_with back to CPU (Needed for Phase 3 simplification)
  torch::Tensor cpu_paired = d_paired_with.cpu();
  std::vector<int> paired_with(cpu_paired.data_ptr<int>(), cpu_paired.data_ptr<int>() + num_cells);

  // is_dual is false because we computed minus the discrete gradient over the dual cells
  auto cpt = launch_extract_critical_points_metal(d_paired_with, H, W, Nx, false);
  CriticalPoints cp = tensors_to_critical_points(cpt);

  // Return the CPU data AND the GPU handles so trace_from_saddles can use them instantly
  return {std::move(cp), std::move(paired_with), d_data, d_paired_with, cpt.saddles};
}

template <bool IS_DUAL = false>
void compute_cell_groups(const GradientData& gdata, const int* fast_crit_map, const int* uf_max_parent,
                         const int* crit_maxes, const int* uf_min_parent, const int* crit_mins,
                         const int* fast_region_id, int* out_face_groups, int* out_vertex_groups, size_t num_crit_maxes,
                         size_t num_crit_mins) {
  RECORD_FUNCTION("cell_groups_gpu", {});
  int H = gdata.d_data.size(0);
  int W = gdata.d_data.size(1);
  int Nx = W + 1;
  int num_cells = 4 * (H + 1) * (W + 1);
  int num_faces = (H + 1) * (W + 1);
  int num_vertices = (H) * (W);

  auto i_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS);

  torch::Tensor d_fast_crit_map = torch::from_blob((void*)fast_crit_map, {num_cells}, torch::kInt32).to(torch::kMPS);

  torch::Tensor d_uf_max_parent =
      torch::from_blob((void*)uf_max_parent, {(long)num_crit_maxes}, torch::kInt32).to(torch::kMPS);
  torch::Tensor d_crit_maxes =
      torch::from_blob((void*)crit_maxes, {(long)num_crit_maxes}, torch::kInt32).to(torch::kMPS);

  torch::Tensor d_uf_min_parent =
      torch::from_blob((void*)uf_min_parent, {(long)num_crit_mins}, torch::kInt32).to(torch::kMPS);
  torch::Tensor d_crit_mins = torch::from_blob((void*)crit_mins, {(long)num_crit_mins}, torch::kInt32).to(torch::kMPS);

  torch::Tensor d_fast_region = torch::from_blob((void*)fast_region_id, {num_cells}, torch::kInt32).to(torch::kMPS);

  // Pre-allocate output on MPS, -2 means unknonw
  torch::Tensor d_out_face_groups = torch::full({num_faces}, -2, i_opts);
  torch::Tensor d_out_vertex_groups = torch::full({num_vertices}, -2, i_opts);
  {
    RECORD_FUNCTION("trace_faces", {});
    launch_cell_groups_metal(gdata.d_data, gdata.d_paired_with, d_fast_crit_map, d_uf_max_parent, d_crit_maxes,
                             d_fast_region, d_out_face_groups, H, W, Nx, IS_DUAL, /*trace_faces=*/true);
  }
  {
    RECORD_FUNCTION("trace_vertices", {});
    launch_cell_groups_metal(gdata.d_data, gdata.d_paired_with, d_fast_crit_map, d_uf_min_parent, d_crit_mins,
                             d_fast_region, d_out_vertex_groups, H, W, Nx, IS_DUAL, /*trace_faces=*/false);
  }

  // Bring the result back to CPU memory synchronously
  {
    RECORD_FUNCTION("Memcpy_Device_To_Host", {});
    // Wrap your existing raw pointers
    auto cpu_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor cpu_face_wrapper = torch::from_blob(out_face_groups, {num_faces}, cpu_opts);
    torch::Tensor cpu_vert_wrapper = torch::from_blob(out_vertex_groups, {num_vertices}, cpu_opts);

    // 2. Direct copy (GPU -> Pre-existing CPU RAM)
    cpu_face_wrapper.copy_(d_out_face_groups, /*non_blocking=*/false);
    cpu_vert_wrapper.copy_(d_out_vertex_groups, /*non_blocking=*/false);
  }
}

// NOTE: trace_from_saddles now accepts GradientData by reference so it can access the MPS tensors
template <bool IS_DUAL = false>
TracedSaddlesVectors trace_from_saddles(const GradientData& gdata, int saddles_count, int H, int W, int Nx) {
  RECORD_FUNCTION("trace_from_saddles", {});

  // auto traced_saddles_tensors = launch_trace_from_saddles_metal(gdata.d_data, gdata.d_paired_with, gdata.d_saddles,
  //                                                               saddles_count, H, W, Nx, IS_DUAL);
  auto traced_saddles_tensors = launch_trace_from_saddles_metal(gdata.d_data, gdata.d_paired_with, gdata.d_saddles,
                                                                saddles_count, H, W, Nx, IS_DUAL);

  TracedSaddlesVectors traced_saddles_vectors = tensors_to_sad_events<IS_DUAL>(traced_saddles_tensors, Nx);

  return traced_saddles_vectors;
}

SaddleNodes trace_raw_arcs_geometry(const GradientData& gdata, const int* fast_crit_map,
                                    const std::vector<int>& max_arcs_len, const std::vector<int>& min_arcs_len) {
  int H = gdata.d_data.size(0);
  int W = gdata.d_data.size(1);
  int Nx = W + 1;
  int num_cells = 4 * (H + 1) * (W + 1);
  RECORD_FUNCTION("trace_raw_arcs_geometry_gpu", {});

  torch::Tensor d_fast_crit_map =
      torch::from_blob((void*)fast_crit_map, {num_cells}, torch::kInt32).to(torch::kMPS, /*non_blocking=*/true);

  SaddleNodes out;
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
    out.saddle_nodes.resize(num_saddles);
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
    launch_trace_raw_arcs_geometry_metal(gdata.d_paired_with, d_fast_crit_map, gdata.d_saddles, d_max_offsets,
                                         d_min_offsets, d_flat_max, d_flat_min, d_saddle_nodes, H, W, Nx, num_saddles);
  }

  out.flat_max_geom = d_flat_max.cpu();
  out.flat_min_geom = d_flat_min.cpu();

  torch::from_blob(out.saddle_nodes.data(), {(long)(num_saddles * sizeof(SaddleNode))}, torch::kUInt8)
      .copy_(d_saddle_nodes);

  return out;
}

// // Redefined because of alignment constraints with metal
// struct GPUDAGNode {
//   int left_id;
//   int right_id;
//   int left_fwd;
//   int right_fwd;
//   int total_len;
//   int _padding;  // 24-byte alignment
// };

// struct GPUPathRef {
//   int id;
//   int fwd;
// };

void simplify_arcs_geometry(SaddleNodes& sn, int num_crit_maxes, int num_crit_mins,
                            std::vector<CancelEvent>& min_cancellations, std::vector<CancelEvent>& max_cancellations) {
  RECORD_FUNCTION("simplify_arcs_geometry_metal_dispatch", {});

  int num_saddles = sn.saddle_nodes.size();
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
        init_t_max[2 * idx] = sn.saddle_nodes[idx].max_arcs[0].target;
        init_t_max[2 * idx + 1] = sn.saddle_nodes[idx].max_arcs[1].target;
        init_t_min[2 * idx] = sn.saddle_nodes[idx].min_arcs[0].target;
        init_t_min[2 * idx + 1] = sn.saddle_nodes[idx].min_arcs[1].target;

        base_max_len[2 * idx] = sn.saddle_nodes[idx].max_arcs[0].length;
        base_max_len[2 * idx + 1] = sn.saddle_nodes[idx].max_arcs[1].length;
        base_max_offset[2 * idx] = sn.saddle_nodes[idx].max_arcs[0].offset;
        base_max_offset[2 * idx + 1] = sn.saddle_nodes[idx].max_arcs[1].offset;

        base_min_len[2 * idx] = sn.saddle_nodes[idx].min_arcs[0].length;
        base_min_len[2 * idx + 1] = sn.saddle_nodes[idx].min_arcs[1].length;
        base_min_offset[2 * idx] = sn.saddle_nodes[idx].min_arcs[0].offset;
        base_min_offset[2 * idx + 1] = sn.saddle_nodes[idx].min_arcs[1].offset;
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
  size_t safe_max_dag = std::max(MIN_DAG_ALLOC_TOTAL, max_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR);

  torch::Tensor d_max_init_t = torch::from_blob(init_t_max.data(), {N2}, torch::kInt32).to(dev);
  torch::Tensor d_max_base_lens = torch::from_blob(base_max_len.data(), {N2}, torch::kInt32).to(dev);
  torch::Tensor d_max_parent = torch::from_blob(max_parent.data(), {num_crit_maxes}, torch::kInt32).to(dev);

  torch::Tensor d_max_weights =
      torch::from_blob(max_weight.data(), {(long)(num_crit_maxes * sizeof(GPUPathRef))}, torch::kUInt8).to(dev);
  torch::Tensor d_max_dag = torch::empty({(long)(safe_max_dag * sizeof(GPUDAGNode))}, byte_opts);

  std::vector<int> flat_max_cancels(num_max_cancels * 2);
  for (int i = 0; i < num_max_cancels; ++i) {
    flat_max_cancels[i * 2] = max_cancellations[i].s_idx;
    flat_max_cancels[i * 2 + 1] = max_cancellations[i].t_idx;
  }
  torch::Tensor d_max_cancels = torch::from_blob(flat_max_cancels.data(), {num_max_cancels * 2}, torch::kInt32).to(dev);

  torch::Tensor d_max_alive = torch::ones({num_saddles}, byte_opts);
  torch::Tensor d_max_pending = torch::ones({num_max_cancels}, byte_opts);
  int max_dag_sz = 0;

  int num_min_cancels = min_cancellations.size();
  size_t safe_min_dag = std::max(MIN_DAG_ALLOC_TOTAL, min_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR);

  torch::Tensor d_min_init_t = torch::from_blob(init_t_min.data(), {N2}, torch::kInt32).to(dev);
  torch::Tensor d_min_base_lens = torch::from_blob(base_min_len.data(), {N2}, torch::kInt32).to(dev);
  torch::Tensor d_min_parent = torch::from_blob(min_parent.data(), {num_crit_mins}, torch::kInt32).to(dev);

  torch::Tensor d_min_weights =
      torch::from_blob(min_weight.data(), {(long)(num_crit_mins * sizeof(GPUPathRef))}, torch::kUInt8).to(dev);
  torch::Tensor d_min_dag = torch::empty({(long)(safe_min_dag * sizeof(GPUDAGNode))}, byte_opts);

  std::vector<int> flat_min_cancels(num_min_cancels * 2);
  for (int i = 0; i < num_min_cancels; ++i) {
    flat_min_cancels[i * 2] = min_cancellations[i].s_idx;
    flat_min_cancels[i * 2 + 1] = min_cancellations[i].t_idx;
  }
  torch::Tensor d_min_cancels = torch::from_blob(flat_min_cancels.data(), {num_min_cancels * 2}, torch::kInt32).to(dev);

  torch::Tensor d_min_alive = torch::ones({num_saddles}, byte_opts);
  torch::Tensor d_min_pending = torch::ones({num_min_cancels}, byte_opts);
  int min_dag_sz = 0;

  {
    RECORD_FUNCTION("Metal_Iterative_Contraction", {});

    launch_simplify_arcs_metal(d_max_cancels, d_max_init_t, d_max_parent, d_max_weights, d_max_dag, d_max_base_lens,
                               d_max_alive, d_max_pending, num_max_cancels, num_crit_maxes, N2, &max_dag_sz);
    launch_simplify_arcs_metal(d_min_cancels, d_min_init_t, d_min_parent, d_min_weights, d_min_dag, d_min_base_lens,
                               d_min_alive, d_min_pending, num_min_cancels, num_crit_mins, N2, &min_dag_sz);
  }

  // Retrieve data and prepare it for assemble_simplified_geometry hapenning on CPU
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

  // Allocate enough room for the GPU nodes + CPU Compression Sweep + CPU Prefix Sum
  size_t required_max_dag = max_dag_sz + num_crit_maxes + N2;
  std::vector<GPUDAGNode> max_dag(required_max_dag);
  if (max_dag_sz > 0) {
    torch::Tensor cpu_max_dag = d_max_dag.cpu();
    std::memcpy(max_dag.data(), cpu_max_dag.data_ptr(), max_dag_sz * sizeof(GPUDAGNode));
  }

  size_t required_min_dag = min_dag_sz + num_crit_mins + N2;
  std::vector<GPUDAGNode> min_dag(required_min_dag);
  if (min_dag_sz > 0) {
    torch::Tensor cpu_min_dag = d_min_dag.cpu();
    std::memcpy(min_dag.data(), cpu_min_dag.data_ptr(), min_dag_sz * sizeof(GPUDAGNode));
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
  assemble_simplified_geometry<GPUDAGNode, GPUPathRef>(
      sn, max_alive, min_alive, init_t_max, init_t_min, base_max_len, base_min_len, base_max_offset, base_min_offset,
      max_parent, min_parent, max_weight, min_weight, max_dag, max_dag_sz, min_dag, min_dag_sz);
}
}  // namespace gpu