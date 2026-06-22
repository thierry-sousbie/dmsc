#pragma once

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "./arcs_geometry_struct.hxx"

// How many dag nodes to preallocate per persistence pair. That should be the upper bound
// for the average depth of the cancellation tree
#define MAX_DAG_ALLOC_PER_PAIR 5
#define MIN_DAG_ALLOC_TOTAL (1UL << 18)

namespace cpu {

struct DAGNode {
  int left_id;
  int right_id;
  bool left_fwd;
  bool right_fwd;
  int total_len;
};

struct PathRef {
  int id;
  bool fwd;
};

// This is templated because we ll use different structures for metal because of alignment
template <typename NodeType, typename RefType, typename Workspace>
void assemble_simplified_geometry(Workspace& ws, const std::vector<uint8_t>& max_alive,
                                  const std::vector<uint8_t>& min_alive, const std::vector<int>& init_t_max,
                                  const std::vector<int>& init_t_min, const std::vector<int>& base_max_len,
                                  const std::vector<int>& base_min_len, const std::vector<int>& base_max_offset,
                                  const std::vector<int>& base_min_offset, const std::vector<int>& max_parent,
                                  const std::vector<int>& min_parent, const std::vector<RefType>& max_weight,
                                  const std::vector<RefType>& min_weight, std::vector<NodeType>& max_dag,
                                  int& max_dag_sz, std::vector<NodeType>& min_dag, int& min_dag_sz) {
  RECORD_FUNCTION("assemble_simplified_geometry", {});
  auto& sn = ws.saddle_nodes;

  int num_saddles = sn.nodes.size();
  int N2 = num_saddles * 2;

  const int* p_max_parent = max_parent.data();
  const int* p_min_parent = min_parent.data();
  const RefType* p_max_weight = max_weight.data();
  const RefType* p_min_weight = min_weight.data();
  NodeType* p_max_dag = max_dag.data();
  NodeType* p_min_dag = min_dag.data();
  const int* p_base_max_len = base_max_len.data();
  const int* p_base_min_len = base_min_len.data();

  using fwd_type = decltype(RefType::fwd);

  auto MergePaths = [&](RefType p1, RefType p2, NodeType* dag, int& dag_sz, const int* base_lens) -> RefType {
    if (p1.id == -1) return p2;
    if (p2.id == -1) return p1;

    TORCH_CHECK((size_t)dag_sz < max_dag.size(),
                "The cancellation tree is too deep to merge path geometries."
                "Please increase MAX_DAG_ALLOC_PER_PAIR in the C++ backend. ");

    int len1 = (p1.id < N2) ? base_lens[p1.id] : dag[p1.id - N2].total_len;
    int len2 = (p2.id < N2) ? base_lens[p2.id] : dag[p2.id - N2].total_len;
    int new_id = dag_sz + N2;

    // C++ aggregate initialization zeroes out any unspecified trailing members (e.g., GPUDAGNode._padding)
    dag[dag_sz++] = {p1.id, p2.id, p1.fwd, p2.fwd, len1 + len2};
    return {new_id, (fwd_type)1};
  };

  std::vector<int> final_max_len(N2, 0), final_min_len(N2, 0);
  std::vector<int> final_max_offset(N2 + 1, 0), final_min_offset(N2 + 1, 0);
  std::vector<RefType> final_max_path(N2, {-1, (fwd_type)1});
  std::vector<RefType> final_min_path(N2, {-1, (fwd_type)1});

  {
    RECORD_FUNCTION("Prefix_Sum_Resolution", {});

    tbb::parallel_invoke(
        // Maxima Resolution & Prefix Sum
        [&]() {
          for (int i = 0; i < num_saddles; ++i) {
            if (!(max_alive[i] && min_alive[i])) continue;

            int TM0 = init_t_max[2 * i];
            if (TM0 != -1) {
              final_max_path[2 * i] =
                  MergePaths({2 * i, (fwd_type)1}, p_max_weight[TM0], p_max_dag, max_dag_sz, p_base_max_len);
              final_max_len[2 * i] = (final_max_path[2 * i].id < N2)
                                         ? p_base_max_len[final_max_path[2 * i].id]
                                         : p_max_dag[final_max_path[2 * i].id - N2].total_len;
            }

            int TM1 = init_t_max[2 * i + 1];
            if (TM1 != -1) {
              final_max_path[2 * i + 1] =
                  MergePaths({2 * i + 1, (fwd_type)1}, p_max_weight[TM1], p_max_dag, max_dag_sz, p_base_max_len);
              final_max_len[2 * i + 1] = (final_max_path[2 * i + 1].id < N2)
                                             ? p_base_max_len[final_max_path[2 * i + 1].id]
                                             : p_max_dag[final_max_path[2 * i + 1].id - N2].total_len;
            }
          }
          for (int i = 0; i < N2; ++i) final_max_offset[i + 1] = final_max_offset[i] + final_max_len[i];
        },
        // Minima Resolution & Prefix Sum
        [&]() {
          for (int i = 0; i < num_saddles; ++i) {
            if (!(max_alive[i] && min_alive[i])) continue;

            int TN0 = init_t_min[2 * i];
            if (TN0 != -1) {
              final_min_path[2 * i] =
                  MergePaths({2 * i, (fwd_type)1}, p_min_weight[TN0], p_min_dag, min_dag_sz, p_base_min_len);
              final_min_len[2 * i] = (final_min_path[2 * i].id < N2)
                                         ? p_base_min_len[final_min_path[2 * i].id]
                                         : p_min_dag[final_min_path[2 * i].id - N2].total_len;
            }

            int TN1 = init_t_min[2 * i + 1];
            if (TN1 != -1) {
              final_min_path[2 * i + 1] =
                  MergePaths({2 * i + 1, (fwd_type)1}, p_min_weight[TN1], p_min_dag, min_dag_sz, p_base_min_len);
              final_min_len[2 * i + 1] = (final_min_path[2 * i + 1].id < N2)
                                             ? p_base_min_len[final_min_path[2 * i + 1].id]
                                             : p_min_dag[final_min_path[2 * i + 1].id - N2].total_len;
            }
          }
          for (int i = 0; i < N2; ++i) final_min_offset[i + 1] = final_min_offset[i] + final_min_len[i];
        });
  }

  auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  // torch::Tensor new_flat_max = torch::empty({(long)final_max_offset.back()}, cpu_int_opts);
  // torch::Tensor new_flat_min = torch::empty({(long)final_min_offset.back()}, cpu_int_opts);

  auto new_flat_max_geom = ws.hlp.temp_flat_max.request({(long)final_max_offset.back()}, cpu_int_opts);
  auto new_flat_min_geom = ws.hlp.temp_flat_min.request({(long)final_min_offset.back()}, cpu_int_opts);

  int* out_max_ptr = new_flat_max_geom.template data_ptr<int>();
  int* out_min_ptr = new_flat_min_geom.template data_ptr<int>();
  auto flat_max_geom = sn.flat_max_geom.get();
  auto flat_min_geom = sn.flat_min_geom.get();
  const int* old_max_ptr = flat_max_geom.template data_ptr<int>();
  const int* old_min_ptr = flat_min_geom.template data_ptr<int>();

  struct StackItem {
    int id;
    bool fwd;
  };
  tbb::enumerable_thread_specific<std::vector<StackItem>> tls_stack;

  auto WriteGeomFlat = [&](RefType path, int* out_flat, int start_offset, const NodeType* dag, const int* old_flat,
                           const int* base_offsets, const int* base_lens) {
    if (path.id == -1) return;

    auto& stack_buf = tls_stack.local();
    stack_buf.clear();
    stack_buf.push_back({path.id, (bool)path.fwd});
    int write_head = start_offset;

    while (!stack_buf.empty()) {
      auto curr = stack_buf.back();
      stack_buf.pop_back();

      if (curr.id < N2) {
        int len = base_lens[curr.id];
        int off = base_offsets[curr.id];
        if (curr.fwd) {
          std::memcpy(&out_flat[write_head], &old_flat[off], len * sizeof(int));
          write_head += len;
        } else {
          for (int k = 0; k < len; ++k) out_flat[write_head++] = old_flat[off + len - 1 - k];
        }
      } else {
        const auto& node = dag[curr.id - N2];
        if (curr.fwd) {
          stack_buf.push_back({node.right_id, (bool)node.right_fwd});
          stack_buf.push_back({node.left_id, (bool)node.left_fwd});
        } else {
          // Explicit bool casts handle integer negation (!1 = 0) perfectly for Metal structs
          stack_buf.push_back({node.left_id, (bool)(!node.left_fwd)});
          stack_buf.push_back({node.right_id, (bool)(!node.right_fwd)});
        }
      }
    }
  };

  {
    RECORD_FUNCTION("Flat_Arena_Assembly", {});
    at::parallel_for(0, num_saddles, 256, [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; ++i) {
        int idx = static_cast<int>(i);
        auto globally_alive = max_alive[idx] && min_alive[idx];
        sn.nodes[idx].alive = globally_alive;

        if (!globally_alive) {
          sn.nodes[idx].max_arcs[0].length = 0;
          sn.nodes[idx].max_arcs[1].length = 0;
          sn.nodes[idx].min_arcs[0].length = 0;
          sn.nodes[idx].min_arcs[1].length = 0;
          continue;
        }

        int TM0 = init_t_max[2 * idx];
        if (TM0 != -1) {
          sn.nodes[idx].max_arcs[0].target = p_max_parent[TM0];
          sn.nodes[idx].max_arcs[0].offset = final_max_offset[2 * idx];
          sn.nodes[idx].max_arcs[0].length = final_max_len[2 * idx];
          WriteGeomFlat(final_max_path[2 * idx], out_max_ptr, final_max_offset[2 * idx], p_max_dag, old_max_ptr,
                        base_max_offset.data(), p_base_max_len);
        }

        int TM1 = init_t_max[2 * idx + 1];
        if (TM1 != -1) {
          sn.nodes[idx].max_arcs[1].target = p_max_parent[TM1];
          sn.nodes[idx].max_arcs[1].offset = final_max_offset[2 * idx + 1];
          sn.nodes[idx].max_arcs[1].length = final_max_len[2 * idx + 1];
          WriteGeomFlat(final_max_path[2 * idx + 1], out_max_ptr, final_max_offset[2 * idx + 1], p_max_dag, old_max_ptr,
                        base_max_offset.data(), p_base_max_len);
        }

        int TN0 = init_t_min[2 * idx];
        if (TN0 != -1) {
          sn.nodes[idx].min_arcs[0].target = p_min_parent[TN0];
          sn.nodes[idx].min_arcs[0].offset = final_min_offset[2 * idx];
          sn.nodes[idx].min_arcs[0].length = final_min_len[2 * idx];
          WriteGeomFlat(final_min_path[2 * idx], out_min_ptr, final_min_offset[2 * idx], p_min_dag, old_min_ptr,
                        base_min_offset.data(), p_base_min_len);
        }

        int TN1 = init_t_min[2 * idx + 1];
        if (TN1 != -1) {
          sn.nodes[idx].min_arcs[1].target = p_min_parent[TN1];
          sn.nodes[idx].min_arcs[1].offset = final_min_offset[2 * idx + 1];
          sn.nodes[idx].min_arcs[1].length = final_min_len[2 * idx + 1];
          WriteGeomFlat(final_min_path[2 * idx + 1], out_min_ptr, final_min_offset[2 * idx + 1], p_min_dag, old_min_ptr,
                        base_min_offset.data(), p_base_min_len);
        }
      }
    });
  }

  sn.flat_max_geom.copy_from_tensor(new_flat_max_geom);
  sn.flat_min_geom.copy_from_tensor(new_flat_min_geom);
}

template <typename Workspace>
void simplify_arcs_geometry(Workspace& ws, bool trace_max_arcs, bool trace_min_arcs) {
  RECORD_FUNCTION("simplify_arcs_geometry_flat", {});
  auto& min_cancellations = ws.p_data.min_cancellations;
  auto& max_cancellations = ws.p_data.max_cancellations;
  auto& sn = ws.saddle_nodes;
  int num_saddles = sn.nodes.size();
  int N2 = num_saddles * 2;
  int num_crit_maxes = ws.gradient_data.cp.maxes.size();
  int num_crit_mins = ws.gradient_data.cp.mins.size();

  std::vector<int> max_parent(num_crit_maxes), min_parent(num_crit_mins);
  std::vector<PathRef> max_weight(num_crit_maxes, {-1, true}), min_weight(num_crit_mins, {-1, true});

  for (int i = 0; i < num_crit_maxes; ++i) max_parent[i] = i;
  for (int i = 0; i < num_crit_mins; ++i) min_parent[i] = i;

  std::vector<int> base_max_len(N2, 0), base_min_len(N2, 0);
  std::vector<int> base_max_offset(N2, 0), base_min_offset(N2, 0);
  std::vector<int> init_t_max(N2, -1), init_t_min(N2, -1);

  // Independent survival trackers
  std::vector<uint8_t> max_alive(num_saddles, 1);
  std::vector<uint8_t> min_alive(num_saddles, 1);

  // n.b.: careful here, this is a heuristic
  size_t safe_max_dag = std::max(MIN_DAG_ALLOC_TOTAL, max_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR);
  size_t safe_min_dag = std::max(MIN_DAG_ALLOC_TOTAL, min_cancellations.size() * MAX_DAG_ALLOC_PER_PAIR);
  std::vector<DAGNode> max_dag(safe_max_dag);
  std::vector<DAGNode> min_dag(safe_min_dag);
  int max_dag_sz = 0;
  int min_dag_sz = 0;

  {
    RECORD_FUNCTION("Extract_Base_Data", {});
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

  int* p_max_parent = max_parent.data();
  int* p_min_parent = min_parent.data();
  PathRef* p_max_weight = max_weight.data();
  PathRef* p_min_weight = min_weight.data();
  DAGNode* p_max_dag = max_dag.data();
  DAGNode* p_min_dag = min_dag.data();
  const int* p_base_max_len = base_max_len.data();
  const int* p_base_min_len = base_min_len.data();

  auto MergePaths = [&](PathRef p1, PathRef p2, DAGNode* dag, int& dag_sz, const int* base_lens) -> PathRef {
    if (p1.id == -1) return p2;
    if (p2.id == -1) return p1;

    int len1 = (p1.id < N2) ? base_lens[p1.id] : dag[p1.id - N2].total_len;
    int len2 = (p2.id < N2) ? base_lens[p2.id] : dag[p2.id - N2].total_len;
    int new_id = dag_sz + N2;
    dag[dag_sz++] = {p1.id, p2.id, p1.fwd, p2.fwd, len1 + len2};
    return {new_id, true};
  };

  auto Find = [&](auto& self, int i, int* parent, PathRef* weight, DAGNode* dag, int& dag_sz, const int* blen) -> int {
    if (i == -1) return -1;
    if (parent[i] == i) return i;

    int p = parent[i];
    int root = self(self, p, parent, weight, dag, dag_sz, blen);

    if (p != -1 && weight[p].id != -1) {
      weight[i] = MergePaths(weight[i], weight[p], dag, dag_sz, blen);
    }
    parent[i] = root;
    return root;
  };

  {
    RECORD_FUNCTION("Decoupled_Parallel_Contraction", {});

    tbb::parallel_invoke(
        // MAXIMA
        [&]() {
          if (!trace_max_arcs) return;
          tbb::parallel_sort(max_cancellations.begin(), max_cancellations.end());

          for (const auto& cancel : max_cancellations) {
            int s = cancel.s_idx;
            if (!max_alive[s]) continue;

            int dead = cancel.t_idx;
            if (dead < 0 || dead >= num_crit_maxes) continue;

            int T0 = init_t_max[2 * s], T1 = init_t_max[2 * s + 1];
            int R0 = Find(Find, T0, p_max_parent, p_max_weight, p_max_dag, max_dag_sz, p_base_max_len);
            int R1 = Find(Find, T1, p_max_parent, p_max_weight, p_max_dag, max_dag_sz, p_base_max_len);

            PathRef wT0 = (T0 == -1) ? PathRef{-1, true} : p_max_weight[T0];
            PathRef wT1 = (T1 == -1) ? PathRef{-1, true} : p_max_weight[T1];

            PathRef Full0 = MergePaths({2 * s, true}, wT0, p_max_dag, max_dag_sz, p_base_max_len);
            PathRef Full1 = MergePaths({2 * s + 1, true}, wT1, p_max_dag, max_dag_sz, p_base_max_len);

            bool merged = false;
            if (R0 == dead && R0 != R1) {
              p_max_parent[R0] = R1;
              p_max_weight[R0] =
                  MergePaths({Full0.id, (bool)(!Full0.fwd)}, Full1, p_max_dag, max_dag_sz, p_base_max_len);
              merged = true;
            } else if (R1 == dead && R0 != R1) {
              p_max_parent[R1] = R0;
              p_max_weight[R1] =
                  MergePaths({Full1.id, (bool)(!Full1.fwd)}, Full0, p_max_dag, max_dag_sz, p_base_max_len);
              merged = true;
            }

            if (merged) max_alive[s] = 0;
          }

          for (int i = 0; i < num_crit_maxes; ++i) {
            Find(Find, i, p_max_parent, p_max_weight, p_max_dag, max_dag_sz, p_base_max_len);
          }
        },

        // MINIMA
        [&]() {
          if (!trace_min_arcs) return;
          tbb::parallel_sort(min_cancellations.begin(), min_cancellations.end());

          for (const auto& cancel : min_cancellations) {
            int s = cancel.s_idx;
            if (!min_alive[s]) continue;

            int dead = cancel.t_idx;
            if (dead < 0 || dead >= num_crit_mins) continue;

            int T0 = init_t_min[2 * s], T1 = init_t_min[2 * s + 1];
            int R0 = Find(Find, T0, p_min_parent, p_min_weight, p_min_dag, min_dag_sz, p_base_min_len);
            int R1 = Find(Find, T1, p_min_parent, p_min_weight, p_min_dag, min_dag_sz, p_base_min_len);

            PathRef wT0 = (T0 == -1) ? PathRef{-1, true} : p_min_weight[T0];
            PathRef wT1 = (T1 == -1) ? PathRef{-1, true} : p_min_weight[T1];

            PathRef Full0 = MergePaths({2 * s, true}, wT0, p_min_dag, min_dag_sz, p_base_min_len);
            PathRef Full1 = MergePaths({2 * s + 1, true}, wT1, p_min_dag, min_dag_sz, p_base_min_len);

            bool merged = false;
            if (R0 == dead && R0 != R1) {
              p_min_parent[R0] = R1;
              p_min_weight[R0] =
                  MergePaths({Full0.id, (bool)(!Full0.fwd)}, Full1, p_min_dag, min_dag_sz, p_base_min_len);
              merged = true;
            } else if (R1 == dead && R0 != R1) {
              p_min_parent[R1] = R0;
              p_min_weight[R1] =
                  MergePaths({Full1.id, (bool)(!Full1.fwd)}, Full0, p_min_dag, min_dag_sz, p_base_min_len);
              merged = true;
            }

            if (merged) min_alive[s] = 0;
          }

          for (int i = 0; i < num_crit_mins; ++i) {
            Find(Find, i, p_min_parent, p_min_weight, p_min_dag, min_dag_sz, p_base_min_len);
          }
        });
  }

  // Delegate the final memory assembly phase so we can reuse the code on GPU
  assemble_simplified_geometry<DAGNode, PathRef>(ws, max_alive, min_alive, init_t_max, init_t_min, base_max_len,
                                                 base_min_len, base_max_offset, base_min_offset, max_parent, min_parent,
                                                 max_weight, min_weight, max_dag, max_dag_sz, min_dag, min_dag_sz);
}
}  // namespace cpu