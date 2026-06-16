#pragma once

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <stdio.h>
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

void simplify_arcs_geometry_test(SaddleNodes& sn, int num_crit_maxes, int num_crit_mins,
                                 std::vector<CancelEvent>& min_cancellations,
                                 std::vector<CancelEvent>& max_cancellations) {
  RECORD_FUNCTION("simplify_arcs_geometry_flat", {});

  int num_saddles = sn.saddle_nodes.size();
  int N2 = num_saddles * 2;

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

  size_t safe_max_dag = std::max(1024UL, max_cancellations.size() * 16);
  size_t safe_min_dag = std::max(1024UL, min_cancellations.size() * 16);
  std::vector<DAGNode> max_dag(safe_max_dag);
  std::vector<DAGNode> min_dag(safe_min_dag);
  int max_dag_sz = 0;
  int min_dag_sz = 0;

  {
    RECORD_FUNCTION("Extract_Base_Data", {});
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
    RECORD_FUNCTION("Iterative_Parallel_Contraction", {});

    // FIX: Bounded, non-recursive Read-Only Find. Protects against -1 out-of-bounds boundary access.
    auto ReadOnlyFind = [](int i, const int* parent) -> int {
      int curr = i;
      while (curr != -1 && parent[curr] != curr) {
        curr = parent[curr];
      }
      return curr;
    };

    tbb::parallel_invoke(
        // ==========================================
        // MAXIMA
        // ==========================================
        [&]() {
          tbb::parallel_sort(max_cancellations.begin(), max_cancellations.end());

          int num_cancels = max_cancellations.size();
          std::vector<uint8_t> pending(num_cancels, 1);
          int total_cancelled = 0;
          int iteration = 0;

          while (total_cancelled < num_cancels) {
            std::vector<int> ready_to_cancel;

            // PHASE 1: EVALUATE (Strictly Read-Only, mimicking GPU thread reads)
            for (int i = 0; i < num_cancels; ++i) {
              if (!pending[i]) continue;

              const auto& cancel = max_cancellations[i];
              int s = cancel.s_idx;
              int dead = cancel.t_idx;

              if (!max_alive[s] || dead < 0 || dead >= num_crit_maxes) {
                pending[i] = 0;
                total_cancelled++;
                continue;
              }

              int T0 = init_t_max[2 * s], T1 = init_t_max[2 * s + 1];

              int R0 = ReadOnlyFind(T0, p_max_parent);
              int R1 = ReadOnlyFind(T1, p_max_parent);

              if ((R0 == dead && R0 != R1) || (R1 == dead && R0 != R1)) {
                ready_to_cancel.push_back(i);
              }
            }

            // PHASE 2: APPLY MUTATIONS (Mimicking the write back)
            for (int i : ready_to_cancel) {
              const auto& cancel = max_cancellations[i];
              int s = cancel.s_idx;
              int dead = cancel.t_idx;

              int T0 = init_t_max[2 * s], T1 = init_t_max[2 * s + 1];
              int R0 = Find(Find, T0, p_max_parent, p_max_weight, p_max_dag, max_dag_sz, p_base_max_len);
              int R1 = Find(Find, T1, p_max_parent, p_max_weight, p_max_dag, max_dag_sz, p_base_max_len);

              if (R0 == dead && R0 != R1) {
                p_max_parent[R0] = R1;
                PathRef wT0 = (T0 == -1) ? PathRef{-1, true} : p_max_weight[T0];
                PathRef wT1 = (T1 == -1) ? PathRef{-1, true} : p_max_weight[T1];
                PathRef Full0 = MergePaths({2 * s, true}, wT0, p_max_dag, max_dag_sz, p_base_max_len);
                PathRef Full1 = MergePaths({2 * s + 1, true}, wT1, p_max_dag, max_dag_sz, p_base_max_len);
                p_max_weight[R0] =
                    MergePaths({Full0.id, (bool)(!Full0.fwd)}, Full1, p_max_dag, max_dag_sz, p_base_max_len);
                max_alive[s] = 0;
                pending[i] = 0;
                total_cancelled++;
              } else if (R1 == dead && R0 != R1) {
                p_max_parent[R1] = R0;
                PathRef wT0 = (T0 == -1) ? PathRef{-1, true} : p_max_weight[T0];
                PathRef wT1 = (T1 == -1) ? PathRef{-1, true} : p_max_weight[T1];
                PathRef Full0 = MergePaths({2 * s, true}, wT0, p_max_dag, max_dag_sz, p_base_max_len);
                PathRef Full1 = MergePaths({2 * s + 1, true}, wT1, p_max_dag, max_dag_sz, p_base_max_len);
                p_max_weight[R1] =
                    MergePaths({Full1.id, (bool)(!Full1.fwd)}, Full0, p_max_dag, max_dag_sz, p_base_max_len);
                max_alive[s] = 0;
                pending[i] = 0;
                total_cancelled++;
              }
            }

            printf("[Maxima] Iteration %d: Cancelled %zu pairs\n", iteration, ready_to_cancel.size());

            if (ready_to_cancel.empty() && total_cancelled < num_cancels) {
              printf("[Maxima] Graph locked! %d pairs left.\n", num_cancels - total_cancelled);
              break;
            }
            iteration++;
          }

          for (int i = 0; i < num_crit_maxes; ++i) {
            Find(Find, i, p_max_parent, p_max_weight, p_max_dag, max_dag_sz, p_base_max_len);
          }
        },

        // ==========================================
        // MINIMA
        // ==========================================
        [&]() {
          tbb::parallel_sort(min_cancellations.begin(), min_cancellations.end());

          int num_cancels = min_cancellations.size();
          std::vector<uint8_t> pending(num_cancels, 1);
          int total_cancelled = 0;
          int iteration = 0;

          while (total_cancelled < num_cancels) {
            std::vector<int> ready_to_cancel;

            // PHASE 1: EVALUATE (Strictly Read-Only, mimicking GPU thread reads)
            for (int i = 0; i < num_cancels; ++i) {
              if (!pending[i]) continue;

              const auto& cancel = min_cancellations[i];
              int s = cancel.s_idx;
              int dead = cancel.t_idx;

              if (!min_alive[s] || dead < 0 || dead >= num_crit_mins) {
                pending[i] = 0;
                total_cancelled++;
                continue;
              }

              int T0 = init_t_min[2 * s], T1 = init_t_min[2 * s + 1];

              int R0 = ReadOnlyFind(T0, p_min_parent);
              int R1 = ReadOnlyFind(T1, p_min_parent);

              if ((R0 == dead && R0 != R1) || (R1 == dead && R0 != R1)) {
                ready_to_cancel.push_back(i);
              }
            }

            // PHASE 2: APPLY MUTATIONS (Mimicking the write back)
            for (int i : ready_to_cancel) {
              const auto& cancel = min_cancellations[i];
              int s = cancel.s_idx;
              int dead = cancel.t_idx;

              int T0 = init_t_min[2 * s], T1 = init_t_min[2 * s + 1];
              int R0 = Find(Find, T0, p_min_parent, p_min_weight, p_min_dag, min_dag_sz, p_base_min_len);
              int R1 = Find(Find, T1, p_min_parent, p_min_weight, p_min_dag, min_dag_sz, p_base_min_len);

              if (R0 == dead && R0 != R1) {
                p_min_parent[R0] = R1;
                PathRef wT0 = (T0 == -1) ? PathRef{-1, true} : p_min_weight[T0];
                PathRef wT1 = (T1 == -1) ? PathRef{-1, true} : p_min_weight[T1];
                PathRef Full0 = MergePaths({2 * s, true}, wT0, p_min_dag, min_dag_sz, p_base_min_len);
                PathRef Full1 = MergePaths({2 * s + 1, true}, wT1, p_min_dag, min_dag_sz, p_base_min_len);
                p_min_weight[R0] =
                    MergePaths({Full0.id, (bool)(!Full0.fwd)}, Full1, p_min_dag, min_dag_sz, p_base_min_len);
                min_alive[s] = 0;
                pending[i] = 0;
                total_cancelled++;
              } else if (R1 == dead && R0 != R1) {
                p_min_parent[R1] = R0;
                PathRef wT0 = (T0 == -1) ? PathRef{-1, true} : p_min_weight[T0];
                PathRef wT1 = (T1 == -1) ? PathRef{-1, true} : p_min_weight[T1];
                PathRef Full0 = MergePaths({2 * s, true}, wT0, p_min_dag, min_dag_sz, p_base_min_len);
                PathRef Full1 = MergePaths({2 * s + 1, true}, wT1, p_min_dag, min_dag_sz, p_base_min_len);
                p_min_weight[R1] =
                    MergePaths({Full1.id, (bool)(!Full1.fwd)}, Full0, p_min_dag, min_dag_sz, p_base_min_len);
                min_alive[s] = 0;
                pending[i] = 0;
                total_cancelled++;
              }
            }

            printf("[Minima] Iteration %d: Cancelled %zu pairs\n", iteration, ready_to_cancel.size());

            if (ready_to_cancel.empty() && total_cancelled < num_cancels) {
              printf("[Minima] Graph locked! %d pairs left.\n", num_cancels - total_cancelled);
              break;
            }
            iteration++;
          }

          for (int i = 0; i < num_crit_mins; ++i) {
            Find(Find, i, p_min_parent, p_min_weight, p_min_dag, min_dag_sz, p_base_min_len);
          }
        });
  }

  std::vector<int> final_max_len(N2, 0), final_min_len(N2, 0);
  std::vector<int> final_max_offset(N2 + 1, 0), final_min_offset(N2 + 1, 0);
  std::vector<PathRef> final_max_path(N2, {-1, true});
  std::vector<PathRef> final_min_path(N2, {-1, true});

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
                  MergePaths({2 * i, true}, p_max_weight[TM0], p_max_dag, max_dag_sz, p_base_max_len);
              final_max_len[2 * i] = (final_max_path[2 * i].id < N2)
                                         ? p_base_max_len[final_max_path[2 * i].id]
                                         : p_max_dag[final_max_path[2 * i].id - N2].total_len;
            }

            int TM1 = init_t_max[2 * i + 1];
            if (TM1 != -1) {
              final_max_path[2 * i + 1] =
                  MergePaths({2 * i + 1, true}, p_max_weight[TM1], p_max_dag, max_dag_sz, p_base_max_len);
              final_max_len[2 * i + 1] = (final_max_path[2 * i + 1].id < N2)
                                             ? p_base_max_len[final_max_path[2 * i + 1].id]
                                             : p_max_dag[final_max_path[2 * i + 1].id - N2].total_len;
            }
          }

          for (int i = 0; i < N2; ++i) {
            final_max_offset[i + 1] = final_max_offset[i] + final_max_len[i];
          }
        },

        // Minima Resolution & Prefix Sum
        [&]() {
          for (int i = 0; i < num_saddles; ++i) {
            if (!(max_alive[i] && min_alive[i])) continue;

            int TN0 = init_t_min[2 * i];
            if (TN0 != -1) {
              final_min_path[2 * i] =
                  MergePaths({2 * i, true}, p_min_weight[TN0], p_min_dag, min_dag_sz, p_base_min_len);
              final_min_len[2 * i] = (final_min_path[2 * i].id < N2)
                                         ? p_base_min_len[final_min_path[2 * i].id]
                                         : p_min_dag[final_min_path[2 * i].id - N2].total_len;
            }

            int TN1 = init_t_min[2 * i + 1];
            if (TN1 != -1) {
              final_min_path[2 * i + 1] =
                  MergePaths({2 * i + 1, true}, p_min_weight[TN1], p_min_dag, min_dag_sz, p_base_min_len);
              final_min_len[2 * i + 1] = (final_min_path[2 * i + 1].id < N2)
                                             ? p_base_min_len[final_min_path[2 * i + 1].id]
                                             : p_min_dag[final_min_path[2 * i + 1].id - N2].total_len;
            }
          }

          for (int i = 0; i < N2; ++i) {
            final_min_offset[i + 1] = final_min_offset[i] + final_min_len[i];
          }
        });
  }

  auto cpu_int_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  torch::Tensor new_flat_max = torch::empty({(long)final_max_offset.back()}, cpu_int_opts);
  torch::Tensor new_flat_min = torch::empty({(long)final_min_offset.back()}, cpu_int_opts);

  int* out_max_ptr = new_flat_max.data_ptr<int>();
  int* out_min_ptr = new_flat_min.data_ptr<int>();

  const int* old_max_ptr = sn.flat_max_geom.data_ptr<int>();
  const int* old_min_ptr = sn.flat_min_geom.data_ptr<int>();

  struct StackItem {
    int id;
    bool fwd;
  };
  tbb::enumerable_thread_specific<std::vector<StackItem>> tls_stack;

  auto WriteGeomFlat = [&](PathRef path, int* out_flat, int start_offset, const DAGNode* dag, const int* old_flat,
                           const int* base_offsets, const int* base_lens) {
    if (path.id == -1) return;

    auto& stack_buf = tls_stack.local();
    stack_buf.clear();
    stack_buf.push_back({path.id, path.fwd});
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
          for (int k = 0; k < len; ++k) {
            out_flat[write_head++] = old_flat[off + len - 1 - k];
          }
        }
      } else {
        const auto& node = dag[curr.id - N2];
        if (curr.fwd) {
          stack_buf.push_back({node.right_id, node.right_fwd});
          stack_buf.push_back({node.left_id, node.left_fwd});
        } else {
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
        sn.saddle_nodes[idx].alive = globally_alive;

        if (!globally_alive) {
          sn.saddle_nodes[idx].max_arcs[0].length = 0;
          sn.saddle_nodes[idx].max_arcs[1].length = 0;
          sn.saddle_nodes[idx].min_arcs[0].length = 0;
          sn.saddle_nodes[idx].min_arcs[1].length = 0;
          continue;
        }

        int TM0 = init_t_max[2 * idx];
        if (TM0 != -1) {
          sn.saddle_nodes[idx].max_arcs[0].target = p_max_parent[TM0];
          sn.saddle_nodes[idx].max_arcs[0].offset = final_max_offset[2 * idx];
          sn.saddle_nodes[idx].max_arcs[0].length = final_max_len[2 * idx];
          WriteGeomFlat(final_max_path[2 * idx], out_max_ptr, final_max_offset[2 * idx], p_max_dag, old_max_ptr,
                        base_max_offset.data(), p_base_max_len);
        }

        int TM1 = init_t_max[2 * idx + 1];
        if (TM1 != -1) {
          sn.saddle_nodes[idx].max_arcs[1].target = p_max_parent[TM1];
          sn.saddle_nodes[idx].max_arcs[1].offset = final_max_offset[2 * idx + 1];
          sn.saddle_nodes[idx].max_arcs[1].length = final_max_len[2 * idx + 1];
          WriteGeomFlat(final_max_path[2 * idx + 1], out_max_ptr, final_max_offset[2 * idx + 1], p_max_dag, old_max_ptr,
                        base_max_offset.data(), p_base_max_len);
        }

        int TN0 = init_t_min[2 * idx];
        if (TN0 != -1) {
          sn.saddle_nodes[idx].min_arcs[0].target = p_min_parent[TN0];
          sn.saddle_nodes[idx].min_arcs[0].offset = final_min_offset[2 * idx];
          sn.saddle_nodes[idx].min_arcs[0].length = final_min_len[2 * idx];
          WriteGeomFlat(final_min_path[2 * idx], out_min_ptr, final_min_offset[2 * idx], p_min_dag, old_min_ptr,
                        base_min_offset.data(), p_base_min_len);
        }

        int TN1 = init_t_min[2 * idx + 1];
        if (TN1 != -1) {
          sn.saddle_nodes[idx].min_arcs[1].target = p_min_parent[TN1];
          sn.saddle_nodes[idx].min_arcs[1].offset = final_min_offset[2 * idx + 1];
          sn.saddle_nodes[idx].min_arcs[1].length = final_min_len[2 * idx + 1];
          WriteGeomFlat(final_min_path[2 * idx + 1], out_min_ptr, final_min_offset[2 * idx + 1], p_min_dag, old_min_ptr,
                        base_min_offset.data(), p_base_min_len);
        }
      }
    });
  }

  sn.flat_max_geom = new_flat_max;
  sn.flat_min_geom = new_flat_min;
}