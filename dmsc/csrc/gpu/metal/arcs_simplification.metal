#include <metal_stdlib>
using namespace metal;

// Same definition as csrc/gpu/arcs_simplification_struct.hxx
// we cannot include due to the way we compile kernels ...
struct DAGNode {
  int left_id;
  int right_id;
  int left_fwd;   // 1 for true, 0 for false
  int right_fwd;  // 1 for true, 0 for false
  int total_len;
  int _padding;
};

struct PathRef {
  int id;
  int fwd;  // 1 for true, 0 for false
};

// No path compression
inline int read_only_find(int i, const device int* parent) {
  int curr = i;
  // Walk the tree until we hit a root or the void
  while (curr != -1 && parent[curr] != curr) {
    curr = parent[curr];
  }
  return curr;
}

// Dynamically allocates a new DAG node safely across 10,000 threads
inline PathRef merge_paths(PathRef p1, PathRef p2, device DAGNode* dag, device atomic_int* dag_sz,
                           const device int* base_lens, int N2) {
  if (p1.id == -1) return p2;
  if (p2.id == -1) return p1;

  int len1 = (p1.id < N2) ? base_lens[p1.id] : dag[p1.id - N2].total_len;
  int len2 = (p2.id < N2) ? base_lens[p2.id] : dag[p2.id - N2].total_len;

  // ATOMIC ALLOCATOR: Grab the next available index in the DAG array instantly
  int alloc_idx = atomic_fetch_add_explicit(dag_sz, 1, memory_order_relaxed);
  int new_id = alloc_idx + N2;

  dag[alloc_idx].left_id = p1.id;
  dag[alloc_idx].right_id = p2.id;
  dag[alloc_idx].left_fwd = p1.fwd;
  dag[alloc_idx].right_fwd = p2.fwd;
  dag[alloc_idx].total_len = len1 + len2;

  PathRef out = {new_id, 1};  // 1 = true
  return out;
}

inline PathRef read_only_find_with_weight(int i, const device int* parent, const device PathRef* weights,
                                          device DAGNode* dag, device atomic_int* dag_sz, const device int* base_lens,
                                          int N2, int stop_node) {
  if (i == -1) return PathRef{-1, 1};

  PathRef acc = weights[i];
  int curr = parent[i];

  while (curr != -1 && parent[curr] != curr && curr != stop_node) {
    if (weights[curr].id != -1) {
      acc = merge_paths(acc, weights[curr], dag, dag_sz, base_lens, N2);
    }
    curr = parent[curr];
  }
  return acc;
}

// Evaluate (Read-Only)
kernel void evaluate_cancellations_metal(
    const device int* cancels [[buffer(0)]], const device int* init_t [[buffer(1)]],
    const device int* parent_ptrs [[buffer(2)]], const device uint8_t* alive [[buffer(3)]],
    const device uint8_t* pending [[buffer(4)]], device atomic_int* ready_count [[buffer(5)]],
    device int* ready_list [[buffer(6)]], device int* ready_R0 [[buffer(7)]], device int* ready_R1 [[buffer(8)]],
    constant int& num_cancels [[buffer(9)]], constant int& num_extrema [[buffer(10)]],
    uint id [[thread_position_in_grid]]) {
  if (id >= (uint)num_cancels) return;
  if (pending[id] == 0) return;

  int s = cancels[id * 2];
  int dead = cancels[id * 2 + 1];

  if (alive[s] == 0 || dead < 0 || dead >= num_extrema) return;

  int T0 = init_t[2 * s];
  int T1 = init_t[2 * s + 1];

  int R0 = read_only_find(T0, parent_ptrs);
  int R1 = read_only_find(T1, parent_ptrs);

  if ((R0 == dead && R0 != R1) || (R1 == dead && R0 != R1)) {
    int write_idx = atomic_fetch_add_explicit(ready_count, 1, memory_order_relaxed);
    ready_list[write_idx] = id;
    ready_R0[write_idx] = R0;
    ready_R1[write_idx] = R1;
  }
}

kernel void contract_cancellations_metal(
    const device int* ready_list [[buffer(0)]], const device int* ready_R0 [[buffer(1)]],
    const device int* ready_R1 [[buffer(2)]], const device int* cancels [[buffer(3)]],
    const device int* init_t [[buffer(4)]], const device int* base_lens [[buffer(5)]],
    device int* parent_ptrs [[buffer(6)]], device PathRef* weights [[buffer(7)]], device DAGNode* dag [[buffer(8)]],
    device atomic_int* dag_sz [[buffer(9)]], device uint8_t* alive [[buffer(10)]],
    device uint8_t* pending [[buffer(11)]], const device atomic_int* ready_count [[buffer(12)]],
    constant int& N2 [[buffer(13)]],
    uint id [[thread_position_in_grid]]) {
  int num_ready = atomic_load_explicit(ready_count, memory_order_relaxed);
  if (id >= (uint)num_ready) return;

  int cancel_idx = ready_list[id];
  int s = cancels[cancel_idx * 2];
  int dead = cancels[cancel_idx * 2 + 1];

  int T0 = init_t[2 * s];
  int T1 = init_t[2 * s + 1];

  int R0 = ready_R0[id];
  int R1 = ready_R1[id];

  if (R0 == dead && R0 != R1) {
    PathRef wT0 = (T0 == -1 || T0 == R0) ? PathRef{-1, 1} : weights[T0];
    PathRef wT1 = (T1 == -1 || T1 == R1) ? PathRef{-1, 1} : weights[T1];

    PathRef Full0 = merge_paths({2 * s, 1}, wT0, dag, dag_sz, base_lens, N2);
    PathRef Full1 = merge_paths({2 * s + 1, 1}, wT1, dag, dag_sz, base_lens, N2);
    weights[R0] = merge_paths({Full0.id, Full0.fwd == 0 ? 1 : 0}, Full1, dag, dag_sz, base_lens, N2);

    parent_ptrs[R0] = R1;

    alive[s] = 0;
    pending[cancel_idx] = 0;

  } else if (R1 == dead && R0 != R1) {
    PathRef wT0 = (T0 == -1 || T0 == R0) ? PathRef{-1, 1} : weights[T0];
    PathRef wT1 = (T1 == -1 || T1 == R1) ? PathRef{-1, 1} : weights[T1];

    PathRef Full0 = merge_paths({2 * s, 1}, wT0, dag, dag_sz, base_lens, N2);
    PathRef Full1 = merge_paths({2 * s + 1, 1}, wT1, dag, dag_sz, base_lens, N2);
    weights[R1] = merge_paths({Full1.id, Full1.fwd == 0 ? 1 : 0}, Full0, dag, dag_sz, base_lens, N2);

    parent_ptrs[R1] = R0;

    alive[s] = 0;
    pending[cancel_idx] = 0;
  }
}

// Compress Paths Step
kernel void compress_paths_step_metal(const device int* parent_in [[buffer(0)]],
                                      const device PathRef* weights_in [[buffer(1)]],
                                      device int* parent_out [[buffer(2)]], device PathRef* weights_out [[buffer(3)]],
                                      device DAGNode* dag [[buffer(4)]], device atomic_int* dag_sz [[buffer(5)]],
                                      const device int* base_lens [[buffer(6)]],
                                      device atomic_int* changed [[buffer(7)]], constant int& num_extrema [[buffer(8)]],
                                      constant int& N2 [[buffer(9)]], uint id [[thread_position_in_grid]]) {
  if (id >= (uint)num_extrema) return;

  int p = parent_in[id];

  if (p != id && p != -1) {
    int pp = parent_in[p];
    if (pp != p && pp != -1) {
      PathRef wp = weights_in[p];
      PathRef w_new = merge_paths(weights_in[id], wp, dag, dag_sz, base_lens, N2);

      parent_out[id] = pp;
      weights_out[id] = w_new;
      atomic_store_explicit(changed, 1, memory_order_relaxed);
      return;
    }
  }

  parent_out[id] = p;
  weights_out[id] = weights_in[id];
}
