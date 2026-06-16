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

// ==========================================
// 3. KERNEL 1: Evaluate (Read-Only)
// ==========================================
kernel void evaluate_cancellations_metal(const device int* cancels [[buffer(0)]],  // Flattened [N, 2]
                                         const device int* init_t [[buffer(1)]],
                                         const device int* parent_ptrs [[buffer(2)]],
                                         const device uint8_t* alive [[buffer(3)]],
                                         const device uint8_t* pending [[buffer(4)]],
                                         device atomic_int* ready_count [[buffer(5)]],
                                         device int* ready_list [[buffer(6)]], constant int& num_cancels [[buffer(7)]],
                                         constant int& num_extrema [[buffer(8)]], uint id [[thread_position_in_grid]]) {
  if (id >= (uint)num_cancels) return;
  if (pending[id] == 0) return;

  int s = cancels[id * 2];
  int dead = cancels[id * 2 + 1];

  if (alive[s] == 0 || dead < 0 || dead >= num_extrema) return;

  int T0 = init_t[2 * s];
  int T1 = init_t[2 * s + 1];

  // Find current topological roots
  int R0 = read_only_find(T0, parent_ptrs);
  int R1 = read_only_find(T1, parent_ptrs);

  // Are they immediate neighbors mathematically?
  if ((R0 == dead && R0 != R1) || (R1 == dead && R0 != R1)) {
    // ATOMIC PUSH_BACK: Add our index to the ready queue
    int write_idx = atomic_fetch_add_explicit(ready_count, 1, memory_order_relaxed);
    ready_list[write_idx] = id;
  }
}

// ==========================================
// 4. KERNEL 2: Contract (Write-Only)
// ==========================================
kernel void contract_cancellations_metal(
    const device int* ready_list [[buffer(0)]], const device int* cancels [[buffer(1)]],
    const device int* init_t [[buffer(2)]], const device int* base_lens [[buffer(3)]],
    device int* parent_ptrs [[buffer(4)]], device PathRef* weights [[buffer(5)]], device DAGNode* dag [[buffer(6)]],
    device atomic_int* dag_sz [[buffer(7)]], device uint8_t* alive [[buffer(8)]], device uint8_t* pending [[buffer(9)]],
    constant int& num_ready [[buffer(10)]], constant int& N2 [[buffer(11)]], uint id [[thread_position_in_grid]]) {
  if (id >= (uint)num_ready) return;

  int cancel_idx = ready_list[id];
  int s = cancels[cancel_idx * 2];
  int dead = cancels[cancel_idx * 2 + 1];

  int T0 = init_t[2 * s];
  int T1 = init_t[2 * s + 1];

  // We do NOT use path compression here to avoid memory races.
  // We already proved they are 1 hop away in Kernel 1.
  int R0 = read_only_find(T0, parent_ptrs);
  int R1 = read_only_find(T1, parent_ptrs);

  PathRef wT0 = (T0 == -1) ? PathRef{-1, 1} : weights[T0];
  PathRef wT1 = (T1 == -1) ? PathRef{-1, 1} : weights[T1];

  if (R0 == dead && R0 != R1) {
    parent_ptrs[R0] = R1;

    PathRef Full0 = merge_paths({2 * s, 1}, wT0, dag, dag_sz, base_lens, N2);
    PathRef Full1 = merge_paths({2 * s + 1, 1}, wT1, dag, dag_sz, base_lens, N2);
    weights[R0] = merge_paths({Full0.id, Full0.fwd == 0 ? 1 : 0}, Full1, dag, dag_sz, base_lens, N2);

    alive[s] = 0;
    pending[cancel_idx] = 0;

  } else if (R1 == dead && R0 != R1) {
    parent_ptrs[R1] = R0;

    PathRef Full0 = merge_paths({2 * s, 1}, wT0, dag, dag_sz, base_lens, N2);
    PathRef Full1 = merge_paths({2 * s + 1, 1}, wT1, dag, dag_sz, base_lens, N2);
    weights[R1] = merge_paths({Full1.id, Full1.fwd == 0 ? 1 : 0}, Full0, dag, dag_sz, base_lens, N2);

    alive[s] = 0;
    pending[cancel_idx] = 0;
  }
}