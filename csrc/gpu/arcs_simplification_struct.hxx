#pragma once

namespace gpu {
// Data structures need to be compatible with csrc/cpu/arcs_simplification.hxx
struct GPUDAGNode {
  int left_id;
  int right_id;
  int left_fwd;   // 1 for true, 0 for false
  int right_fwd;  // 1 for true, 0 for false
  int total_len;
  int _padding;
};

struct GPUPathRef {
  int id;
  int fwd;  // 1 for true, 0 for false
};
}