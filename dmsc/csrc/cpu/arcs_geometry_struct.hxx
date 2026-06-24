#pragma once

#include <torch/extension.h>

#include <vector>

#include "../managed_tensor.hxx"

namespace cpu {
struct Arc {
  int target;  // index of the connected max/min
  int offset;  // starting index in the flat geometry array
  int length;  // number of cells forming the manifold
};

// Store arcs incident to a saddle point
struct SaddleNode {
  int alive;  // int not bool for alignment
  Arc max_arcs[2];
  Arc min_arcs[2];
};

struct SaddleNodes {
  std::vector<SaddleNode> nodes;
  ManagedTensor flat_max_geom;
  ManagedTensor flat_min_geom;

  void reset() {
    // nodes.clear();
  }
};
}  // namespace cpu